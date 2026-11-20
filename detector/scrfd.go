package detector

import (
	"embed"
	"fmt"
	"image"
	"math"
	"sort"
	"sync"

	"github.com/disintegration/imaging"

	"github.com/metatube-community/metatube-sdk-go/detector/onnx"
)

// The detector runs the same models the reference implementation uses, so a face
// found here is the face found there.
//
//go:embed models/face_detect_scrfd.onnx models/face_detect_genderage.onnx
var models embed.FS

const (
	// scrfdSide is the padded square side fed to SCRFD; the reference implementation
	// scales the long edge to this and pads the rest with zeros.
	scrfdSide = 640
	// genderSide is the gender net's fixed input.
	genderSide = 96
	// sizeMultiple is SCRFD's constraint: width and height are multiples of 32.
	sizeMultiple = 32
	// genderMargin is how much wider than the face box the gender crop is taken,
	// matching the upstream attribute model.
	genderMargin = 1.5
	// nmsIOU is the overlap above which a second box is dropped.
	nmsIOU = 0.4
	// scrfdConfidence is the score floor used inside the decoder. It is a coarse
	// cut only; the caller applies the configured threshold afterwards.
	scrfdConfidence = 0.05
	// genderFemale and genderMale are the two logits of the 3-way head; the third
	// output is the age estimate and carries no gender meaning.
	genderFemale = 0
	genderMale   = 1
)

// faceBox is a detected face in source-image pixels.
type faceBox struct {
	X1, Y1, X2, Y2 float64
	// Score is the detector's confidence.
	Score float64
}

func (f faceBox) width() float64  { return f.X2 - f.X1 }
func (f faceBox) height() float64 { return f.Y2 - f.Y1 }
func (f faceBox) area() float64   { return f.width() * f.height() }

// scrfdDecoder turns the graph's nine outputs into boxes.
//
// Each stride contributes three tensors: scores, box distances and keypoints.
// The distances are relative to anchor centres, so the anchor grid has to be
// rebuilt exactly as the model expects — treating them as coordinates collapses
// every box into the top-left corner.
type scrfdDecoder struct {
	strides    []int
	numAnchors int
}

// newScrfdDecoder picks the stride grouping from the output count. The shipped
// model has 9 outputs: three strides, two anchors each.
func newScrfdDecoder(outputs int) (scrfdDecoder, error) {
	switch outputs {
	case 9, 6:
		return scrfdDecoder{strides: []int{8, 16, 32}, numAnchors: 2}, nil
	case 15, 10:
		return scrfdDecoder{strides: []int{8, 16, 32, 64, 128}, numAnchors: 1}, nil
	default:
		return scrfdDecoder{}, fmt.Errorf("detector: unexpected SCRFD output count %d", outputs)
	}
}

// anchorCenters rebuilds the grid the model was trained against.
func anchorCenters(height, width, stride, numAnchors int) [][2]float32 {
	centers := make([][2]float32, 0, height*width*numAnchors)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			for a := 0; a < numAnchors; a++ {
				centers = append(centers, [2]float32{float32(x * stride), float32(y * stride)})
			}
		}
	}
	return centers
}

// decode turns one inference pass into boxes in source-image coordinates.
//
// The anchor grid has to be rebuilt from the size of the tensor actually fed in:
// the network is fully convolutional, so the grid is inputW/stride × inputH/stride
// and not a fixed 640/stride. Sizing it from a constant silently mismatches every
// head and drops all detections on the floor.
func (d scrfdDecoder) decode(outputs map[string]*onnx.Tensor, names []string, inputW, inputH int, scale float64, srcW, srcH int) []faceBox {
	var boxes []faceBox
	// names are in graph order: all score heads, then all box heads, then all
	// keypoint heads, one per stride.
	per := len(d.strides)
	for i, stride := range d.strides {
		score := outputs[names[i]]
		reg := outputs[names[per+i]]
		if score == nil || reg == nil {
			continue
		}
		gridH, gridW := inputH/stride, inputW/stride
		centers := anchorCenters(gridH, gridW, stride, d.numAnchors)
		n := len(score.Data)
		if n != len(centers) || len(reg.Data) != n*4 {
			continue
		}
		for j := 0; j < n; j++ {
			s := score.Data[j]
			if s < scrfdConfidence {
				continue
			}
			dx1 := reg.Data[j*4+0] * float32(stride)
			dy1 := reg.Data[j*4+1] * float32(stride)
			dx2 := reg.Data[j*4+2] * float32(stride)
			dy2 := reg.Data[j*4+3] * float32(stride)

			x1 := (centers[j][0] - dx1) / float32(scale)
			y1 := (centers[j][1] - dy1) / float32(scale)
			x2 := (centers[j][0] + dx2) / float32(scale)
			y2 := (centers[j][1] + dy2) / float32(scale)

			boxes = append(boxes, faceBox{
				X1:    clampFloat(float64(x1), 0, float64(srcW)),
				Y1:    clampFloat(float64(y1), 0, float64(srcH)),
				X2:    clampFloat(float64(x2), 0, float64(srcW)),
				Y2:    clampFloat(float64(y2), 0, float64(srcH)),
				Score: float64(s),
			})
		}
	}
	return nms(boxes, nmsIOU)
}

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// nms drops overlapping boxes, keeping the highest-scoring one first.
func nms(boxes []faceBox, iou float64) []faceBox {
	sort.SliceStable(boxes, func(i, j int) bool { return boxes[i].Score > boxes[j].Score })
	keep := make([]faceBox, 0, len(boxes))
	for _, candidate := range boxes {
		drop := false
		for _, kept := range keep {
			if overlap(candidate, kept) > iou {
				drop = true
				break
			}
		}
		if !drop {
			keep = append(keep, candidate)
		}
	}
	return keep
}

func overlap(a, b faceBox) float64 {
	x1 := math.Max(a.X1, b.X1)
	y1 := math.Max(a.Y1, b.Y1)
	x2 := math.Min(a.X2, b.X2)
	y2 := math.Min(a.Y2, b.Y2)
	if x2 <= x1 || y2 <= y1 {
		return 0
	}
	inter := (x2 - x1) * (y2 - y1)
	union := a.area() + b.area() - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// modelSet holds both loaded graphs; sessions are built once and reused.
type modelSet struct {
	detector *onnx.Model
	gender   *onnx.Model
	decoder  scrfdDecoder
	err      error
}

var (
	modelsOnce sync.Once
	modelsLoad modelSet
)

// loadModels parses the embedded graphs on first use.
func loadModels() modelSet {
	modelsOnce.Do(func() {
		detectRaw, err := models.ReadFile("models/face_detect_scrfd.onnx")
		if err != nil {
			modelsLoad.err = err
			return
		}
		detector, err := onnx.Load(detectRaw)
		if err != nil {
			modelsLoad.err = err
			return
		}
		decoder, err := newScrfdDecoder(len(detector.OutputNames()))
		if err != nil {
			modelsLoad.err = err
			return
		}
		genderRaw, err := models.ReadFile("models/face_detect_genderage.onnx")
		if err != nil {
			modelsLoad.err = err
			return
		}
		gender, err := onnx.Load(genderRaw)
		if err != nil {
			modelsLoad.err = err
			return
		}
		modelsLoad = modelSet{detector: detector, gender: gender, decoder: decoder}
	})
	return modelsLoad
}

// scrfdInput letterboxes an image into the padded square SCRFD expects.
//
// The reference implementation scales the long edge to the side and fills the
// remainder with zeros, so the return value carries both the scale and the box
// the resized image occupies.
func scrfdInput(img image.Image) (*onnx.Tensor, float64, int, int) {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, 0, 0, 0
	}

	scale := math.Min(1, float64(scrfdSide)/float64(max(srcW, srcH)))
	newW := int(math.Round(float64(srcW) * scale))
	newH := int(math.Round(float64(srcH) * scale))
	inputW := max(sizeMultiple, (newW+sizeMultiple-1)/sizeMultiple*sizeMultiple)
	inputH := max(sizeMultiple, (newH+sizeMultiple-1)/sizeMultiple*sizeMultiple)

	resized := imaging.Resize(img, newW, newH, imaging.Linear)
	tensor := onnx.NewTensor(1, 3, inputH, inputW)
	// SCRFD wants (x-127.5)/128 in BGR order.
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			r, g, b, _ := resized.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			base := y*inputW + x
			tensor.Data[0*inputH*inputW+base] = (float32(b>>8) - 127.5) / 128
			tensor.Data[1*inputH*inputW+base] = (float32(g>>8) - 127.5) / 128
			tensor.Data[2*inputH*inputW+base] = (float32(r>>8) - 127.5) / 128
		}
	}
	return tensor, scale, srcW, srcH
}

