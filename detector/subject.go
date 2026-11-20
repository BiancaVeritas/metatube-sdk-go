package detector

import (
	"image"
	"math"

	"github.com/disintegration/imaging"

	"github.com/metatube-community/metatube-sdk-go/detector/onnx"
)

// Gender and main-subject selection, following the reference implementation:
// crop around the box at a fixed margin, run the age/gender head, and prefer a
// female face when one is found.

// isFemale reports whether the crop around box looks female.
//
// The crop is taken at the box centre with genderMargin times the box's long
// edge, which is what the attribute model was trained on; sampling the bare box
// loses the hair and shoulders it uses.
func isFemale(gender *onnx.Model, img image.Image, box faceBox) (female bool, ok bool) {
	if gender == nil || box.width() <= 0 || box.height() <= 0 {
		return false, false
	}
	cx := (box.X1 + box.X2) / 2
	cy := (box.Y1 + box.Y2) / 2
	scale := genderSide / (math.Max(box.width(), box.height()) * genderMargin)

	crop := affineCenterCrop(img, cx, cy, scale, genderSide)
	if crop == nil {
		return false, false
	}

	tensor := onnx.NewTensor(1, 3, genderSide, genderSide)
	for i := 0; i < genderSide*genderSide; i++ {
		// The head expects BGR with (x-127.5)/128, matching cv2's blobFromImage.
		r, g, b := crop[i*3+0], crop[i*3+1], crop[i*3+2]
		tensor.Data[0*genderSide*genderSide+i] = (float32(b) - 127.5) / 128
		tensor.Data[1*genderSide*genderSide+i] = (float32(g) - 127.5) / 128
		tensor.Data[2*genderSide*genderSide+i] = (float32(r) - 127.5) / 128
	}

	out, err := gender.Run(tensor)
	if err != nil {
		return false, false
	}
	for _, t := range out {
		if t.Len() < 3 {
			continue
		}
		best, bestIdx := t.Data[0], 0
		for i := 1; i < 3; i++ {
			if t.Data[i] > best {
				best, bestIdx = t.Data[i], i
			}
		}
		return bestIdx == genderFemale, true
	}
	return false, false
}

// affineCenterCrop samples a size×size RGB patch centred on (cx, cy).
//
// It reproduces cv2.warpAffine with [[s,0,-cx*s+size/2],[0,s,-cy*s+size/2]]:
// nearest pixel, zero outside the image. Pulling in OpenCV just for this is not
// worth a build dependency.
func affineCenterCrop(img image.Image, cx, cy, scale float64, size int) []uint8 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 || scale <= 0 {
		return nil
	}
	out := make([]uint8, size*size*3)
	for ty := 0; ty < size; ty++ {
		sy := (float64(ty)+0.5-float64(size)/2)/scale + cy
		iy := int(math.Round(sy))
		for tx := 0; tx < size; tx++ {
			sx := (float64(tx)+0.5-float64(size)/2)/scale + cx
			ix := int(math.Round(sx))
			if ix < 0 || ix >= w || iy < 0 || iy >= h {
				continue // zero-filled, like the padded ROI
			}
			r, g, b, _ := img.At(bounds.Min.X+ix, bounds.Min.Y+iy).RGBA()
			base := (ty*size + tx) * 3
			out[base+0] = uint8(r >> 8)
			out[base+1] = uint8(g >> 8)
			out[base+2] = uint8(b >> 8)
		}
	}
	return out
}

// subjectScore ranks faces by how likely they are the cover's main subject.
//
// Area alone is not enough: a profile shot has a tall, narrow box that can still
// be large while showing only an outline, so aspect ratio and detector
// confidence enter the score too.
func subjectScore(f faceBox) float64 {
	w, h := f.width(), f.height()
	if w <= 0 || h <= 0 {
		return 0
	}
	aspect := math.Min(w, h) / math.Max(w, h)
	// Below 0.6 the box is a profile or partly occluded; above it the score ramps
	// linearly to 1.
	frontal := 0.35 + 0.65*math.Min(1, math.Max(0, (aspect-0.6)/0.35))
	return w * h * f.Score * frontal
}

// detectFaces runs SCRFD over the image and returns boxes in source pixels.
func runSCRFD(img image.Image, conf float64) []faceBox {
	loaded := loadModels()
	if loaded.err != nil || loaded.detector == nil {
		return nil
	}
	tensor, scale, srcW, srcH := scrfdInput(img)
	if tensor == nil {
		return nil
	}
	out, err := loaded.detector.Run(tensor)
	if err != nil {
		return nil
	}
	boxes := loaded.decoder.decode(
		out, loaded.detector.OutputNames(),
		tensor.Shape[3], tensor.Shape[2], // inputW, inputH
		scale, srcW, srcH,
	)
	kept := boxes[:0]
	for _, box := range boxes {
		if box.Score >= conf {
			kept = append(kept, box)
		}
	}
	return kept
}

// resizedForDetection shrinks very large images before detection, the way the
// reference implementation caps the working size.
func resizedForDetection(img image.Image, maxSide int) image.Image {
	bounds := img.Bounds()
	if maxSide <= 0 || max(bounds.Dx(), bounds.Dy()) <= maxSide {
		return img
	}
	return imaging.Resize(img, maxSide, 0, imaging.Linear)
}

