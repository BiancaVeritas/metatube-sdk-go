package detector

import (
	"image"
	"sort"

	"github.com/disintegration/imaging"
)

const (
	// maxImageWidth caps the working size: detection cost grows with the square of
	// the side, and a crop decision does not need more.
	maxImageWidth = 650
	// defaultConfidence is the score floor for a face. Sweeping thirty real covers
	// found every subject at this threshold with no "whole upper body" boxes.
	defaultConfidence = 0.45
)

// FindPrimaryFaceAxisRatio reports where the crop window should be centred along
// the dominant axis, as a fraction of the image's size.
//
// ratio is the crop window's shape: a value below 1 means the window is narrower
// than the image, so the axis is horizontal. The bool is false when no face is
// found, which leaves the caller on its default position.
func FindPrimaryFaceAxisRatio(img image.Image, ratio float64, advanced bool, debugs ...debugFunc) (float64, bool) {
	if img == nil {
		return 0, false
	}
	bounds := img.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return 0, false
	}

	working := resizedForDetection(img, maxImageWidth)
	faces := runSCRFD(working, defaultConfidence)
	if len(faces) == 0 && advanced {
		// A portrait cover can put the subject's face off-axis. Retrying on rotated
		// copies costs seconds only when the first pass found nothing.
		faces = detectRotated(working)
	}
	for _, fn := range debugs {
		fn(working, faces)
	}
	if len(faces) == 0 {
		return 0, false
	}

	// Scale the boxes back to the original image coordinates.
	scale := float64(bounds.Dx()) / float64(working.Bounds().Dx())

	face, ok := pickSubjectFace(working, faces)
	if !ok {
		return 0, false
	}
	cx := (face.X1 + face.X2) / 2 * scale

	axis := dominantAxis(bounds.Dx(), bounds.Dy(), ratio)
	switch axis {
	case 0:
		return clampFraction(cx / float64(bounds.Dx())), true
	default:
		cy := (face.Y1 + face.Y2) / 2 * scale
		return clampFraction(cy / float64(bounds.Dy())), true
	}
}

// dominantAxis picks the dimension the crop window slides along: a window
// narrower than the image slides horizontally, a shorter one slides vertically.
func dominantAxis(width, height int, ratio float64) int {
	if float64(height)*ratio < float64(width) {
		return 0
	}
	return 1
}

func clampFraction(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// pickSubjectFace chooses the face the cover is about.
//
// Faces are ranked by subjectScore and then filtered by gender: a cover often
// shows a male face larger and sharper than the actress, so picking on area
// alone lands on the wrong person. A failed gender pass falls back to the
// best-scoring face rather than failing the crop.
func pickSubjectFace(img image.Image, faces []faceBox) (faceBox, bool) {
	if len(faces) == 0 {
		return faceBox{}, false
	}
	ranked := append([]faceBox(nil), faces...)
	sort.SliceStable(ranked, func(i, j int) bool {
		return subjectScore(ranked[i]) > subjectScore(ranked[j])
	})

	loaded := loadModels()
	if loaded.gender != nil {
		for _, face := range ranked {
			if female, ok := isFemale(loaded.gender, img, face); ok && female {
				return face, true
			}
		}
	}
	return ranked[0], true
}

// detectRotated retries detection on the image turned by the angles a cover is
// printed at, then maps the boxes back.
func detectRotated(img image.Image) []faceBox {
	var found []faceBox
	for _, angle := range []float64{90, 270} {
		rotated := imaging.Rotate(img, angle, nil)
		for _, face := range runSCRFD(rotated, defaultConfidence) {
			found = append(found, unrotateBox(face, rotated.Bounds().Dx(), rotated.Bounds().Dy(), img.Bounds(), angle))
		}
	}
	return found
}

// unrotateBox maps a box found on a rotated copy back to the original frame.
func unrotateBox(box faceBox, rotW, rotH int, orig image.Rectangle, angle float64) faceBox {
	// Rotating by 90 puts the original top-left at the rotated top-right, and so
	// on; the mapping swaps the axes and mirrors one of them.
	var x1, y1, x2, y2 float64
	switch int(angle) % 360 {
	case 90:
		x1, y1 = box.Y1, float64(rotW)-box.X2
		x2, y2 = box.Y2, float64(rotW)-box.X1
	case 270:
		x1, y1 = float64(rotH)-box.Y2, box.X1
		x2, y2 = float64(rotH)-box.Y1, box.X2
	default:
		return box
	}
	return faceBox{
		X1: clampFloat(x1, 0, float64(orig.Dx())), Y1: clampFloat(y1, 0, float64(orig.Dy())),
		X2: clampFloat(x2, 0, float64(orig.Dx())), Y2: clampFloat(y2, 0, float64(orig.Dy())),
		Score: box.Score,
	}
}
