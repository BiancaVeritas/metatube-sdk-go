package detector

import "image"

// debugFunc receives the working image and the faces found in it.
//
// It exists so callers can draw what the detector saw; the signature keeps the
// production path free of any debug plumbing.
type debugFunc func(image.Image, []faceBox)
