package detector

import (
	"image"
	"image/jpeg"
	"os"
)

// writeJPEG saves a debug rendering when a case misses.
func writeJPEG(path string, img image.Image) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return jpeg.Encode(file, img, &jpeg.Options{Quality: 80})
}
