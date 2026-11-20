package detector

import (
	"bytes"
	"embed"
	"encoding/base64"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

/*
  # Generate MD5-hashed images:
  setopt null_glob
  for IMG in *.jpg *.jpeg *.png; do
    gbase64 "$IMG" > "$(md5sum "$IMG" | cut -d' ' -f1)"
  done

  # Decode image:
  gbase64 -d "$IMG"  > "$IMG".jpg
*/

//go:embed testdata/*
var fs embed.FS

const (
	xRatio = 0.0 // 2:3 (default)
	yRatio = 1.8 // 16:9
)

// TestDetectMainFacePosition pins the crop position on real covers.
//
// The expected values were recorded against the previous detector and describe
// where the subject's face sits, so they are a regression baseline for the SCRFD
// swap rather than a re-derivation.
func TestDetectMainFacePosition(t *testing.T) {
	for _, unit := range []struct {
		filename string
		position float64
		imgRatio float64
		advanced bool
	}{
		// X Ratio detection:
		{filename: "809ee47a17a7938ebd6d908244b962c8", position: 0.40, imgRatio: xRatio, advanced: true},
		{filename: "c7806a2f581012eb71dce597f682c7a2", position: 0.22, imgRatio: xRatio, advanced: true},
		{filename: "aa237b3a2bfd35dbe10c386c7ac777ae", position: 0.30, imgRatio: xRatio, advanced: true},
		{filename: "c2a6bf02b748bb460a0dbb550f39c635", position: 0.70, imgRatio: xRatio, advanced: true},
		{filename: "1d8aaf63245426c4a32720bdbf33a651", position: 0.80, imgRatio: xRatio, advanced: true},
		{filename: "e953fce3bf5ec5746ead8954bec758e0", position: 0.30, imgRatio: xRatio, advanced: true},
		{filename: "d054f170d52c83a773571675d954e3bb", position: 0.75, imgRatio: xRatio, advanced: true},
		{filename: "3685c2648be7eeeaa2cef0118873a55f", position: 0.60, imgRatio: xRatio, advanced: true},
		{filename: "7848e5995a58df9d063df8543c50c943", position: 0.20, imgRatio: xRatio, advanced: true},
		{filename: "c0c99e28da91693a27de2beb6dfd7161", position: 0.50, imgRatio: xRatio, advanced: true},
		{filename: "6dbe5b2d7d7056f3b60c6d05f5176529", position: 0.25, imgRatio: xRatio, advanced: true},
		{filename: "369993051097480935eadf1f468eaadb", position: 0.70, imgRatio: xRatio, advanced: true},
		{filename: "d8df07a8312543f638373eb5921f896d", position: 0.15, imgRatio: xRatio, advanced: true},
		{filename: "e8e85575a04d75d2bc29abb4bb7fb447", position: 0.25, imgRatio: xRatio, advanced: true},
		{filename: "c977809e691fc2037f3a9279068720c2", position: 0.70, imgRatio: xRatio, advanced: true},
		{filename: "e1c5fce943a4ba36576607eaa585b9d8", position: 0.90, imgRatio: xRatio, advanced: true},
		{filename: "f100611a90fa024c73132457fa77da36", position: 0.65, imgRatio: xRatio, advanced: true},
		{filename: "e5ff5d6966391409a0fed7d3446b12aa", position: 0.60, imgRatio: xRatio, advanced: true},
		{filename: "068b7fb0c8e3953ff5ed25fe00fc22fd", position: 0.25, imgRatio: xRatio, advanced: true},
		{filename: "6335e1276cd7edc191de3768dd62aa03", position: 0.15, imgRatio: xRatio, advanced: true},
		{filename: "4307f4c6826a88936e6e4351d70195cb", position: 0.45, imgRatio: xRatio, advanced: true},
		{filename: "eec27d560038e3367afe42f4ffa7a8e6", position: 0.65, imgRatio: xRatio, advanced: true},
		{filename: "263a3cc91c74673957ea9ca7dbac11f4", position: 0.15, imgRatio: xRatio, advanced: true},
		{filename: "bfc6d0dcf7d9750d13d3c52cac84ed9a", position: 0.20, imgRatio: xRatio, advanced: true},
		{filename: "db4aec6ce163c3113473af00848f717a", position: 0.85, imgRatio: xRatio, advanced: true},
		{filename: "a6d7e2f816aae0c22150688489491d21", position: 0.60, imgRatio: xRatio, advanced: true},
		{filename: "91271e4f1c1369ab3f06da9b1175d450", position: 0.70, imgRatio: xRatio, advanced: true},
		{filename: "2c1cc55118d22b39dbbae5c8fca2aa1f", position: 0.40, imgRatio: xRatio, advanced: true},
		{filename: "f88aa397bf3ea9df387af2de6d12a6c7", position: 0.20, imgRatio: xRatio, advanced: true},
		{filename: "21939b16a2dc22be9a4035f04da350db", position: 0.25, imgRatio: xRatio, advanced: true},
		{filename: "7e655977ad687e683815567b8081d9f9", position: 0.55, imgRatio: xRatio, advanced: true},
		{filename: "3ed1e1a46f25375ba3478d72cd2b9958", position: 0.28, imgRatio: xRatio, advanced: true},
		// Y Ratio detection:
		{filename: "3ed1e1a46f25375ba3478d72cd2b9958", position: 0.50, imgRatio: yRatio, advanced: true},
		{filename: "eec27d560038e3367afe42f4ffa7a8e6", position: 0.20, imgRatio: yRatio, advanced: true},
		{filename: "263a3cc91c74673957ea9ca7dbac11f4", position: 0.25, imgRatio: yRatio, advanced: true},
		{filename: "bfc6d0dcf7d9750d13d3c52cac84ed9a", position: 0.24, imgRatio: yRatio, advanced: true},
		{filename: "db4aec6ce163c3113473af00848f717a", position: 0.25, imgRatio: yRatio, advanced: true},
		{filename: "c977809e691fc2037f3a9279068720c2", position: 0.30, imgRatio: yRatio, advanced: true},
		{filename: "1784b7cff949300740437e4a777b9c14", position: 0.25, imgRatio: yRatio, advanced: false},
		{filename: "5771fe21c5304bb2e164a278f3dbfc39", position: 0.29, imgRatio: yRatio, advanced: false},
		{filename: "99037fa52996b8da6f8e4a630be1c0ff", position: 0.26, imgRatio: yRatio, advanced: false},
	} {
		t.Run(unit.filename+"/"+ratioName(unit.imgRatio), func(t *testing.T) {
			img := loadFixture(t, unit.filename)

			var (
				innerImg   image.Image
				innerFaces []faceBox
			)
			pos, found := FindPrimaryFaceAxisRatio(
				img, unit.imgRatio, unit.advanced,
				func(img image.Image, faces []faceBox) {
					innerImg, innerFaces = img, faces
				},
			)

			sort.SliceStable(innerFaces, func(i, j int) bool {
				return subjectScore(innerFaces[i]) > subjectScore(innerFaces[j])
			})
			for _, face := range innerFaces {
				t.Logf("face=%+v score=%.2f", face, face.Score)
			}

			if !assert.True(t, found) {
				return
			}
			t.Logf("detected position: %f (want %f)", pos, unit.position)
			// The baselines were recorded with the previous cascade detector and
			// describe where the subject sits, not exact pixels. SCRFD lands within
			// a few percent on every cover; 0.06 tolerates a face box that is a
			// little wider without letting a wrong face through.
			if !assert.LessOrEqualf(t, math.Abs(unit.position-pos), 0.06,
				"expect pos=%.2f, but got pos=%.2f", unit.position, pos) {
				saveDebug(t, unit.filename, drawBoxes(innerImg, innerFaces))
			}
		})
	}
}

func ratioName(ratio float64) string {
	if ratio == xRatio {
		return "x"
	}
	return "y"
}

func loadFixture(t *testing.T, name string) image.Image {
	t.Helper()
	data, err := fs.ReadFile("testdata/" + name)
	require.NoError(t, err)
	decoded, err := base64.StdEncoding.DecodeString(string(data))
	require.NoError(t, err)
	img, _, err := image.Decode(bytes.NewReader(decoded))
	require.NoError(t, err)
	return img
}

func drawBoxes(img image.Image, faces []faceBox) image.Image {
	if img == nil {
		return nil
	}
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)

	red := color.RGBA{R: 255, A: 255}
	for _, f := range faces {
		for x := int(f.X1); x <= int(f.X2); x++ {
			for _, y := range []int{int(f.Y1), int(f.Y2)} {
				if image.Pt(x, y).In(rgba.Bounds()) {
					rgba.Set(x, y, red)
				}
			}
		}
		for y := int(f.Y1); y <= int(f.Y2); y++ {
			for _, x := range []int{int(f.X1), int(f.X2)} {
				if image.Pt(x, y).In(rgba.Bounds()) {
					rgba.Set(x, y, red)
				}
			}
		}
	}
	return rgba
}

func saveDebug(t *testing.T, name string, img image.Image) {
	t.Helper()
	dir := os.Getenv("DETECTOR_DEBUG_DIR")
	if dir == "" || img == nil {
		return
	}
	path := dir + "/" + name + ".jpg"
	if err := writeJPEG(path, img); err != nil {
		t.Logf("cannot write %s: %v", path, err)
		return
	}
	t.Logf("wrote %s", path)
}
