package detector

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/metatube-community/metatube-sdk-go/detector/onnx"
)

// TestGenderReadsOnlyTwoLogits pins the one detail that is easy to get wrong.
//
// The gender/age head emits three values, but only the first two are gender
// logits; the third is the age estimate. Taking an argmax over all three reads
// the age as a gender, and since it often dominates on real faces, every woman
// then comes back "not female" and the subject ranking silently falls back to the
// largest face.
func TestGenderReadsOnlyTwoLogits(t *testing.T) {
	// The models are embedded, so read them the same way the detector does rather
	// than depending on the test's working directory.
	raw, err := models.ReadFile("models/face_detect_genderage.onnx")
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	model, err := onnx.Load(raw)
	require.NoError(t, err)

	input := readParityInput(t, "gender96_input.bin")
	out, err := model.Run(input)
	require.NoError(t, err)

	var logits []float32
	for _, tensor := range out {
		logits = tensor.Data
	}
	require.Len(t, logits, 3)

	// The age slot dominates this natural-looking input, which is exactly the case
	// that breaks an argmax over all three.
	argmaxAll := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[argmaxAll] {
			argmaxAll = i
		}
	}
	assert.NotEqualf(t, genderFemale, argmaxAll,
		"this fixture no longer exercises the bug: age slot %d is not the max", argmaxAll)

	// The verdict must come from the female/male pair alone.
	female := logits[genderFemale] > logits[genderMale]
	assert.Equal(t, logits[0] > logits[1], female)
}

// TestGenderClassifiesRealFaces runs the head over actual cover faces and
// requires that at least one is classified female — a regression guard for the
// logit-index bug, which makes every face read as male.
func TestGenderClassifiesRealFaces(t *testing.T) {
	model := loadModels()
	require.NoError(t, model.err)
	require.NotNil(t, model.gender)

	img := loadFixture(t, "1784b7cff949300740437e4a777b9c14")
	working := resizedForDetection(img, maxImageWidth)
	faces := runSCRFD(working, defaultConfidence)
	require.NotEmpty(t, faces, "no faces detected on the fixture")

	femaleCount := 0
	for _, face := range faces {
		if isFemale, ok := isFemale(model.gender, working, face); ok && isFemale {
			femaleCount++
		}
	}
	assert.Positive(t, femaleCount, "no face classified female; the gender logits are probably read wrong")
}

func readParityInput(t *testing.T, name string) *onnx.Tensor {
	t.Helper()
	// Resolve through this file rather than the working directory: the parity
	// fixtures belong to the onnx package and live beside its tests.
	_, self, _, ok := runtime.Caller(0)
	require.True(t, ok, "cannot resolve test file path")
	path := filepath.Join(filepath.Dir(self), "onnx", "testdata", name)
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Zero(t, len(data)%4)
	floats := make([]float32, len(data)/4)
	for i := range floats {
		floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return &onnx.Tensor{Shape: []int{1, 3, genderSide, genderSide}, Data: floats}
}

