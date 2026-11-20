package onnx

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	scrfdPath  = "../models/face_detect_scrfd.onnx"
	genderPath = "../models/face_detect_genderage.onnx"
)

func loadModel(t *testing.T, path string) *Model {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	m, err := Load(data)
	require.NoError(t, err)
	require.NotNil(t, m)
	return m
}

func TestLoadScrfd(t *testing.T) {
	m := loadModel(t, scrfdPath)
	assert.Equal(t, "input.1", m.InputName())
	assert.Len(t, m.OutputNames(), 9)
	assert.Equal(t, 158, m.NodeCount())
}

func TestLoadGender(t *testing.T) {
	m := loadModel(t, genderPath)
	assert.Equal(t, "data", m.InputName())
	assert.Equal(t, []string{"fc1"}, m.OutputNames())
}

// TestScrfdShapes pins the output shapes the decoder relies on. A wrong stride
// grouping or anchor count shows up here as a shape mismatch.
func TestScrfdShapes(t *testing.T) {
	m := loadModel(t, scrfdPath)
	// 640x640 input, the size the crop path uses.
	got, err := m.Run(NewTensor(1, 3, 640, 640))
	require.NoError(t, err)

	want := map[string][]int{
		"448": {12800, 1}, "471": {3200, 1}, "494": {800, 1},
		"451": {12800, 4}, "474": {3200, 4}, "497": {800, 4},
		"454": {12800, 10}, "477": {3200, 10}, "500": {800, 10},
	}
	for name, shape := range want {
		tensor, ok := got[name]
		require.True(t, ok, "missing output %s", name)
		assert.Equal(t, shape, tensor.Shape, "output %s", name)
	}
}

// TestScrfdScoresAreNotEmpty proves the graph actually computes something: a
// zero input must still yield finite sigmoid scores in [0,1].
func TestScrfdScoresAreNotEmpty(t *testing.T) {
	m := loadModel(t, scrfdPath)
	got, err := m.Run(NewTensor(1, 3, 64, 64))
	require.NoError(t, err)
	scores := got["448"]
	require.NotNil(t, scores)
	assert.NotEmpty(t, scores.Data)
	for _, v := range scores.Data {
		assert.False(t, v < 0 || v > 1, "score %v outside [0,1]", v)
	}
}

// TestGenderOutputShape pins the 3-way output the attribute model reads.
func TestGenderOutputShape(t *testing.T) {
	m := loadModel(t, genderPath)
	got, err := m.Run(NewTensor(1, 3, 96, 96))
	require.NoError(t, err)
	fc := got["fc1"]
	require.NotNil(t, fc)
	assert.Equal(t, []int{1, 3}, fc.Shape)
	for _, v := range fc.Data {
		assert.False(t, v != v, "NaN in gender output")
	}
}


