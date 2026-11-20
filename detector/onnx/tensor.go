// Package onnx runs ONNX inference graphs in pure Go.
//
// It exists because the detector has to read the same SCRFD and gender/age
// models the upstream services use, while the SDK stays dependency-free at
// build time: no cgo, no shared libraries, and the same source cross-compiles to
// every supported platform.
//
// The supported operator set is deliberately small — it covers the two models
// this repository ships and nothing else. An unknown operator fails the load
// instead of producing a wrong answer.
package onnx

import (
	"fmt"
	"sort"
)

// Tensor is a dense row-major float32 tensor.
type Tensor struct {
	Shape []int
	Data  []float32
}

// NewTensor allocates a zeroed tensor of the given shape.
func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: make([]float32, size)}
}

// Len is the element count.
func (t *Tensor) Len() int { return len(t.Data) }

// Dims is the number of dimensions.
func (t *Tensor) Dims() int { return len(t.Shape) }

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	return &Tensor{Shape: append([]int(nil), t.Shape...), Data: append([]float32(nil), t.Data...)}
}

// Reshape returns the same data under a new shape. The element count must match.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	size := 1
	for _, d := range shape {
		size *= d
	}
	if size != len(t.Data) {
		return nil, fmt.Errorf("onnx: reshape %v to %v: element count mismatch (%d vs %d)", t.Shape, shape, len(t.Data), size)
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: t.Data}, nil
}

// At indexes a 4-D tensor as NCHW.
func (t *Tensor) At(n, c, y, x int) float32 {
	_, cn, h, w := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	return t.Data[((n*cn+c)*h+y)*w+x]
}

func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor%v", t.Shape)
}

// operator is one executable node.
type operator interface {
	// Run consumes the named inputs and returns its outputs in order.
	Run(inputs []*Tensor) ([]*Tensor, error)
	// Init resolves constant inputs (weights, axes, shapes) once at load time.
	Init(inputs []*Tensor) error
}

// Model is a loaded graph ready for inference.
type Model struct {
	nodes       []operator
	nodeInputs  [][]string
	nodeOutputs [][]string
	constants   map[string]*Tensor
	inputs      []string
	outputs     []string
}

const maxNodes = 4096

// supported lists every operator this package implements.
var supported = map[string]bool{
	"Conv": true, "Relu": true, "Add": true, "Mul": true, "Sub": true,
	"MaxPool": true, "AveragePool": true, "Resize": true,
	"Shape": true, "Gather": true, "Slice": true, "Concat": true,
	"Transpose": true, "Reshape": true, "Sigmoid": true, "Unsqueeze": true,
	"Squeeze": true, "Constant": true, "BatchNormalization": true,
	"GlobalAveragePool": true, "Gemm": true, "Flatten": true, "Softmax": true,
	"Clip": true, "Pad": true, "Split": true, "Identity": true, "Cast": true,
	"Div": true, "Sqrt": true, "Exp": true, "Log": true, "LeakyRelu": true,
	"PRelu": true, "HardSigmoid": true, "MatMul": true, "ReduceMean": true,
}

// SupportedOperators reports the implemented operator names, sorted. It exists
// so callers can log what a toolchain-free build can run.
func SupportedOperators() []string {
	names := make([]string, 0, len(supported))
	for name := range supported {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
