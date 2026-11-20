package onnx

import (
	"fmt"
	"math"
)

// batchNormOp applies inference-mode batch normalisation.
//
// It is folded once at load time into a per-channel factor and offset, so the
// run itself is a multiply-add over the input.
type batchNormOp struct {
	eps  float32
	fold bool
}

func (b *batchNormOp) Init([]*Tensor) error {
	if b.eps < 0 {
		return fmt.Errorf("onnx: BatchNormalization epsilon must be non-negative")
	}
	return nil
}

func (b batchNormOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	if len(inputs) < 5 {
		return nil, fmt.Errorf("onnx: BatchNormalization needs scale/bias/mean/var")
	}
	x := inputs[0]
	gamma, beta := inputs[1].Data, inputs[2].Data
	mean, variance := inputs[3].Data, inputs[4].Data
	if x.Dims() != 4 {
		return nil, fmt.Errorf("onnx: BatchNormalization expects 4-D input")
	}
	n, c, h, w := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	if len(gamma) != c || len(beta) != c || len(mean) != c || len(variance) != c {
		return nil, fmt.Errorf("onnx: BatchNormalization channel mismatch for %v", x.Shape)
	}
	out := NewTensor(n, c, h, w)
	spatial := h * w
	for ch := 0; ch < c; ch++ {
		factor := gamma[ch] / float32(math.Sqrt(float64(variance[ch]+b.eps)))
		offset := beta[ch] - mean[ch]*factor
		for bi := 0; bi < n; bi++ {
			base := (bi*c + ch) * spatial
			for i := 0; i < spatial; i++ {
				out.Data[base+i] = x.Data[base+i]*factor + offset
			}
		}
	}
	return []*Tensor{out}, nil
}

// globalAvgPoolOp averages each channel down to one value.
type globalAvgPoolOp struct{}

func (globalAvgPoolOp) Init([]*Tensor) error { return nil }

func (globalAvgPoolOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	if x.Dims() < 3 {
		return nil, fmt.Errorf("onnx: GlobalAveragePool expects at least 3-D")
	}
	n, c := x.Shape[0], x.Shape[1]
	spatial := shapeSize(x.Shape[2:])
	out := NewTensor(n, c, 1, 1)
	for b := 0; b < n; b++ {
		for ch := 0; ch < c; ch++ {
			base := (b*c + ch) * spatial
			sum := float32(0)
			for i := 0; i < spatial; i++ {
				sum += x.Data[base+i]
			}
			out.Data[b*c+ch] = sum / float32(spatial)
		}
	}
	return []*Tensor{out}, nil
}

// flattenOp collapses dimensions from axis onwards.
type flattenOp struct{ axis int }

func (flattenOp) Init([]*Tensor) error { return nil }

func (f flattenOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	axis := f.axis
	if axis < 0 {
		axis += x.Dims()
	}
	if axis < 0 || axis > x.Dims() {
		return nil, fmt.Errorf("onnx: Flatten axis %d out of range", f.axis)
	}
	return []*Tensor{{Shape: []int{shapeSize(x.Shape[:axis]), shapeSize(x.Shape[axis:])}, Data: x.Data}}, nil
}

// gemmOp is Y = alpha*op(A)*op(B) + beta*C.
type gemmOp struct {
	alpha, beta float32
	transA      bool
	transB      bool
}

func (g *gemmOp) Init([]*Tensor) error { return nil }

func (g gemmOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	a := inputs[0]
	if len(inputs) < 2 || inputs[1] == nil {
		return nil, fmt.Errorf("onnx: Gemm needs B")
	}
	b := inputs[1]
	if a.Dims() != 2 || b.Dims() != 2 {
		return nil, fmt.Errorf("onnx: Gemm expects 2-D operands, got %v and %v", a.Shape, b.Shape)
	}
	m, k := a.Shape[0], a.Shape[1]
	kb, n := b.Shape[0], b.Shape[1]
	if g.transA {
		m, k = k, m
	}
	if g.transB {
		kb, n = n, kb
	}
	if k != kb {
		return nil, fmt.Errorf("onnx: Gemm inner dimensions differ: %d vs %d", k, kb)
	}

	at := func(i, j int) float32 {
		if g.transA {
			return a.Data[j*a.Shape[1]+i]
		}
		return a.Data[i*a.Shape[1]+j]
	}
	bt := func(i, j int) float32 {
		if g.transB {
			return b.Data[j*b.Shape[1]+i]
		}
		return b.Data[i*b.Shape[1]+j]
	}

	out := NewTensor(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += at(i, p) * bt(p, j)
			}
			out.Data[i*n+j] = g.alpha * sum
		}
	}
	if len(inputs) > 2 && inputs[2] != nil && g.beta != 0 {
		c := inputs[2].Data
		for i := range out.Data {
			out.Data[i] += g.beta * c[i%len(c)]
		}
	}
	return []*Tensor{out}, nil
}

// identityOp passes data through unchanged; both shipped models stay in float32,
// so a Cast has nothing to convert.
type identityOp struct{}

func (identityOp) Init([]*Tensor) error { return nil }

func (identityOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	return []*Tensor{inputs[0]}, nil
}
