package onnx

import (
	"fmt"
	"math"
)

// shapeOp reports a tensor's dimensions.
type shapeOp struct{}

func (shapeOp) Init([]*Tensor) error { return nil }

func (shapeOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	out := NewTensor(len(x.Shape))
	for i, d := range x.Shape {
		out.Data[i] = float32(d)
	}
	return []*Tensor{out}, nil
}

// gatherOp indexes a tensor along one axis.
type gatherOp struct{ axis int }

func (g gatherOp) Init([]*Tensor) error { return nil }

func (g gatherOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	data, indices := inputs[0], inputs[1]
	axis := g.axis
	if axis < 0 {
		axis += data.Dims()
	}
	if axis < 0 || axis >= data.Dims() {
		return nil, fmt.Errorf("onnx: Gather axis %d out of range", g.axis)
	}
	outer := shapeSize(data.Shape[:axis])
	inner := shapeSize(data.Shape[axis+1:])
	dim := data.Shape[axis]
	count := len(indices.Data)

	outShape := append(append(append([]int(nil), data.Shape[:axis]...), indices.Shape...), data.Shape[axis+1:]...)
	out := &Tensor{Shape: outShape, Data: make([]float32, outer*count*inner)}
	for o := 0; o < outer; o++ {
		for i := 0; i < count; i++ {
			sel := int(indices.Data[i])
			if sel < 0 {
				sel += dim
			}
			if sel < 0 || sel >= dim {
				return nil, fmt.Errorf("onnx: Gather index %d out of range", sel)
			}
			copy(out.Data[(o*count+i)*inner:(o*count+i+1)*inner], data.Data[(o*dim+sel)*inner:(o*dim+sel+1)*inner])
		}
	}
	return []*Tensor{out}, nil
}

// sliceOp takes a sub-range along the named axes.
type sliceOp struct {
	starts, ends, axes *Tensor
}

func (s *sliceOp) Init(inputs []*Tensor) error {
	if len(inputs) < 3 {
		return fmt.Errorf("onnx: Slice needs starts and ends")
	}
	s.starts, s.ends = inputs[1], inputs[2]
	if len(inputs) > 3 {
		s.axes = inputs[3]
	}
	return nil
}

func (s sliceOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	out := inputs[0].Clone()
	for i := range s.starts.Data {
		axis := i
		if s.axes != nil {
			axis = int(s.axes.Data[i])
		}
		if axis < 0 {
			axis += out.Dims()
		}
		if axis < 0 || axis >= out.Dims() {
			return nil, fmt.Errorf("onnx: Slice axis %d out of range", axis)
		}
		dim := out.Shape[axis]
		start, end := int(s.starts.Data[i]), int(s.ends.Data[i])
		if start < 0 {
			start += dim
		}
		if end < 0 {
			end += dim
		}
		start, end = clampInt(start, 0, dim), clampInt(end, 0, dim)
		if end < start {
			end = start
		}
		out = sliceAxis(out, axis, start, end)
	}
	return []*Tensor{out}, nil
}

func sliceAxis(t *Tensor, axis, start, end int) *Tensor {
	outer := shapeSize(t.Shape[:axis])
	inner := shapeSize(t.Shape[axis+1:])
	length := end - start
	outShape := append([]int(nil), t.Shape...)
	outShape[axis] = length
	out := &Tensor{Shape: outShape, Data: make([]float32, outer*length*inner)}
	for o := 0; o < outer; o++ {
		for i := 0; i < length; i++ {
			copy(
				out.Data[(o*length+i)*inner:(o*length+i+1)*inner],
				t.Data[(o*t.Shape[axis]+start+i)*inner:(o*t.Shape[axis]+start+i+1)*inner],
			)
		}
	}
	return out
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// concatOp joins tensors along an axis.
type concatOp struct{ axis int }

func (concatOp) Init([]*Tensor) error { return nil }

func (c concatOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	axis := c.axis
	if axis < 0 {
		axis += inputs[0].Dims()
	}
	outShape := append([]int(nil), inputs[0].Shape...)
	total := 0
	for _, in := range inputs {
		total += in.Shape[axis]
	}
	outShape[axis] = total

	outer := shapeSize(outShape[:axis])
	inner := shapeSize(outShape[axis+1:])
	out := &Tensor{Shape: outShape, Data: make([]float32, outer*total*inner)}
	for o := 0; o < outer; o++ {
		cursor := 0
		for _, in := range inputs {
			n := in.Shape[axis]
			copy(out.Data[(o*total+cursor)*inner:(o*total+cursor+n)*inner], in.Data[(o*n)*inner:(o*n+n)*inner])
			cursor += n
		}
	}
	return []*Tensor{out}, nil
}

// transposeOp permutes dimensions.
type transposeOp struct{ perm []int }

func (transposeOp) Init([]*Tensor) error { return nil }

func (t transposeOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	outShape := make([]int, len(t.perm))
	for i, p := range t.perm {
		if p < 0 || p >= x.Dims() {
			return nil, fmt.Errorf("onnx: Transpose perm %v invalid for %v", t.perm, x.Shape)
		}
		outShape[i] = x.Shape[p]
	}
	out := &Tensor{Shape: outShape, Data: make([]float32, len(x.Data))}
	for i := range out.Data {
		outIdx := unflatten(i, outShape)
		inIdx := make([]int, len(outShape))
		for j, p := range t.perm {
			inIdx[p] = outIdx[j]
		}
		out.Data[i] = x.Data[flatten(inIdx, x.Shape)]
	}
	return []*Tensor{out}, nil
}

func flatten(idx, shape []int) int {
	flat := 0
	for i, d := range shape {
		flat = flat*d + idx[i]
	}
	return flat
}

// reshapeOp reinterprets the data under a new shape, resolving the -1 wildcard.
type reshapeOp struct{ shape *Tensor }

func (r *reshapeOp) Init(inputs []*Tensor) error {
	if len(inputs) < 2 || inputs[1] == nil {
		return fmt.Errorf("onnx: Reshape needs a shape input")
	}
	r.shape = inputs[1]
	return nil
}

func (r reshapeOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	dims := make([]int, len(r.shape.Data))
	known, wildcard := 1, -1
	for i, v := range r.shape.Data {
		d := int(math.Round(float64(v)))
		dims[i] = d
		if d == -1 {
			wildcard = i
			continue
		}
		if d >= 0 {
			known *= d
		}
	}
	if wildcard >= 0 {
		if known <= 0 {
			return nil, fmt.Errorf("onnx: Reshape cannot infer dimension")
		}
		dims[wildcard] = x.Len() / known
	}
	return []*Tensor{{Shape: dims, Data: x.Data}}, nil
}

// unsqueezeOp inserts size-1 dimensions.
type unsqueezeOp struct{ axes []int }

func (unsqueezeOp) Init([]*Tensor) error { return nil }

func (u unsqueezeOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	shape := append([]int(nil), x.Shape...)
	for _, a := range u.axes {
		if a < 0 {
			a += len(shape) + 1
		}
		if a < 0 || a > len(shape) {
			return nil, fmt.Errorf("onnx: Unsqueeze axis %d out of range", a)
		}
		shape = append(shape[:a], append([]int{1}, shape[a:]...)...)
	}
	return []*Tensor{{Shape: shape, Data: x.Data}}, nil
}

// squeezeOp drops size-1 dimensions.
type squeezeOp struct{ axes []int }

func (squeezeOp) Init([]*Tensor) error { return nil }

func (s squeezeOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	drop := make(map[int]bool, len(s.axes))
	for _, a := range s.axes {
		if a < 0 {
			a += x.Dims()
		}
		drop[a] = true
	}
	shape := make([]int, 0, x.Dims())
	for i, d := range x.Shape {
		if drop[i] || (len(s.axes) == 0 && d == 1) {
			continue
		}
		shape = append(shape, d)
	}
	return []*Tensor{{Shape: shape, Data: x.Data}}, nil
}

// resizeOp upsamples with nearest-neighbour, which is all the FPN neck needs.
//
// The target size is the fourth input. In this graph it is computed at run time
// from the sibling branch's Shape, so it cannot be resolved at load time.
type resizeOp struct{}

func (resizeOp) Init([]*Tensor) error { return nil }

func (resizeOp) Run(inputs []*Tensor) ([]*Tensor, error) {
	if len(inputs) < 1 || inputs[0] == nil {
		return nil, fmt.Errorf("onnx: Resize needs an input tensor")
	}
	x := inputs[0]
	n, c := x.Shape[0], x.Shape[1]
	inH, inW := x.Shape[2], x.Shape[3]

	outH, outW := inH, inW
	if len(inputs) > 3 && inputs[3] != nil && len(inputs[3].Data) == 4 {
		outH = int(math.Round(float64(inputs[3].Data[2])))
		outW = int(math.Round(float64(inputs[3].Data[3])))
	}
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("onnx: Resize target %dx%d is invalid", outH, outW)
	}

	out := NewTensor(n, c, outH, outW)
	// coordinate_transformation_mode=asymmetric, nearest_mode=floor.
	sy := float64(inH) / float64(outH)
	sx := float64(inW) / float64(outW)
	for b := 0; b < n; b++ {
		for ch := 0; ch < c; ch++ {
			inBase := (b*c + ch) * inH * inW
			outBase := (b*c + ch) * outH * outW
			for oy := 0; oy < outH; oy++ {
				iy := int(math.Floor(float64(oy) * sy))
				if iy >= inH {
					iy = inH - 1
				}
				for ox := 0; ox < outW; ox++ {
					ix := int(math.Floor(float64(ox) * sx))
					if ix >= inW {
						ix = inW - 1
					}
					out.Data[outBase+oy*outW+ox] = x.Data[inBase+iy*inW+ix]
				}
			}
		}
	}
	return []*Tensor{out}, nil
}
