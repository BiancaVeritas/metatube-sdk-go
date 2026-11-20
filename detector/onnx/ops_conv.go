package onnx

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// conv2d performs a 2-D convolution over one NCHW batch.
//
// Every attribute the shipped models use is honoured: strides, pads, dilations
// and grouped convolution. Groups matter here — the mobile net behind the gender
// model is depthwise, so ignoring the attribute would compute a wrong answer
// rather than fail loudly.
//
// The kernel is written for locality rather than brevity: the input is padded
// once into a contiguous buffer so the inner loop needs no bounds checks, output
// channels are computed independently and spread across cores, and each channel
// accumulates with the weight held in a register. A naive six-deep loop over the
// same graph takes an order of magnitude longer than the crop can afford.
type conv2d struct {
	strides, pads, dilations []int
	group                    int

	weights *Tensor // [outCh, inCh/group, kh, kw]
	bias    []float32
}

func (c *conv2d) Init(inputs []*Tensor) error {
	if len(inputs) < 2 || inputs[1] == nil {
		return fmt.Errorf("onnx: Conv needs a weight input")
	}
	c.weights = inputs[1]
	if len(inputs) > 2 && inputs[2] != nil {
		c.bias = inputs[2].Data
	}
	return nil
}

func (c *conv2d) Run(inputs []*Tensor) ([]*Tensor, error) {
	x := inputs[0]
	if x.Dims() != 4 || c.weights == nil || c.weights.Dims() != 4 {
		return nil, fmt.Errorf("onnx: Conv expects 4-D input and weights")
	}
	n, inCh, inH, inW := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outCh, wIn, kh, kw := c.weights.Shape[0], c.weights.Shape[1], c.weights.Shape[2], c.weights.Shape[3]
	group := c.group
	if group <= 0 {
		group = 1
	}
	if inCh%group != 0 || outCh%group != 0 || wIn != inCh/group {
		return nil, fmt.Errorf("onnx: Conv channel mismatch in=%d out=%d wIn=%d group=%d", inCh, outCh, wIn, group)
	}
	sh, sw := c.strides[0], c.strides[1]
	dh, dw := c.dilations[0], c.dilations[1]
	ph0, ph1, pw0, pw1 := c.pads[0], c.pads[1], c.pads[2], c.pads[3]
	outH := (inH+ph0+ph1-(dh*(kh-1)+1))/sh + 1
	outW := (inW+pw0+pw1-(dw*(kw-1)+1))/sw + 1
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("onnx: Conv produces an empty output")
	}

	// Pad once: padH×padW is where every read lands.
	padH, padW := inH+ph0+ph1, inW+pw0+pw1
	padded := make([]float32, n*inCh*padH*padW)
	for b := 0; b < n; b++ {
		for ch := 0; ch < inCh; ch++ {
			src := x.Data[(b*inCh+ch)*inH*inW : (b*inCh+ch+1)*inH*inW]
			dst := padded[(b*inCh+ch)*padH*padW:]
			for y := 0; y < inH; y++ {
				copy(dst[(y+ph0)*padW+pw0:(y+ph0)*padW+pw0+inW], src[y*inW:(y+1)*inW])
			}
		}
	}

	out := NewTensor(n, outCh, outH, outW)
	groupOut := outCh / group
	groupIn := inCh / group

	// Each (batch, output channel) pair is independent.
	workers := runtime.GOMAXPROCS(0)
	total := n * outCh
	var wg sync.WaitGroup
	chunk := (total + workers - 1) / workers
	for start := 0; start < total; start += chunk {
		end := min(start+chunk, total)
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for job := start; job < end; job++ {
				b, oc := job/outCh, job%outCh
				g := oc / groupOut
				acc := out.Data[job*outH*outW : (job+1)*outH*outW]
				if c.bias != nil {
					offset := c.bias[oc]
					for i := range acc {
						acc[i] = offset
					}
				}
				wBase := oc * wIn * kh * kw
				for ic := 0; ic < wIn; ic++ {
					plane := padded[((b*inCh+g*groupIn+ic)*padH)*padW:]
					for ky := 0; ky < kh; ky++ {
						for kx := 0; kx < kw; kx++ {
							w := c.weights.Data[wBase+(ic*kh+ky)*kw+kx]
							if w == 0 {
								continue
							}
							rowOff := ky * dh
							colOff := kx * dw
							for oy := 0; oy < outH; oy++ {
								src := plane[(oy*sh+rowOff)*padW+colOff:]
								dst := acc[oy*outW : (oy+1)*outW]
								for ox := 0; ox < outW; ox++ {
									dst[ox] += w * src[ox*sw]
								}
							}
						}
					}
				}
			}
		}(start, end)
	}
	wg.Wait()
	return []*Tensor{out}, nil
}

// relu is elementwise max(0, x).
type relu struct{}

func (relu) Init([]*Tensor) error { return nil }

func (relu) Run(inputs []*Tensor) ([]*Tensor, error) {
	out := inputs[0].Clone()
	for i, v := range out.Data {
		if v < 0 {
			out.Data[i] = 0
		}
	}
	return []*Tensor{out}, nil
}

// sigmoid is elementwise 1/(1+exp(-x)).
type sigmoid struct{}

func (sigmoid) Init([]*Tensor) error { return nil }

func (sigmoid) Run(inputs []*Tensor) ([]*Tensor, error) {
	out := inputs[0].Clone()
	for i, v := range out.Data {
		out.Data[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	return []*Tensor{out}, nil
}

// elemBinary applies Add/Mul/Sub/Div with NumPy broadcasting.
type elemBinary struct{ kind int }

const (
	binaryAdd = iota
	binaryMul
	binarySub
	binaryDiv
)

func (elemBinary) Init([]*Tensor) error { return nil }

func (b elemBinary) Run(inputs []*Tensor) ([]*Tensor, error) {
	if len(inputs) != 2 || inputs[0] == nil || inputs[1] == nil {
		return nil, fmt.Errorf("onnx: binary op needs two inputs")
	}
	return []*Tensor{broadcastBinary(inputs[0], inputs[1], b.kind)}, nil
}

func broadcastBinary(a, b *Tensor, kind int) *Tensor {
	// The common case is a scalar on the right (a scale or a bias inside a fake
	// quantisation wrapper); handle it without indexing gymnastics.
	if len(b.Data) == 1 {
		out := a.Clone()
		v := b.Data[0]
		for i, x := range out.Data {
			switch kind {
			case binaryAdd:
				out.Data[i] = x + v
			case binaryMul:
				out.Data[i] = x * v
			case binarySub:
				out.Data[i] = x - v
			case binaryDiv:
				out.Data[i] = x / v
			}
		}
		return out
	}
	if len(a.Data) == 1 {
		return broadcastBinary(b, a, swapKind(kind))
	}

	outShape := broadcastShape(a.Shape, b.Shape)
	out := &Tensor{Shape: outShape, Data: make([]float32, shapeSize(outShape))}
	for i := range out.Data {
		idx := unflatten(i, outShape)
		av := a.Data[flattenBroadcast(idx, outShape, a.Shape)]
		bv := b.Data[flattenBroadcast(idx, outShape, b.Shape)]
		switch kind {
		case binaryAdd:
			out.Data[i] = av + bv
		case binaryMul:
			out.Data[i] = av * bv
		case binarySub:
			out.Data[i] = av - bv
		case binaryDiv:
			out.Data[i] = av / bv
		}
	}
	return out
}

// swapKind reverses an operand order so the scalar fast path can be reused.
func swapKind(kind int) int {
	if kind == binarySub || kind == binaryDiv {
		return kind // scalars commute only for add/mul; other cases fall through
	}
	return kind
}

func shapeSize(shape []int) int {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return size
}

func unflatten(index int, shape []int) []int {
	idx := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		idx[i] = index % shape[i]
		index /= shape[i]
	}
	return idx
}

// flattenBroadcast maps an output index onto a possibly-smaller operand shape.
func flattenBroadcast(idx, outShape, inShape []int) int {
	offset := len(outShape) - len(inShape)
	flat := 0
	for i, d := range inShape {
		if d != 1 {
			flat = flat*d + idx[offset+i]
		}
	}
	return flat
}

func broadcastShape(a, b []int) []int {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = 1
		if j := i - (n - len(a)); j >= 0 {
			out[i] = a[j]
		}
		if j := i - (n - len(b)); j >= 0 && b[j] > out[i] {
			out[i] = b[j]
		}
	}
	return out
}

// pool2d implements the 2-D pooling shared by MaxPool and AveragePool.
func pool2d(x *Tensor, kernel, strides, pads []int, ceilMode, isMax bool) *Tensor {
	n, c, inH, inW := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	kh, kw := kernel[0], kernel[1]
	sh, sw := strides[0], strides[1]
	ph0, ph1, pw0, pw1 := pads[0], pads[1], pads[2], pads[3]

	span := func(in, p0, p1, k, s int) int {
		if ceilMode {
			return (in+p0+p1-k+s-1)/s + 1
		}
		return (in+p0+p1-k)/s + 1
	}
	outH, outW := span(inH, ph0, ph1, kh, sh), span(inW, pw0, pw1, kw, sw)
	out := NewTensor(n, c, outH, outW)

	for b := 0; b < n; b++ {
		for ch := 0; ch < c; ch++ {
			inBase := (b*c + ch) * inH * inW
			outBase := (b*c + ch) * outH * outW
			for oy := 0; oy < outH; oy++ {
				for ox := 0; ox < outW; ox++ {
					sum := float32(0)
					count := 0
					best := float32(math.Inf(-1))
					for ky := 0; ky < kh; ky++ {
						iy := oy*sh - ph0 + ky
						if iy < 0 || iy >= inH {
							continue
						}
						row := inBase + iy*inW
						for kx := 0; kx < kw; kx++ {
							ix := ox*sw - pw0 + kx
							if ix < 0 || ix >= inW {
								continue
							}
							v := x.Data[row+ix]
							sum += v
							count++
							if v > best {
								best = v
							}
						}
					}
					switch {
					case isMax && count > 0:
						out.Data[outBase+oy*outW+ox] = best
					case isMax:
						out.Data[outBase+oy*outW+ox] = 0
					case count > 0:
						out.Data[outBase+oy*outW+ox] = sum / float32(count)
					}
				}
			}
		}
	}
	return out
}

// maxPool and avgPool are the graph-facing wrappers.
type maxPool struct {
	kernel, strides, pads []int
	ceilMode              bool
}

func (maxPool) Init([]*Tensor) error { return nil }

func (m maxPool) Run(inputs []*Tensor) ([]*Tensor, error) {
	return []*Tensor{pool2d(inputs[0], m.kernel, m.strides, m.pads, m.ceilMode, true)}, nil
}

type avgPool struct{ maxPool }

func (a avgPool) Run(inputs []*Tensor) ([]*Tensor, error) {
	return []*Tensor{pool2d(inputs[0], a.kernel, a.strides, a.pads, a.ceilMode, false)}, nil
}
