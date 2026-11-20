package onnx

import (
	"fmt"

	"github.com/metatube-community/metatube-sdk-go/detector/onnx/onnxpb"
)

// attrInts reads an INTS attribute, falling back to its singular form.
func attrInts(node *onnxpb.NodeProto, name string, fallback []int) []int {
	for _, a := range node.Attribute {
		if a.GetName() != name {
			continue
		}
		if ints := a.GetInts(); len(ints) > 0 {
			out := make([]int, len(ints))
			for i, v := range ints {
				out[i] = int(v)
			}
			return out
		}
		return []int{int(a.GetI())}
	}
	return fallback
}

func attrInt(node *onnxpb.NodeProto, name string, fallback int) int {
	for _, a := range node.Attribute {
		if a.GetName() == name {
			return int(a.GetI())
		}
	}
	return fallback
}

func attrFloat(node *onnxpb.NodeProto, name string, fallback float32) float32 {
	for _, a := range node.Attribute {
		if a.GetName() == name {
			return a.GetF()
		}
	}
	return fallback
}

func attrString(node *onnxpb.NodeProto, name, fallback string) string {
	for _, a := range node.Attribute {
		if a.GetName() == name {
			return string(a.GetS())
		}
	}
	return fallback
}

// newOperator builds the operator for one graph node.
func newOperator(node *onnxpb.NodeProto) (operator, error) {
	switch node.OpType {
	case "Conv":
		return &conv2d{
			strides:   attrInts(node, "strides", []int{1, 1}),
			pads:      padAttr(node),
			dilations: attrInts(node, "dilations", []int{1, 1}),
			group:     attrInt(node, "group", 1),
		}, nil
	case "Relu":
		return relu{}, nil
	case "Sigmoid":
		return sigmoid{}, nil
	case "Add":
		return elemBinary{kind: binaryAdd}, nil
	case "Mul":
		return elemBinary{kind: binaryMul}, nil
	case "Sub":
		return elemBinary{kind: binarySub}, nil
	case "Div":
		return elemBinary{kind: binaryDiv}, nil
	case "MaxPool":
		return maxPool{
			kernel:   attrInts(node, "kernel_shape", nil),
			strides:  attrInts(node, "strides", []int{1, 1}),
			pads:     padAttr(node),
			ceilMode: attrInt(node, "ceil_mode", 0) == 1,
		}, nil
	case "AveragePool":
		return avgPool{maxPool{
			kernel:   attrInts(node, "kernel_shape", nil),
			strides:  attrInts(node, "strides", []int{1, 1}),
			pads:     padAttr(node),
			ceilMode: attrInt(node, "ceil_mode", 0) == 1,
		}}, nil
	case "GlobalAveragePool":
		return globalAvgPoolOp{}, nil
	case "BatchNormalization":
		return &batchNormOp{eps: attrFloat(node, "epsilon", 1e-5)}, nil
	case "Resize":
		return &resizeOp{}, nil
	case "Shape":
		return shapeOp{}, nil
	case "Gather":
		return gatherOp{axis: attrInt(node, "axis", 0)}, nil
	case "Slice":
		return &sliceOp{}, nil
	case "Concat":
		return concatOp{axis: attrInt(node, "axis", 0)}, nil
	case "Transpose":
		return transposeOp{perm: attrInts(node, "perm", nil)}, nil
	case "Reshape":
		return &reshapeOp{}, nil
	case "Unsqueeze":
		return unsqueezeOp{axes: attrInts(node, "axes", nil)}, nil
	case "Squeeze":
		return squeezeOp{axes: attrInts(node, "axes", nil)}, nil
	case "Flatten":
		return flattenOp{axis: attrInt(node, "axis", 1)}, nil
	case "Gemm":
		return &gemmOp{
			alpha:  attrFloat(node, "alpha", 1),
			beta:   attrFloat(node, "beta", 1),
			transA: attrInt(node, "transA", 0) == 1,
			transB: attrInt(node, "transB", 0) == 1,
		}, nil
	case "Identity", "Cast":
		return identityOp{}, nil
	default:
		return nil, fmt.Errorf("onnx: unsupported operator %q", node.OpType)
	}
}

// padAttr normalises the several ways ONNX spells padding.
func padAttr(node *onnxpb.NodeProto) []int {
	if pads := attrInts(node, "pads", nil); len(pads) == 4 {
		return pads
	}
	// auto_pad=VALID means no padding; the shipped models use explicit pads.
	return []int{0, 0, 0, 0}
}

