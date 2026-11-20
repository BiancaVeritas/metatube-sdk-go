package onnx

import (
	"encoding/binary"
	"fmt"
	"math"

	"google.golang.org/protobuf/proto"

	"github.com/metatube-community/metatube-sdk-go/detector/onnx/onnxpb"
)

// binaryLE is the byte order ONNX uses for raw tensor data.
var binaryLE = binary.LittleEndian

// Load parses an ONNX model and prepares it for inference.
//
// Loading is eager: operator parameters and constant operands are resolved once
// here, so an unsupported or malformed model fails at startup instead of midway
// through a crop.
func Load(data []byte) (*Model, error) {
	model := &onnxpb.ModelProto{}
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("onnx: parse model: %w", err)
	}
	graph := model.GetGraph()
	if graph == nil {
		return nil, fmt.Errorf("onnx: model has no graph")
	}
	if len(graph.Node) > maxNodes {
		return nil, fmt.Errorf("onnx: graph has %d nodes, limit is %d", len(graph.Node), maxNodes)
	}

	constants := make(map[string]*Tensor, len(graph.Initializer))
	for _, init := range graph.Initializer {
		t, err := tensorFromProto(init)
		if err != nil {
			return nil, fmt.Errorf("onnx: initializer %q: %w", init.GetName(), err)
		}
		constants[init.GetName()] = t
	}

	m := &Model{
		constants: constants,
		inputs:    make([]string, 0, len(graph.Input)),
		outputs:   make([]string, 0, len(graph.Output)),
	}
	for _, in := range graph.Input {
		if _, isConst := constants[in.GetName()]; isConst {
			continue // graph inputs that are also initializers are constants
		}
		m.inputs = append(m.inputs, in.GetName())
	}
	for _, out := range graph.Output {
		m.outputs = append(m.outputs, out.GetName())
	}
	if len(m.inputs) == 0 {
		return nil, fmt.Errorf("onnx: model has no runtime input")
	}

	for _, node := range graph.Node {
		op, err := newOperator(node)
		if err != nil {
			return nil, err
		}
		// Resolve constant operands now; runtime values stay nil until Run.
		constInputs := make([]*Tensor, len(node.Input))
		for i, name := range node.Input {
			constInputs[i] = constants[name]
		}
		if err := op.Init(constInputs); err != nil {
			return nil, fmt.Errorf("onnx: init %s (%s): %w", node.OpType, node.GetName(), err)
		}
		m.nodes = append(m.nodes, op)
		m.nodeInputs = append(m.nodeInputs, node.Input)
		m.nodeOutputs = append(m.nodeOutputs, node.Output)
	}
	return m, nil
}

// InputName is the graph's runtime input name.
func (m *Model) InputName() string {
	if len(m.inputs) == 0 {
		return ""
	}
	return m.inputs[0]
}

// OutputNames lists the graph outputs in declaration order.
func (m *Model) OutputNames() []string { return m.outputs }

// NodeCount is the number of executable nodes.
func (m *Model) NodeCount() int { return len(m.nodes) }

func tensorFromProto(t *onnxpb.TensorProto) (*Tensor, error) {
	dims := make([]int, 0, len(t.GetDims()))
	for _, d := range t.GetDims() {
		if d < 0 {
			return nil, fmt.Errorf("negative dimension %d", d)
		}
		dims = append(dims, int(d))
	}
	tensor := &Tensor{Shape: dims, Data: make([]float32, shapeSize(dims))}

	raw := t.GetRawData()
	switch onnxpb.TensorProto_DataType(t.GetDataType()) {
	case onnxpb.TensorProto_FLOAT:
		if len(raw) > 0 {
			if len(raw)%4 != 0 {
				return nil, fmt.Errorf("raw float data length %d is not a multiple of 4", len(raw))
			}
			for i := 0; i < len(raw); i += 4 {
				tensor.Data[i/4] = math.Float32frombits(binaryLE.Uint32(raw[i:]))
			}
			return tensor, nil
		}
		copy(tensor.Data, t.GetFloatData())
		return tensor, nil

	case onnxpb.TensorProto_INT64, onnxpb.TensorProto_INT32:
		// Shape, axes and index operands reach the graph as floats; the values are
		// small integers by construction.
		width := 8
		if onnxpb.TensorProto_DataType(t.GetDataType()) == onnxpb.TensorProto_INT32 {
			width = 4
		}
		if len(raw) > 0 {
			if len(raw)%width != 0 {
				return nil, fmt.Errorf("raw int data length %d is not a multiple of %d", len(raw), width)
			}
			for i := 0; i < len(raw); i += width {
				if width == 8 {
					tensor.Data[i/width] = float32(int64(binaryLE.Uint64(raw[i:])))
				} else {
					tensor.Data[i/width] = float32(int32(binaryLE.Uint32(raw[i:])))
				}
			}
			return tensor, nil
		}
		for i, v := range t.GetInt64Data() {
			if i < len(tensor.Data) {
				tensor.Data[i] = float32(v)
			}
		}
		for i, v := range t.GetInt32Data() {
			if i < len(tensor.Data) {
				tensor.Data[i] = float32(v)
			}
		}
		return tensor, nil

	default:
		return nil, fmt.Errorf("unsupported initializer data type %v", t.GetDataType())
	}
}
