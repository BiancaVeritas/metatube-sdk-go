package onnx

import "fmt"

// Run executes the graph on one input tensor and returns the named outputs.
//
// The graph is evaluated in declaration order, which is the order ONNX requires
// of a valid model: every node's inputs are defined before it.
func (m *Model) Run(input *Tensor) (map[string]*Tensor, error) {
	env := make(map[string]*Tensor, len(m.constants)+len(m.nodes)*2)
	for name, tensor := range m.constants {
		env[name] = tensor
	}
	env[m.inputs[0]] = input

	for i, node := range m.nodes {
		if err := m.runNode(env, node, i); err != nil {
			return nil, err
		}
	}

	out := make(map[string]*Tensor, len(m.outputs))
	for _, name := range m.outputs {
		t, ok := env[name]
		if !ok {
			return nil, fmt.Errorf("onnx: output %q was never produced", name)
		}
		out[name] = t
	}
	return out, nil
}

// runNode feeds one operator.
//
// Empty input names keep their slot as nil: Resize spells its ROI and scales
// operands that way, and dropping them would shift the sizes tensor into the
// wrong position.
func (m *Model) runNode(env map[string]*Tensor, node operator, index int) error {
	in := m.nodeInputs[index]
	values := make([]*Tensor, len(in))
	for i, name := range in {
		if name == "" {
			continue
		}
		t, ok := env[name]
		if !ok {
			return fmt.Errorf("onnx: node %d reads undefined tensor %q", index, name)
		}
		values[i] = t
	}
	results, err := node.Run(values)
	if err != nil {
		return fmt.Errorf("onnx: node %d: %w", index, err)
	}
	outs := m.nodeOutputs[index]
	if len(results) != len(outs) {
		return fmt.Errorf("onnx: node %d produced %d outputs, graph declares %d", index, len(results), len(outs))
	}
	for i, name := range outs {
		if name == "" {
			continue
		}
		env[name] = results[i]
	}
	return nil
}
