package features

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"runtime"
)

type Model struct {
	session   *ort.AdvancedSession
	input     *ort.Tensor[float32]
	output    *ort.Tensor[float32]
	available bool
}

func InitializeORT() error {
	libPath := "/usr/lib/libonnxruntime.so"
	if runtime.GOOS == "windows" {
		libPath = "onnxruntime.dll"
	} else if runtime.GOOS == "darwin" {
		libPath = "libonnxruntime.dylib"
	}
	ort.SetSharedLibraryPath(libPath)
	return ort.InitializeEnvironment()
}

func NewModel(modelPath string) (*Model, error) {
	// Ensure environment is initialized
	_ = InitializeORT()

	inputShape := ort.NewShape(1, 60, 6)
	inputData := make([]float32, 1*60*6)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %v", err)
	}

	outputShape := ort.NewShape(1, 3)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %v", err)
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"input"}, []string{"output"},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	return &Model{
		session:   session,
		input:     inputTensor,
		output:    outputTensor,
		available: true,
	}, nil
}

func (m *Model) Predict(features []float32) (float32, error) {
	if !m.available {
		return 0, fmt.Errorf("model not available")
	}
	data := m.input.GetData()
	copy(data, features)
	err := m.session.Run()
	if err != nil {
		return 0, fmt.Errorf("inference failed: %v", err)
	}
	outData := m.output.GetData()
	return outData[0], nil
}

func (m *Model) Close() {
	if m.session != nil { m.session.Destroy() }
	if m.input != nil { m.input.Destroy() }
	if m.output != nil { m.output.Destroy() }
}
