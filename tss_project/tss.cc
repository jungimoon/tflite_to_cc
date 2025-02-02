#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <vector>
#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tss_project/models/tss_model.h"

// This is an example that is TSS to inference

// input data (293,3,3,10)
// output data (293,)
const char* input_file = "/home/jgmoon/tinyml/devinlife/tensorflow/tss_project/data/target_list.bin";   
const char* output_file = "/home/jgmoon/tinyml/devinlife/tensorflow/tss_project/data/wqmn_list.bin"; 

const int BATCH_SIZE = 293;
const int HEIGHT = 3;
const int WIDTH = 3;
const int CHANNELS = 10;
const int INPUT_SIZE = BATCH_SIZE * HEIGHT * WIDTH * CHANNELS;
const int OUTPUT_SIZE = BATCH_SIZE; // (293,)

using namespace tflite;

// define NSE
double nse(const std::vector<float>& predictions, const std::vector<float>& targets);

// load data
void LoadInputData(tflite::Interpreter* interpreter);
void LoadOutputData(std::vector<float>& output_data, tflite::Interpreter* interpreter);

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main() {
  // Load model
  // model not embedded
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(reinterpret_cast<const char*>(g_model), g_model_len);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Tensor resize (293,3,3,10)
  std::vector<int> input_dims = {293, 3, 3, 10}; 
  interpreter->ResizeInputTensor(interpreter->inputs()[0], input_dims);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  LoadInputData(interpreter.get());

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  // Read inference result
  float* model_output = interpreter->typed_output_tensor<float>(0);

  // Read ground truth data
  std::vector<float> expected_output;
  LoadOutputData(expected_output, interpreter.get());

  // Convert `model_output` to `std::vector<float>`
  std::vector<float> predictions(expected_output.size());
  for (size_t i = 0; i < expected_output.size(); ++i) {
      predictions[i] = model_output[i];
  }

  // Comparison of expected and truth data
  std::cout << "\n=== Model Output vs. Expected Output ===\n";
  for (int i = 0; i < 10; i++) {
    std::cout << "Inference value: " << predictions[i]
              << ", Ground Truth: " << expected_output[i] << std::endl;
  }

  // Compute NSE accuracy
  double nse_score = nse(predictions, expected_output);

  // Print NSE result
  std::cout << "\nNash-Sutcliffe Efficiency (NSE) Score: " << nse_score << std::endl;

  return 0;
}




void LoadInputData(tflite::Interpreter* interpreter) {
    std::ifstream file(input_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open input.bin file" << std::endl;
        exit(1);
    }

    std::vector<float> input_data(INPUT_SIZE);
    file.read(reinterpret_cast<char*>(input_data.data()), input_data.size() * sizeof(float));
    file.close();

    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor, input_data.data(), input_data.size() * sizeof(float));

    std::cout << "Input data loaded successfully from input.bin!" << std::endl;
}

void LoadOutputData(std::vector<float>& output_data, tflite::Interpreter* interpreter) {
    std::ifstream file(output_file, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output.bin file" << std::endl;
        exit(1);
    }

    // 모델이 예상하는 출력 크기 가져오기
    int output_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_index);

    // ✅ 모델이 예상하는 출력 크기 (293,)로 설정
    output_data.resize(293);

    // ✅ 파일에서 데이터 읽기
    file.read(reinterpret_cast<char*>(output_data.data()), output_data.size() * sizeof(float));
    file.close();

    // ✅ 첫 10개 값 출력 (디버깅)
    std::cout << "\n=== First 10 values from output.bin ===" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Ground Truth [" << i << "]: " << output_data[i] << std::endl;
    }

    std::cout << "✅ Output data successfully loaded from output.bin!" << std::endl;
}


// Nash-Sutcliffe Efficiency 
double nse(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size()) {
        std::cerr << "Error: predictions and targets size mismatch" << std::endl;
        return -1.0;
    }

    // calculate mean value
    double mean_target = std::accumulate(targets.begin(), targets.end(), 0.0) / targets.size();

    // son
    double numerator = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        numerator += (targets[i] - predictions[i]) * (targets[i] - predictions[i]);
    }

    // mother
    double denominator = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        denominator += (targets[i] - mean_target) * (targets[i] - mean_target);
    }

    return 1.0 - (numerator / denominator);
}