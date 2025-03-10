#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace nvinfer1;

// TensorRT Logger
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

// TensorRT 엔진 로드 함수
ICudaEngine* loadEngine(const std::string& enginePath, Logger& logger) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file!" << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    return runtime->deserializeCudaEngine(buffer.data(), size);
}

// 이미지 정규화 함수 (mean, std 적용)
void preprocessImage(const cv::Mat& img, float* inputBuffer, const int inputWidth, const int inputHeight) {
    cv::Mat resized, floatImg;
    
    // 1. 이미지 리사이즈 (128x128)
    cv::resize(img, resized, cv::Size(inputWidth, inputHeight));

    // 2. float 변환 및 정규화 (0~1 스케일링)
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    // 3. mean/std 정규화
    const float mean[3] = {0.485f, 0.456f, 0.406f};  // 예제값 (PyTorch ImageNet Mean)
    const float std[3]  = {0.229f, 0.224f, 0.225f};  // 예제값 (PyTorch ImageNet Std)

    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                float pixel = floatImg.at<cv::Vec3f>(i, j)[c];
                inputBuffer[index++] = (pixel - mean[c]) / std[c];  // 정규화 적용
            }
        }
    }
}

int main() {
    Logger logger;
    std::string enginePath = "../lseg_image_encoder_128.trt";  // TensorRT 엔진 파일
    std::string imagePath = "../cat.jpeg";   // 입력 이미지 파일

    // 1. TensorRT 엔진 로드
    ICudaEngine* engine = loadEngine(enginePath, logger);
    if (!engine) return -1;
    IExecutionContext* context = engine->createExecutionContext();

    // 2. 입력 및 출력 정보
    int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex("output");

    // 3. 입력 및 출력 크기
    int inputWidth = 128, inputHeight = 128;
    int inputSize = 3 * inputWidth * inputHeight * sizeof(float);  
    int outputSize = 64 * 64 * 512 * sizeof(float);

    // 4. GPU 메모리 할당
    void* d_input, * d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 5. 이미지 로드 및 정규화
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    std::vector<float> inputData(3 * inputWidth * inputHeight);
    preprocessImage(img, inputData.data(), inputWidth, inputHeight);
    
    cudaMemcpy(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice);

    // 6. 실행 시간 측정 (CUDA 이벤트)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // 시작 시간 기록
    void* bindings[] = {d_input, d_output};
    context->enqueueV2(bindings, 0, nullptr);
    cudaEventRecord(stop);   // 종료 시간 기록

    cudaEventSynchronize(stop); // 실행 완료 대기

    // 7. 실행 시간 측정
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Inference 수행 시간: " << milliseconds << " ms" << std::endl;

    // 8. 결과 가져오기
    std::vector<float> outputData(64 * 64 * 512);
    cudaMemcpy(outputData.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    std::cout << "Inference 완료!" << std::endl;

    // 9. 메모리 해제
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    context->destroy();
    engine->destroy();

    return 0;
}
