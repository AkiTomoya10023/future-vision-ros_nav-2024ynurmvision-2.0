#ifndef YOLOX_DETECTOR_HPP
#define YOLOX_DETECTOR_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "logging.h"
#include "Eigen/Core"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "cuda.h"
#include "../debug.h"

namespace yolox_detector
{
    using namespace std;
#undef CHECK
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

    static constexpr int YOLOX_INPUT_W = 416; // Width of input
    static constexpr int YOLOX_INPUT_H = 416; // Height of input
    static constexpr int NUM_CLASSES = 8;     // Number of classes
    static constexpr int NUM_COLORS = 4;      // Number of color
    static constexpr int TOPK = 128;          // TopK
    static constexpr float NMS_THRESH = 0.3;
    static constexpr float BBOX_CONF_THRESH = 0.75;
    static constexpr float MERGE_CONF_ERROR = 0.15;
    static constexpr float MERGE_MIN_IOU = 0.9;

    struct ArmorObject
    {
        cv::Point2f apex[4];
        cv::Rect_<float> rect;
        int cls;
        int color;
        int area;
        float prob;
        std::vector<cv::Point2f> pts;
    };

    struct InferDeleter
    {
        template <typename T>
        void operator()(T *obj) const
        {
            delete obj;
        }
    };
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    inline void enableDLA(
        nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config, int useDLACore, bool allowGPUFallback = true)
    {
        if (useDLACore >= 0)
        {
            if (builder->getNbDLACores() == 0)
            {
                std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                          << std::endl;
                assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
            }
            if (allowGPUFallback)
            {
                config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            }
            if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
            {
                // User has not requested INT8 Mode.
                // By default run in FP16 mode. FP32 mode is not permitted.
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(useDLACore);
        }
    }

    static auto StreamDeleter = [](cudaStream_t *pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };

    inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
    {
        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
        if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
        {
            pStream.reset(nullptr);
        }

        return pStream;
    }

    class ArmorDetector
    {
    public:
        ArmorDetector();
        ~ArmorDetector();

        string onnx_file;
        nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.
        nvinfer1::Dims mOutputDims;
        Logger gLogger_;
        bool build();
        bool build_from_engine();
        bool build_from_onnx();
        cv::Mat detect(std::vector<ArmorObject>& objects, cv::Mat &src);
        void doInference(float *input, float *output, const int output_size, cv::Size input_shape);
        bool initModel(string model_path);

    private:
        int DEVICE_ = 0; // GPU id
        // int input_name_;
        // int output_name_;
        int input_w_;
        int input_h_;
        int output_size_;
        int dlaCore = 0;
        bool fp16 = false;
        bool int8 = false;

        Eigen::Matrix<float, 3, 3> transform_matrix;

        string path_to_engine;

        const int inputIndex_ = 0;
        const int outputIndex_ = 1;

        // SampleUniquePtrr<nvinfer1::IRuntime> runtime_;
        SampleUniquePtr<nvinfer1::IExecutionContext> context_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_; //!< The TensorRT engine used to run the network
    };

}

#endif
