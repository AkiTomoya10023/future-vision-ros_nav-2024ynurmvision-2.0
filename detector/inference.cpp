#include "inference.h"

using namespace nvinfer1;
using namespace yolox_detector;

static inline int argmax(const float *ptr, int len)
{
    int max_arg = 0;
    for (int i = 1; i < len; i++)
    {
        if (ptr[i] > ptr[max_arg])
            max_arg = i;
    }
    return max_arg;
}

/**
 * @brief 海伦公式计算三角形面积
 *
 * @param pts 三角形顶点
 * @return float 面积
 */
float calcTriangleArea(cv::Point2f pts[3])
{
    auto a = sqrt(pow((pts[0] - pts[1]).x, 2) + pow((pts[0] - pts[1]).y, 2));
    auto b = sqrt(pow((pts[1] - pts[2]).x, 2) + pow((pts[1] - pts[2]).y, 2));
    auto c = sqrt(pow((pts[2] - pts[0]).x, 2) + pow((pts[2] - pts[0]).y, 2));

    auto p = (a + b + c) / 2.f;

    return sqrt(p * (p - a) * (p - b) * (p - c));
}

/**
 * @brief 计算四边形面积
 *
 * @param pts 四边形顶点
 * @return float 面积
 */
float calcTetragonArea(cv::Point2f pts[4])
{
    return calcTriangleArea(&pts[0]) + calcTriangleArea(&pts[1]);
}

/**
 * @brief Resize the image using letterbox
 * @param img Image before resize
 * @param transform_matrix Transform Matrix of Resize
 * @return Image after resize
 */
inline cv::Mat scaledResize(cv::Mat &img, Eigen::Matrix<float, 3, 3> &transform_matrix)
{
    float r = std::min(YOLOX_INPUT_W / (img.cols * 1.0), YOLOX_INPUT_H / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    int dw = YOLOX_INPUT_W - unpad_w;
    int dh = YOLOX_INPUT_H - unpad_h;

    dw /= 2;
    dh /= 2;

    transform_matrix << 1.0 / r, 0, -dw / r,
        0, 1.0 / r, -dh / r,
        0, 0, 1;

    cv::Mat re;
    cv::resize(img, re, cv::Size(unpad_w, unpad_h));
    cv::Mat out;
    cv::copyMakeBorder(re, out, dh, dh, dw, dw, 0);

    return out;
}

static float *blobFromImage(cv::Mat &img)
{
    float *blob = new float[img.total() * 3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}

/**
 * @brief Generate grids and stride.
 * @param target_w Width of input.
 * @param target_h Height of input.
 * @param strides A vector of stride.
 * @param grid_strides Grid stride generated in this function.
 */
static void generate_grids_and_stride(const int target_w, const int target_h,
                                      std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;

        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

/**
 * @brief Generate Proposal
 * @param grid_strides Grid strides
 * @param feat_ptr Original predition result.
 * @param prob_threshold Confidence Threshold.
 * @param objects Objects proposed.
 */
static void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float *feat_ptr,
                                   Eigen::Matrix<float, 3, 3> &transform_matrix, float prob_threshold,
                                   std::vector<ArmorObject> &objects)
{

    const int num_anchors = grid_strides.size();
    // Travel all the anchors
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (9 + NUM_COLORS + NUM_CLASSES);

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
        float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
        float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
        float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
        float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
        float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
        float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
        float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;

        int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
        int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);

        float box_objectness = (feat_ptr[basic_pos + 8]);

        float color_conf = (feat_ptr[basic_pos + 9 + box_color]);
        float cls_conf = (feat_ptr[basic_pos + 9 + NUM_COLORS + box_class]);

        // float box_prob = (box_objectness + cls_conf + color_conf) / 3.0;
        float box_prob = box_objectness;

        if (box_prob >= prob_threshold)
        {
            ArmorObject obj;

            Eigen::Matrix<float, 3, 4> apex_norm;
            Eigen::Matrix<float, 3, 4> apex_dst;

            apex_norm << x_1, x_2, x_3, x_4,
                y_1, y_2, y_3, y_4,
                1, 1, 1, 1;

            apex_dst = transform_matrix * apex_norm;

            for (int i = 0; i < 4; i++)
            {
                obj.apex[i] = cv::Point2f(apex_dst(0, i), apex_dst(1, i));
                obj.pts.push_back(obj.apex[i]);
            }

            vector<cv::Point2f> tmp(obj.apex, obj.apex + 4);
            obj.rect = cv::boundingRect(tmp);

            obj.cls = box_class;
            obj.color = box_color;
            obj.prob = box_prob;

            objects.push_back(obj);
        }

    } // point anchor loop
}

/**
 * @brief Calculate intersection area between two objects.
 * @param a Object a.
 * @param b Object b.
 * @return Area of intersection.
 */
static inline float intersection_area(const ArmorObject &a, const ArmorObject &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<ArmorObject> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<ArmorObject> &objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(std::vector<ArmorObject> &faceobjects, std::vector<int> &picked,
                              float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        ArmorObject &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            ArmorObject &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area / union_area;
            if (iou > nms_threshold || isnan(iou))
            {
                keep = 0;
                // Stored for Merge
                if (iou > MERGE_MIN_IOU && abs(a.prob - b.prob) < MERGE_CONF_ERROR && a.cls == b.cls && a.color == b.color)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        b.pts.push_back(a.apex[i]);
                    }
                }
                // cout<<b.pts_x.size()<<endl;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

/**
 * @brief Decode outputs.
 * @param prob Original predition output.
 * @param objects Vector of objects predicted.
 * @param img_w Width of Image.
 * @param img_h Height of Image.
 */
static void decodeOutputs(const float *prob, std::vector<ArmorObject> &objects,
                          Eigen::Matrix<float, 3, 3> &transform_matrix, const int img_w, const int img_h)
{
    std::vector<ArmorObject> proposals;
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(YOLOX_INPUT_W, YOLOX_INPUT_H, strides, grid_strides);
    generateYoloxProposals(grid_strides, prob, transform_matrix, BBOX_CONF_THRESH, proposals);
    qsort_descent_inplace(proposals);

    if (proposals.size() >= TOPK)
        proposals.resize(TOPK);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
    }
}

ArmorDetector::ArmorDetector()
{
}

bool ArmorDetector::initModel(string model_path)
{
    this->path_to_engine = model_path;
    std::cout << model_path << std::endl;
    if (!build())
    {
        return false;
    }
    return true;
}

bool ArmorDetector::build()
{
    std::filesystem::path engine_file(this->path_to_engine);
    //    path_to_engine = engine_file.stem();
    if (std::filesystem::exists(engine_file))
    {
        std::cout << "Build from engine file." << std::endl;
        if (!build_from_engine())
        {
            std::cerr << "Build from engine file FAILED." << std::endl;
            return false;
        }
    }
    else
    {
        std::cout << "Not find engine file, start to build from onnx." << std::endl;
        engine_file.replace_extension(".onnx");
        if (std::filesystem::exists(engine_file))
        {
            std::cout << "Build engine from onnx file: " << engine_file << std::endl;
            onnx_file = engine_file.string();
            if (!build_from_onnx())
            {
                std::cerr << "Build from onnx file FAILED." << std::endl;
                return false;
            }
        }
        else
        {
            std::cerr << "Not found onnx file: " << engine_file << std::endl;
            return false;
        }
    }
    return true;
}

bool ArmorDetector::build_from_engine()
{
    cudaSetDevice(this->DEVICE_);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::cout << path_to_engine << std::endl;
    //    path_to_engine = "../model/tup_yolox2.engine";
    std::ifstream file(path_to_engine, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else
    {
        std::cerr << "invalid path_to_engine: " << path_to_engine << std::endl;
        return false;
    }

    auto runtime = SampleUniquePtr<IRuntime>(createInferRuntime(gLogger_));
    assert(runtime != nullptr);
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, size), InferDeleter());
    assert(engine_ != nullptr);
    this->context_ = SampleUniquePtr<IExecutionContext>(engine_->createExecutionContext());
    assert(context_ != nullptr);
    delete trtModelStream;

    auto input_dims = engine_->getBindingDimensions(inputIndex_);
    this->input_h_ = input_dims.d[2];
    this->input_w_ = input_dims.d[3];
    std::cout << "INPUT_HEIGHT: " << this->input_h_ << std::endl;
    std::cout << "INPUT_WIDTH: " << this->input_w_ << std::endl;

    auto out_dims = engine_->getBindingDimensions(outputIndex_);
    this->output_size_ = 1;
    for (int j = 0; j < out_dims.nbDims; ++j)
    {
        this->output_size_ *= out_dims.d[j];
    }

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine_->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    if (fp16)
    {
        assert(engine_->getBindingDataType(this->inputIndex_) == nvinfer1::DataType::kHALF);
        assert(engine_->getBindingDataType(this->outputIndex_) == nvinfer1::DataType::kHALF);
    }
    else
    {
        assert(engine_->getBindingDataType(this->inputIndex_) == nvinfer1::DataType::kFLOAT);
        assert(engine_->getBindingDataType(this->outputIndex_) == nvinfer1::DataType::kFLOAT);
    }

    // // Prepare GridAndStrides
    // if(this->p6_)
    // {
    //     generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_p6_, this->grid_strides_);
    // }
    // else
    // {
    //     generate_grids_and_stride(this->input_w_, this->input_h_, this->strides_, this->grid_strides_);
    // }
    return true;
}

bool ArmorDetector::build_from_onnx()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger_.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    // cunstruct net work
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger_.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto parsed = parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger_.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (builder->platformHasFastFp16())
    {
        std::cout << "[INFO]: platform support fp16, enable fp16" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
        fp16 = true;
    }
    else
    {
        std::cout << "[INFO]: platform do not support fp16, enable fp32" << std::endl;
    }

    // if (int8)
    // {
    //     config->setFlag(BuilderFlag::kINT8);
    //     samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    // }

    // ArmorDetector::enableDLA(builder.get(), config.get(), this->dlaCore);

    // CUDA stream used for profiling by the builder.
    auto profileStream = makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "[INFO]: total gpu mem: " << (total >> 20) << "MB, free gpu mem: " << (free >> 20) << "MB" << std::endl;
    std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(free);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(gLogger_.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    if (!engine_)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 3);

    std::cout << "[INFO]: Completed creating Engine" << std::endl;

    // nvinfer1::IHostMemory* model_data = engine_->serialize();
    FILE *f = fopen(path_to_engine.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), f);
    fclose(f);
    std::cout << "[INFO]: Saved engine file path: " << path_to_engine << std::endl;

    return true;
}

void ArmorDetector::doInference(float *input, float *output, const int output_size, cv::Size input_shape)
{
    // // const ICudaEngine &engine = context->getEngine();
    // const ICudaEngine &engine = context->getEngine();

    // // Pointers to input and output device buffers to pass to engine.
    // // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    //    const int inputIndex = 0;

    //    assert(engine_->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    // const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    //    const int outputIndex = 1;
    //    assert(engine_->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    //    int mBatchSize = engine.getMaxBatchSize();

    //    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex_], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_], output_size * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex_], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    this->context_->enqueueV2(buffers, stream, nullptr);
    // context->enqueueV3(stream);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex_], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex_]));
    CHECK(cudaFree(buffers[outputIndex_]));
}

cv::Mat ArmorDetector::detect(std::vector<ArmorObject> &objects, cv::Mat &src)
{

    // preprocess
    int img_w = src.cols;
    int img_h = src.rows;
    cv::Mat pr_img = scaledResize(src, transform_matrix);
    cv::Mat res_img = src.clone();

    //    float *input_blob = new float[pr_img.total() * 3];
    float *blob;
    float *prob = new float[output_size_];
    blob = blobFromImage(pr_img);

    // run inference
    auto start = std::chrono::steady_clock::now();
    doInference(blob, prob, output_size_, pr_img.size());

    //    std::vector<ArmorObject> objects;
    decodeOutputs(prob, objects, transform_matrix, img_w, img_h);
    if (objects.empty())
    {
        // fmt::print(fmt::fg(fmt::color::red), "[DETECT] ERROR: 传入了空的src\n");
        std::cerr << "[DET]: Input empty src image" << std::endl;
        delete prob;
        delete blob;
        return src;
    }
    // draw_objects(src, objects, input_image_path);
    for (auto object = objects.begin(); object != objects.end(); ++object)
    {
        // 对候选框预测角点进行平均,降低误差
        if ((*object).pts.size() >= 8)
        {
            auto N = (*object).pts.size();
            cv::Point2f pts_final[4];

            for (int i = 0; i < N; i++)
            {
                pts_final[i % 4] += (*object).pts[i];
            }

            for (int i = 0; i < 4; i++)
            {
                pts_final[i].x = pts_final[i].x / (N / 4);
                pts_final[i].y = pts_final[i].y / (N / 4);
            }

            (*object).apex[0] = pts_final[0];
            (*object).apex[1] = pts_final[1];
            (*object).apex[2] = pts_final[2];
            (*object).apex[3] = pts_final[3];
        }
        (*object).area = (int)(calcTetragonArea((*object).apex));
#ifdef SHOW_ALL_ARMOR
        std::vector<cv::Point2f> tmp((*object).apex, (*object).apex + 4);
        (*object).rect = cv::boundingRect(tmp);
        auto end = std::chrono::steady_clock::now();
        auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[DET]: cost: " << cost << "ms" << std::endl;
        cv::rectangle(res_img, (*object).rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(res_img, std::to_string(cost), cv::Point2i(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0));
#endif
    }
    
    delete prob;
    delete blob;
    
    return res_img;
}

ArmorDetector::~ArmorDetector()
{
    // context_->destroy();
    // engine_->destroy();
    // runtime_->destroy();
    // 使用InferDeleter()函数delete 报错Destroying an engine object before destroying objects it created leads to undefined behavior.
    context_.reset();
    engine_.reset();
}
