#include "NvInfer.h"
#include "calibrator.hpp"
#include "utils.hpp"
#include "logger.hpp"
#include "preprocess.hpp"

#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace nvinfer1;

namespace model{

/*
 * calibrator的构造函数
 * 保证数据集的数量可以被batchSize整除
 * calibration在device上进行的，需要分配空间
 */
Int8EntropyCalibrator::Int8EntropyCalibrator(
    const int&    batchSize,
    const string& calibrationDataPath,
    const string& calibrationTablePath,
    const int&    inputSize,
    const int&    inputH,
    const int&    inputW):

    m_batchSize(batchSize),
    m_inputH(inputH),
    m_inputW(inputW),
    m_inputSize(inputSize),
    m_inputCount(batchSize * inputSize),
    m_calibrationTablePath(calibrationTablePath)
{
    m_imageList = loadDataList(calibrationDataPath);
    m_imageList.resize(static_cast<int>(m_imageList.size() / m_batchSize) * m_batchSize);
    std::random_shuffle(m_imageList.begin(), m_imageList.end(), 
                        [](int i){ return rand() % i; });
    CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));
}

bool Int8EntropyCalibrator::getBatch(
    void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)
        return false;

    LOG("%3d/%3d (%3dx%3d): %s", 
        m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());
    
    cv::Mat input_image;
   
    for (int i = 0; i < m_batchSize; i ++){
        
        std::string file_name = m_imageList.at(m_imageIndex++);
      
        if (!file_name.empty() && (file_name.back() == '\n' || file_name.back() == '\r')) {
            std::cout << "File name contains a newline character.\n";
            file_name.erase(file_name.find_last_not_of("\r\n") + 1);
            std::cout << "Cleaned file name: " << file_name << "\n";
        } else {
            std::cout << "File name does not contain a newline character.\n";
        }
        input_image = cv::imread(file_name);
        preprocess::preprocess_resize_gpu(
            input_image, 
            m_deviceInput + i * m_inputSize,
            m_inputH, m_inputW, 
            preprocess::tactics::GPU_BILINEAR_CENTER);
    }

    bindings[0] = m_deviceInput;

    return true;
}
    
/* 
 * 读取calibration table的信息来创建INT8的推理引擎, 
 * 将calibration table的信息存储到calibration cache，这样可以防止每次创建int推理引擎的时候都需要跑一次calibration
 * 如果没有calibration table的话就会直接跳过这一步，之后调用writeCalibrationCache来创建calibration table
 */
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    m_calibrationCache.clear();

    ifstream input(m_calibrationTablePath, ios::binary);
    input >> noskipws;
    if (m_readCache && input.good())
        copy(istream_iterator<char>(input), istream_iterator<char>(), back_inserter(m_calibrationCache));

    length = m_calibrationCache.size();
    if (length){
        LOG("Using cached calibration table to build INT8 trt engine...");
        output = &m_calibrationCache[0];
    }else{
        LOG("Creating new calibration table to build INT8 trt engine...");
        output = nullptr;
    }
    return output;
}

/* 
 * 将calibration cache的信息写入到calibration table中
*/
void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    ofstream output(m_calibrationTablePath, ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}

} // namespace model
