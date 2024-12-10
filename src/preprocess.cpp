#include "opencv2/opencv.hpp"
#include "preprocess.hpp"
#include "utils.hpp"
#include "timer.hpp"

namespace preprocess {


// 根据比例进行缩放 (GPU版本)
void preprocess_resize_gpu(
    cv::Mat &h_src, float* d_tar, 
    const int& tar_h, const int& tar_w, 
    tactics tac) 
{
    uint8_t* d_src  = nullptr;

    int height   = h_src.rows;
    int width    = h_src.cols;
    int chan     = 3;

    int src_size  = height * width * chan * sizeof(uint8_t);
    int norm_size = 3 * sizeof(float);


    // 分配device上的src的内存
    CUDA_CHECK(cudaMalloc(&d_src, src_size));

    // 将数据拷贝到device上
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data, src_size, cudaMemcpyHostToDevice));

    // device上处理resize, BGR2RGB的核函数
    resize_bilinear_gpu(d_tar, d_src, tar_w, tar_h, width, height, tac);

    // host和device进行同步处理
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_src));
}

} // namespace process
