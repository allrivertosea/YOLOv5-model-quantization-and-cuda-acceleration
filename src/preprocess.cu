#include "cuda_runtime_api.h"
#include "stdio.h"
#include <iostream>
#include "preprocess.hpp"

namespace preprocess{

TransInfo    trans;
AffineMatrix affine_matrix;

void warpaffine_init(int srcH, int srcW, int tarH, int tarW){
    trans.src_h = srcH;
    trans.src_w = srcW;
    trans.tar_h = tarH;
    trans.tar_w = tarW;
    affine_matrix.init(trans);
}

__host__ __device__ void affine_transformation(
    float trans_matrix[6], 
    int src_x, int src_y, 
    float* tar_x, float* tar_y)
{
    *tar_x = trans_matrix[0] * src_x + trans_matrix[1] * src_y + trans_matrix[2];
    *tar_y = trans_matrix[3] * src_x + trans_matrix[4] * src_y + trans_matrix[5];
}



__global__ void warpaffine_BGR2RGB_kernel(
    float* tar, uint8_t* src, 
    TransInfo trans,
    AffineMatrix affine_matrix)
{
    float src_x, src_y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    affine_transformation(affine_matrix.reverse, x + 0.5, y + 0.5, &src_x, &src_y);

    int src_x1 = floor(src_x - 0.5);
    int src_y1 = floor(src_y - 0.5);
    int src_x2 = src_x1 + 1;
    int src_y2 = src_y1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y1 > trans.src_h || src_x1 > trans.src_w) {
    } else {
        float tw   = src_x - src_x1;
        float th   = src_y - src_y1;

        float a1_1 = (1.0 - tw) * (1.0 - th);
        float a1_2 = tw * (1.0 - th);
        float a2_1 = (1.0 - tw) * th;
        float a2_2 = tw * th;

        int srcIdx1_1 = (src_y1 * trans.src_w + src_x1) * 3;
        int srcIdx1_2 = (src_y1 * trans.src_w + src_x2) * 3;
        int srcIdx2_1 = (src_y2 * trans.src_w + src_x1) * 3;
        int srcIdx2_2 = (src_y2 * trans.src_w + src_x2) * 3;

        int tarIdx    = y * trans.tar_w  + x;
        int tarArea   = trans.tar_w * trans.tar_h;

        tar[tarIdx + tarArea * 0] = 
            round((a1_1 * src[srcIdx1_1 + 2] + 
                   a1_2 * src[srcIdx1_2 + 2] +
                   a2_1 * src[srcIdx2_1 + 2] +
                   a2_2 * src[srcIdx2_2 + 2])) / 255.0f;

        tar[tarIdx + tarArea * 1] = 
            round((a1_1 * src[srcIdx1_1 + 1] + 
                   a1_2 * src[srcIdx1_2 + 1] +
                   a2_1 * src[srcIdx2_1 + 1] +
                   a2_2 * src[srcIdx2_2 + 1])) / 255.0f;

        tar[tarIdx + tarArea * 2] = 
            round((a1_1 * src[srcIdx1_1 + 0] + 
                   a1_2 * src[srcIdx1_2 + 0] +
                   a2_1 * src[srcIdx2_1 + 0] +
                   a2_2 * src[srcIdx2_2 + 0])) / 255.0f;
    }
}


void resize_bilinear_gpu(
    float* d_tar, uint8_t* d_src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    tactics tac) 
{
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(tarW / 32 + 1, tarH / 32 + 1, 1);
   
    //scaled resize
    float scaled_h = (float)srcH / tarH;
    float scaled_w = (float)srcW / tarW;
    float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);

    switch (tac) {

    case tactics::GPU_WARP_AFFINE:
        warpaffine_init(srcH, srcW, tarH, tarW);
        warpaffine_BGR2RGB_kernel <<<dimGrid, dimBlock>>> 
                (d_tar, d_src, trans, affine_matrix);
        break;
    default:
        LOGE("ERROR: Wrong GPU resize tactics selected. Program terminated");
        exit(1);
    }
}

} // namespace preprocess
