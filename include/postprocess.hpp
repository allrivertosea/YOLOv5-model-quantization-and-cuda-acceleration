#ifndef TRT_POSTPROCESS_HPP
#define TRT_POSTPROCESS_HPP

#include <vector>
#include "detector.hpp"

namespace postprocess {
    struct TransInfo{
    int src_w = 0;
    int src_h = 0;
    int tar_w = 0;
    int tar_h = 0;
    TransInfo() = default;
    TransInfo(int srcW, int srcH, int tarW, int tarH):
        src_w(srcW), src_h(srcH), tar_w(tarW), tar_h(tarH){}
};

struct AffineMatrix{
    float forward[6];
    float reverse[6];
    float forward_scale;
    float reverse_scale;

    void calc_forward_matrix(TransInfo trans){
        forward[0] = forward_scale;
        forward[1] = 0;
        forward[2] = - forward_scale * trans.src_w * 0.5 + trans.tar_w * 0.5;
        forward[3] = 0;
        forward[4] = forward_scale;
        forward[5] = - forward_scale * trans.src_h * 0.5 + trans.tar_h * 0.5;
    };

    void calc_reverse_matrix(TransInfo trans){
        reverse[0] = reverse_scale;
        reverse[1] = 0;
        reverse[2] = - reverse_scale * trans.tar_w * 0.5 + trans.src_w * 0.5;
        reverse[3] = 0;
        reverse[4] = reverse_scale;
        reverse[5] = - reverse_scale * trans.tar_h * 0.5 + trans.src_h * 0.5;
    };

    void init(TransInfo trans){
        float scaled_w = (float)trans.tar_w / trans.src_w;
        float scaled_h = (float)trans.tar_h / trans.src_h;
        forward_scale = (scaled_w < scaled_h ? scaled_w : scaled_h);
        reverse_scale = 1 / forward_scale;
    
        calc_forward_matrix(trans);
        calc_reverse_matrix(trans);
    }
};

// 对结构体设置default instance
extern  TransInfo    trans;
extern  AffineMatrix affine_matrix;

// GPU解码和NMS函数声明
std::vector<model::detector::bbox> decode_bbox_gpu(float* output, int boxes_count, int class_count);
void decode_kernel_trigger(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

// 核函数声明
static __global__ void decode_kernel(
    float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float *parray, int max_objects, int NUM_BOX_ELEMENT);

static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT);
static __device__ float iou_calc(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom);
static __global__ void affine_kernel(float* src, 
    AffineMatrix affine_matrix,int NUM_BOX_ELEMENT,int max_objects);
__host__ __device__ void affine_transformation(
    float trans_matrix[6], 
    int src_x, int src_y, 
    float* tar_x, float* tar_y);
static __device__ bool compare_bboxes(const float *b1, const float *b2);
}


#endif // TRT_POSTPROCESS_HPP
