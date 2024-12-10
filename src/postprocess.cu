#include "postprocess.hpp"
#include "preprocess.hpp"
#include <vector>
#include <cuda_runtime.h>
#include "detector.hpp"

#ifndef checkRuntime
#define checkRuntime(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            assert(0);\
        }\
    }
#endif  // checkRuntime

namespace postprocess{
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


static __global__ void decode_kernel(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
    float *parray, int max_objects, int NUM_BOX_ELEMENT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bboxes) return;

    float* tensor = predict + idx * (num_classes + 5); 

    float obj_conf = tensor[4];
    if (obj_conf < confidence_threshold) return;

    int label = -1;
    float max_conf = -1.0f;
    for (int i = 5; i < num_classes + 5; ++i) {
        if (tensor[i] > max_conf) {
            max_conf = tensor[i];
            label = i - 5;
        }
    }
    float confidence = max_conf * obj_conf;
    if (confidence < confidence_threshold) return;

    float cx = tensor[0];
    float cy = tensor[1];
    float w = tensor[2];
    float h = tensor[3];

    float x0 = cx - w / 2;
    float y0 = cy - h / 2;
    float x1 = x0 + w;
    float y1 = y0 + h;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)  
        return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = x0;
    *pout_item++ = y0;
    *pout_item++ = x1;
    *pout_item++ = y1;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
    
}

std::vector<model::detector::bbox> decode_bbox_gpu(float* output, int boxes_count, int class_count) {
    std::vector<model::detector::bbox> box_result;
 
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    float *predict_device = nullptr;
    float *output_device = nullptr;
    float *output_host = nullptr;
    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 13;

    checkRuntime(cudaMalloc(&predict_device, boxes_count * 13 * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(predict_device, output, boxes_count * 13 * sizeof(float), cudaMemcpyHostToDevice, stream));
   
    float conf_threshold = 0.3; 
    float nms_threshold  = 0.5;

    decode_kernel_trigger(predict_device, boxes_count, class_count, conf_threshold,
        nms_threshold, output_device, max_objects, NUM_BOX_ELEMENT, stream);

    // 使用cudaMemcpyAsync将解码好的output_device拷贝到CPU上面去
    checkRuntime(cudaMemcpyAsync(output_host, output_device,
                                 sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));

    // 同步
    checkRuntime(cudaStreamSynchronize(stream));
    int num_boxes = min((int)output_host[0], max_objects);
    // 遍历每一个box
    for (int i = 0; i < num_boxes; i++)
    {   
        float *ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6]; 
        if (keep_flag)
        {
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]);
        }
    }
    // 销毁先前创建的CUDA流对象, 释放流对象占用的内存空间
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device)); 
    checkRuntime(cudaFree(output_device)); 
    checkRuntime(cudaFreeHost(output_host)); 
    return box_result;
}


static __global__ void affine_kernel(float* src, 
    AffineMatrix affine_matrix, int NUM_BOX_ELEMENT, int max_objects)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_objects) return;

    float *pcurrent = src + 1 + idx * NUM_BOX_ELEMENT;
    float x0 = pcurrent[0];
    float y0 = pcurrent[1];
    float x1 = pcurrent[2];
    float y1 = pcurrent[3];

    affine_transformation(affine_matrix.reverse, x0, y0, &x0, &y0);
    pcurrent[0] = x0;
    pcurrent[1] = y0;

    affine_transformation(affine_matrix.reverse, x1, y1, &x1, &y1);
    pcurrent[2] = x1;
    pcurrent[3] = y1;
}



void decode_kernel_trigger(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    // 确保每个block的线程不超过512，每个block最多1024
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    /* 解码部分 */
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold, 
        parray, max_objects, NUM_BOX_ELEMENT
    );
    /* 对目标框进行仿射变换 */
    int srch=720, srcw=1280, tarh=288, tarw=512;
    warpaffine_init(srch, srcw, tarh, tarw);
    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    affine_kernel<<<grid, block, 0, stream>>>(parray,affine_matrix,NUM_BOX_ELEMENT,max_objects);
    /* 进行NMS非极大值抑制 */
    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}


static __device__ float iou_calc(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom)
{

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __device__ bool compare_bboxes(const float *b1, const float *b2) {
    return b1[4] > b2[4]; // 按置信度（bboxes[4]）从大到小排序
}


static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = min((int)*bboxes, max_objects);
    if (idx >= count) return;
   
    float *pcurrent = bboxes + 1  + idx * NUM_BOX_ELEMENT;
    // 遍历每一个bbox
    for (int i = 0; i < count; ++i){
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        // NMS计算需要保证类别必须相同
        if (i == idx || pcurrent[5] != pitem[5]){
            continue;
        }
        
        if (pitem[4] > pcurrent[4]){
            if (pitem[4] == pcurrent[4] && i < idx){
                continue;
            }
                
            float iou = iou_calc(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold){
                pcurrent[6] = 0;
                return;
            }
        }
    }
}




}
