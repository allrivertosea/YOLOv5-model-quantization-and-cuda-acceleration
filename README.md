# yolov5-quantization-and-infer
Quantization and deployment of YOLOv5 vehicle and pedestrian 8-object detection model using TensorRT, with CUDA programming for accelerating pre-processing and post-processing.
![INT8推理结果](https://raw.githubusercontent.com/allrivertosea/YOLOv5-model-quantization-and-inference/main/data/result/car4-detect-int8.png)
## 项目简介

- 基于**Tensorrt and CUDA**加速**Yolov5.6.0**
- 支持**Ubuntu20.04**
- 支持**C++**

## 环境说明

- Tensorrt 8.6.1.6
- Cuda 11.7 Cudnn 8.9.0
- Opencv 4.5.5
- Cmake 3.16.3
- RTX 3090

## 使用说明

```
make
cd bin
./od-trt-infer
```

## 推理性能

CUDA编程加速说明：前处理resize和BGR2RGB、后处理decode、affine和NMS均编写核函数进行CUDA加速

TensorRT加速说明：模型转换为onnx格式后，量化到INT8精度

前处理和后处理CUDA加速，INT8模型推理加速
![CUDA加速](https://raw.githubusercontent.com/allrivertosea/YOLOv5-model-quantization-and-inference/main/data/前处理和后处理CUDA编程加速.png)

前处理和后处理CPU，INT8模型推理加速
![CPU计算](https://raw.githubusercontent.com/allrivertosea/YOLOv5-model-quantization-and-inference/main/data/前处理和后处理CPU.png)
