#include "model.hpp"
#include "logger.hpp"
#include "worker.hpp"
#include "utils.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    string onnxPath    = "/APP/cudaprj/deploy-yolov5-od/yolov5-calibration-and-infer/models/onnx/yolov5n_od.onnx";

    auto level         = logger::Level::VERB;
    auto params        = model::Params();

    params.img         = {288, 512, 3};
    params.task        = model::task_type::DETECTION;
    params.dev         = model::device::GPU;
    params.prec        = model::precision::INT8;

    auto worker   = thread::create_worker(onnxPath, level, params);

    worker->inference("/APP/cudaprj/deploy-yolov5-od/yolov5-calibration-and-infer/data/source/car1.jpg");
    worker->inference("/APP/cudaprj/deploy-yolov5-od/yolov5-calibration-and-infer/data/source/car2.jpg");
    worker->inference("/APP/cudaprj/deploy-yolov5-od/yolov5-calibration-and-infer/data/source/car3.jpg");
    worker->inference("/APP/cudaprj/deploy-yolov5-od/yolov5-calibration-and-infer/data/source/car4.jpg");

    return 0;
}

