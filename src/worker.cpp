#include "worker.hpp"
#include "detector.hpp"
#include "logger.hpp"
#include "memory"

using namespace std;

namespace thread{

Worker::Worker(string onnxPath, logger::Level level, model::Params params) {
    m_logger = logger::create_logger(level);

    // 这里根据task_type选择创建的子类，可扩充任务类型
    if (params.task == model::task_type::DETECTION) 
        m_detector = model::detector::make_detector(onnxPath, level, params);
}

void Worker::inference(string imagePath) {

    if (m_detector != nullptr) {
        m_detector->init_model();
        m_detector->load_image(imagePath);
        m_detector->inference();
    }
}

shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params) 
{
    // 使用智能指针来创建一个实例
    return make_shared<Worker>(onnxPath, level, params);
}

}; // namespace thread
