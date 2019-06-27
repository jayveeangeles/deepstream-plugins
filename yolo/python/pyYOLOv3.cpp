#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "trt_utils.h"
#include "ds_image.h"
#include "yolo.h"
#include "yolov3.h"
#include "yolo_config_parser.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

namespace py = pybind11;

void getParams(py::dict networkInfo, py::dict inferParams, NetworkInfo *yoloInfo, InferParams *yoloInferParams)
{
  for (auto item : networkInfo)
  {
    if (std::string(py::str(item.first)) == "networkType")
      yoloInfo->networkType = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "configFilePath")
      yoloInfo->configFilePath = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "wtsFilePath")
      yoloInfo->wtsFilePath = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "labelsFilePath")
      yoloInfo->labelsFilePath = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "precision")
      yoloInfo->precision = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "deviceType")
      yoloInfo->deviceType = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "calibrationTablePath")
      yoloInfo->calibrationTablePath = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "enginePath")
      yoloInfo->enginePath = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "inputBlobName")
      yoloInfo->inputBlobName = std::string(py::str(item.second));
    else continue;
  }

  for (auto item : inferParams)
  {
    if (std::string(py::str(item.first)) == "calibImages")
      yoloInferParams->calibImages = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "calibImagesPath")
      yoloInferParams->calibImagesPath = std::string(py::str(item.second));
    else if (std::string(py::str(item.first)) == "printPerfInfo")
      yoloInferParams->printPerfInfo = item.second.cast<py::bool_>();
    else if (std::string(py::str(item.first)) == "printPredictionInfo")
      yoloInferParams->printPredictionInfo = item.second.cast<py::bool_>();
    else if (std::string(py::str(item.first)) == "probThresh")
      yoloInferParams->probThresh = item.second.cast<py::float_>();
    else if (std::string(py::str(item.first)) == "nmsThresh")
      yoloInferParams->nmsThresh = item.second.cast<py::float_>();
    else continue;
  }
}

PYBIND11_MODULE(YoloV3, m) {
  py::class_<Yolo>(m, "Yolo");

  py::class_<BBox>(m, "BBox")
    .def(py::init<>())
    .def_readwrite("x1", &BBox::x1)
    .def_readwrite("y1", &BBox::y1)
    .def_readwrite("x2", &BBox::x2)
    .def_readwrite("y2", &BBox::y2);

  py::class_<BBoxInfo>(m, "BBoxInfo")
    .def(py::init<>())
    .def_readwrite("box", &BBoxInfo::box)
    .def_readwrite("label", &BBoxInfo::label)
    .def_readwrite("prob", &BBoxInfo::prob);

  py::class_<YoloV3, Yolo>(m, "YoloV3")
    .def(py::init<const uint &, const NetworkInfo &, const InferParams &>())
    .def(py::init([](uint batchSize, py::dict networkInfo, py::dict inferParams) {
        NetworkInfo yoloInfo;
        InferParams yoloInferParams;

        getParams(networkInfo, inferParams, &yoloInfo, &yoloInferParams);

        // we don't have to upcast here like in the example since we're only using this
        // the base class for YoloV3 purposes, we can have another class for YoloV2 instead
        return std::unique_ptr<YoloV3>(new YoloV3(batchSize, yoloInfo, yoloInferParams));
    }))
    .def("getInputH", &YoloV3::getInputH)
    .def("getInputW", &YoloV3::getInputW)
    .def("doInference", &YoloV3::doInference)
    .def("decodeDetections", &YoloV3::decodeDetections)
    .def("getNMSThresh", &YoloV3::getNMSThresh)
    .def("getNumClasses", &YoloV3::getNumClasses)
    .def("getClassName", &YoloV3::getClassName)
    .def("getNetworkType", &YoloV3::getNetworkType)
    .def("detect", [](YoloV3 &self, py::array_t<unsigned char, py::array::c_style | py::array::forcecast> image){
      py::buffer_info buf1 = image.request();

      cv::Mat matx(static_cast<int>(buf1.shape[0]), static_cast<int>(buf1.shape[1]), CV_8UC3, (void *)buf1.ptr);
      cv::Mat letterBoxImage = getLetterBoxBlob(matx, self.getInputH(), self.getInputW());

      self.doInference((unsigned char*) letterBoxImage.data, 1);

      auto binfo = self.decodeDetections(0, matx.rows, matx.cols);
      auto remaining = nmsAllClasses(self.getNMSThresh(), binfo, self.getNumClasses());

      return remaining;
    });
}