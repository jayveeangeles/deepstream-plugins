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

  py::class_<NetworkInfo>(m, "NetworkInfo")
    .def(py::init<>())
    .def_readwrite("networkType", &NetworkInfo::networkType)
    .def_readwrite("configFilePath", &NetworkInfo::configFilePath)
    .def_readwrite("wtsFilePath", &NetworkInfo::wtsFilePath)
    .def_readwrite("labelsFilePath", &NetworkInfo::labelsFilePath)
    .def_readwrite("precision", &NetworkInfo::precision)
    .def_readwrite("deviceType", &NetworkInfo::deviceType)
    .def_readwrite("calibrationTablePath", &NetworkInfo::calibrationTablePath)
    .def_readwrite("enginePath", &NetworkInfo::enginePath)
    .def_readwrite("inputBlobName", &NetworkInfo::inputBlobName);
  
  py::class_<InferParams>(m, "InferParams")
    .def(py::init<>())
    .def_readwrite("printPerfInfo", &InferParams::printPerfInfo)
    .def_readwrite("printPredictionInfo", &InferParams::printPredictionInfo)
    .def_readwrite("calibImages", &InferParams::calibImages)
    .def_readwrite("calibImagesPath", &InferParams::calibImagesPath)
    .def_readwrite("probThresh", &InferParams::probThresh)
    .def_readwrite("nmsThresh", &InferParams::nmsThresh);

  py::class_<YoloV3, Yolo>(m, "YoloV3")
    .def(py::init<const uint &, const NetworkInfo &, const InferParams &>())
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