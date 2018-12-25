# distutils: language = c++
from __future__ import division, print_function, absolute_import

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "bridge.h":
    cdef int CV_8UC3

    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int)
        void* data
        int rows
        int cols
        int channels()
        void deallocate()
        void release()

    cdef cppclass YoloV3:
        YoloV3() except +
        YoloV3(const unsigned int, const NetworkInfo&, const InferParams&) except +
        string getNetworkType()
        float getNMSThresh()
        string getClassName(const int&)
        int getInputH()
        int getInputW()
        bool isPrintPredictions()
        bool isPrintPerfInfo()
        void doInference(const unsigned char*)
        vector[BBoxInfo] decodeDetections(const int&, const int&, const int&)

    cdef cppclass DsImageSingle:
        DsImageSingle() except +
        DsImageSingle(string&, const int&, const int&) except +
        DsImageSingle(const Mat&, const string, const int&, const int&) except +
        int getImageHeight()
        int getImageWidth()
        Mat getLetterBoxedImage()
        Mat getOriginalImage()
        string getImageName()
        void addBBox(BBoxInfo, const string&)
        void showImage()
        void saveImageJPEG(const string&)

    ctypedef struct NetworkInfo:
        string networkType
        string configFilePath
        string wtsFilePath
        string labelsFilePath
        string precision
        string calibrationTablePath
        string enginePath
        string inputBlobName

    ctypedef struct InferParams:
        bool printPerfInfo
        bool printPredictionInfo
        string calibrationImages
        float probThresh
        float nmsThresh

    cdef struct BBox:
        float x1, y1, x2, y2

    cdef struct BBoxInfo:
        BBox box
        int label
        float prob

    void yoloConfigParserInit(int, char**)
    Mat blobFromDsImage(const DsImageSingle&, const int&, const int&);

    NetworkInfo getYoloNetworkInfo()
    InferParams getYoloInferParams()

    string getNetworkType()
    string getPrecision()

    bool getDecode()
    bool getViewDetections()
    bool getSaveDetections()
    string getSaveDetectionsPath()

    unsigned int getBatchSize()

    vector[BBoxInfo] decodeDetections(const int&, const int&, const int&)
    vector[BBoxInfo] nonMaximumSuppression(const float, vector[BBoxInfo])