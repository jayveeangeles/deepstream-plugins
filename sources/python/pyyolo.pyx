# distutils: language = c++
# distutils: sources = bridge.cpp

from __future__ import division, print_function, absolute_import

import numpy as np

import logging
from libc.string cimport memcpy
from libc.stdlib cimport malloc
from libc.string cimport strcpy, strlen
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

from cython.operator cimport dereference as deref
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC_RAW

cdef class PyYoloV3:
    cdef unique_ptr[YoloV3] thisptr
    cdef NetworkInfo yoloInfo
    cdef InferParams yoloInferParams
    cdef unsigned int batchSize
    cdef string saveDetectionsPath
    cdef bool saveDetections

    def __cinit__(self, args):
        cdef int argc = len(args)
        cdef char** argv = <char**> malloc(len(args) * sizeof(char*))

        for i, arg in enumerate(args):
            argv[i] = <char *> malloc((len(arg) + 1) * sizeof(char))
            strcpy(argv[i], arg.encode('utf-8'))

        yoloConfigParserInit(argc, argv)
        self.yoloInfo = getYoloNetworkInfo()
        self.yoloInferParams = getYoloInferParams()
        self.batchSize = getBatchSize()
        self.saveDetections = getSaveDetections()
        self.saveDetectionsPath = getSaveDetectionsPath()

        self.thisptr.reset(new YoloV3(self.batchSize, self.yoloInfo, self.yoloInferParams))

    # Code adapted from https://github.com/pjreddie/darknet/blob/master/python/darknet.py

    def getNetworkType(self):
        return getNetworkType()

    def getPrecision(self):
        return getPrecision()

    def getYoloNetworkInfo(self):
        return self.yoloInfo
    
    def getYoloInferParams(self):
        return self.yoloInferParams

    def detect(self, np.ndarray ary, string imageName):
        # Code adapted from https://github.com/solivr/cython_opencvMat
        assert ary.ndim==3 and ary.shape[2]==3, "ASSERT::3channel RGB only!!"
        cdef timespec tp1
        cdef timespec tp2
        cdef float t_diff
        
        clock_gettime(CLOCK_MONOTONIC_RAW, &tp1)
        cdef np.ndarray[np.uint8_t, ndim=3, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.uint8)
        cdef unsigned int* im_buff = <unsigned int*> np_buff.data
        cdef int r = ary.shape[0]
        cdef int c = ary.shape[1]
        cdef Mat m
        m.create(r, c, CV_8UC3)
        memcpy(m.data, im_buff, r*c*3)
        # End of adapted code block

        # convert image from 
        cdef DsImageSingle dsImage = DsImageSingle(m, imageName, deref(self.thisptr).getInputH(), deref(self.thisptr).getInputW())
        cdef Mat trtInput = blobFromDsImage(dsImage, deref(self.thisptr).getInputH(), deref(self.thisptr).getInputW())
        
        deref(self.thisptr).doInference(<unsigned char *> trtInput.data)

        cdef vector[BBoxInfo] binfo = deref(self.thisptr).decodeDetections(0, dsImage.getImageHeight(), dsImage.getImageWidth())
        cdef vector[BBoxInfo] remaining = nonMaximumSuppression(deref(self.thisptr).getNMSThresh(), binfo)

        cdef unsigned int predLen = remaining.size()
        predictions = []
        cdef string className

        IF YOLO34PY_FORMAT:
            cdef float width, height, xmid, ymid

            for idx in range(predLen):
                className = deref(self.thisptr).getClassName(remaining[idx].label)
                width = remaining[idx].box.x2 - remaining[idx].box.x1
                height = remaining[idx].box.y2 - remaining[idx].box.y1
                xmid = remaining[idx].box.x1 + width/2
                ymid = remaining[idx].box.y1 + height/2
                predictions.append((className, remaining[idx].prob, (xmid, ymid, width, height)))
                if (self.saveDetections):
                    dsImage.addBBox( remaining[idx], className )
        ELSE:
            for idx in range(predLen):
                className = deref(self.thisptr).getClassName(remaining[idx].label)
                predictions.append({
                    "box": {
                        "x1": remaining[idx].box.x1,
                        "x2": remaining[idx].box.x2,
                        "y1": remaining[idx].box.y1,
                        "y2": remaining[idx].box.y2
                    },
                    "label": className.decode('utf-8'),
                    "prob": remaining[idx].prob
                })
                if (self.saveDetections):
                    dsImage.addBBox( remaining[idx], className )

        if (self.saveDetections):
            dsImage.saveImageJPEG( self.saveDetectionsPath )
        
        m.release()
        clock_gettime(CLOCK_MONOTONIC_RAW, &tp2)
        t_diff = ((tp2.tv_sec-tp1.tv_sec)*(1000) + (tp2.tv_nsec-tp1.tv_nsec)/1000000.0)
        
        if self.yoloInferParams.printPerfInfo:
            logging.info("inference time: %f" % t_diff)

        return predictions

    # End of adapted code block

    def __dealloc__(self):
        self.thisptr.reset()