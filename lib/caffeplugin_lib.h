/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/
#ifndef __CAFFEPLUGIN_LIB__
#define __CAFFEPLUGIN_LIB__

#include <glib.h>
#include "caffehelper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaffePluginCtx CaffePluginCtx;

typedef struct
{

  std::string network;
  std::string modelPath;
  std::string weightsFile;
  float nms;
  float confidence;
  uint inferLoopLimit;
  uint preprocessDeadline;

} CaffePluginInitParams;

struct CaffePluginCtx
{
  CaffePluginInitParams initParams;
  trt::Caffe<BATCH_SIZE>* inferenceNetwork;
  std::array<cv::Mat, BATCH_SIZE> images;
  std::array<std::vector<trt::DetectionObject>, BATCH_SIZE> results;
};

// Initialize library context
CaffePluginCtx* CaffePluginCtxInit(CaffePluginInitParams*);

// Deinitialize library context
void CaffePluginCtxDeinit(CaffePluginCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif