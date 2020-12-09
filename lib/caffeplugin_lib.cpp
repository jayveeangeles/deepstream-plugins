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

#include "caffeplugin_lib.h"
#include "caffetrt.h"
#include "ssdtrt.h"
#include "frcnntrt.h"
#include "caffehelper.h"

#include <iomanip>
#include <sys/time.h>

CaffePluginCtx* CaffePluginCtxInit(CaffePluginInitParams* initParams)
{
  CaffePluginCtx* ctx = new CaffePluginCtx;
  ctx->initParams = *initParams;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  initLibNvInferPlugins(&gLogger, "");

  // ctx->testAtom = gLogger.defineTest("TensorRT", 0, {});
  // gLogger.reportTestStart(ctx->testAtom);

  setReportableSeverity(Logger::Severity::kINFO);

  if (ctx->initParams.network == "FRCNN") {
    trt::FasterRCNNParams params;

    initializeParams(params, prop, ctx->initParams.modelPath, ctx->initParams.weightsFile);

    params.confThre = ctx->initParams.confidence;
    params.nmsThre  = ctx->initParams.nms;
    
    ctx->inferenceNetwork = new trt::FasterRCNN<BATCH_SIZE>(params);
  } else if (ctx->initParams.network == "SSD") {
    trt::SSDParams params;

    initializeParams(params, prop, ctx->initParams.modelPath, ctx->initParams.weightsFile);

    params.confThre = ctx->initParams.confidence;

    ctx->inferenceNetwork = new trt::SSD<BATCH_SIZE>(params);
  } else {
    std::cerr << "Network type not supported" << '\n';
    
    return nullptr;
  }
  
  if (!ctx->inferenceNetwork->initEngine()) {
    std::cerr << "Couldn'\t init engine" << '\n';
    return nullptr;
  }

  std::cout << "engine successfully init" << '\n';

  return ctx;
}

void CaffePluginCtxDeinit(CaffePluginCtx* ctx) {
  // gLogger.reportPass(ctx->testAtom);

  if (ctx->inferenceNetwork) {
    delete ctx->inferenceNetwork;
    ctx->inferenceNetwork = nullptr;
  }

  if (ctx) {
    delete ctx;
    ctx = nullptr;
  }
}