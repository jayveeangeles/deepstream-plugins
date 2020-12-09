/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FRCNN_TRT
#define FRCNN_TRT

#include "caffehelper.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "caffetrt.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>

//! \brief  The FasterRCNN class implements the FasterRCNN sample
//!
//! \details It creates the network using a caffe model
//!
namespace trt 
{
//!
//! \brief The FasterRCNNParams structure groups the additional parameters required by
//!         the FasterRCNN sample.
//!
struct FasterRCNNParams : public CaffeNetworkParams
{
  float nmsThre;
  float confThre;
  int nmsMaxOut = 300;
};

template<int TBatchSize>
class FasterRCNN : public Caffe <TBatchSize>
{
  template <typename T>
  using SampleUniquePtr = typename Caffe<TBatchSize>::template SampleUniquePtr<T>;
public:
  FasterRCNN(FasterRCNNParams params) {
    initClass(std::forward<FasterRCNNParams>(params));
  }

  virtual ~FasterRCNN();

  virtual bool initEngine();

  virtual std::array<std::vector<DetectionObject>, TBatchSize> infer(
    const std::array<cv::Mat, TBatchSize>& images) override { 
      
    return infer(images, mConfThre, mNMSThre); 
  }

  std::array<std::vector<DetectionObject>, TBatchSize> infer(
    const std::array<cv::Mat, TBatchSize>& images, const float confThre, const float nmsThre=0.3f);

private:
  int mNMSMaxOut;
  float mConfThre;
  float mNMSThre;

  cudaStream_t stream;
  cudaEvent_t inputReady;
  // cudaEvent_t start, end;

  //!
  //! \brief Buffer abstraction
  //!
  std::unique_ptr<samplesCommon::BufferManager> buffers;

  //!
  //! \brief TensorRT execution context
  //!
  SampleUniquePtr<nvinfer1::IExecutionContext> context;

  //!
  //! \brief General purpose timer
  //!
  Stopwatch<std::chrono::microseconds, std::chrono::steady_clock> inferTimer;

  //!
  //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
  //!
  void processInput(const samplesCommon::BufferManager& buffers, 
    const std::array<cv::Mat, TBatchSize>& images);

  //!
  //! \brief Init Class, set params
  //!
  void initClass(FasterRCNNParams params);

  //!
  //! \brief Parse Python list of classnames
  //!
  std::vector<std::string> parseClassesFile(void);

  //!
  //! \brief Filters output detections, handles post-processing of bounding boxes and verify results
  //!
  std::array<std::vector<DetectionObject>, TBatchSize> getDetections(
    const samplesCommon::BufferManager& buffers, 
    const std::array<cv::Mat, TBatchSize>& images, 
    const float nmsThreshold, const float scoreThreshold);

  //!
  //! \brief Performs inverse bounding box transform and clipping
  //!
  void bboxTransformInvAndClip(
    const float* rois, const float* deltas, float* predBBoxes, 
    const float* imInfo, const int N, const int numCls);

  //!
  //! \brief Performs non maximum suppression on final bounding boxes
  //!
  std::vector<int> nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, 
    float* bbox, const int classNum, const int numClasses, const float nmsThreshold);
};

//!
//! \brief Destructor
//!
template<int TBatchSize>
FasterRCNN<TBatchSize>::~FasterRCNN() {
  gLogWarning << "Shutting down engine." << '\n';
  cudaStreamDestroy(stream);
  cudaEventDestroy(inputReady);

  // Caffe<TBatchSize>::teardown();

  gLogWarning << "Shut down engine." << '\n';
}

//! \brief Overrides base init engine function
//!
//! \details Overrides base class function + creates an execution context to run inference.
//!          Buffers are now created here instead of RAII. Alloc/Dealloc is not negligible.
//!
//! \return Returns true if the engine and execution context were created
//!
template<int TBatchSize>
inline 
bool FasterRCNN<TBatchSize>::initEngine() {
  auto successInit = Caffe<TBatchSize>::initEngine();

  // context.reset(this->mEngine->createExecutionContext());
  // context = std::move(SampleUniquePtr<nvinfer1::IExecutionContext>(this->mEngine->createExecutionContext()));
  if (!successInit)
    return false;

  context.reset(this->mEngine->createExecutionContext());
  
  if (!context)
    return false;

  buffers = std::make_unique<samplesCommon::BufferManager>(this->mEngine, TBatchSize);

  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK(cudaEventCreate(&inputReady));

  return successInit;
}

//! \brief Init Class
//!
//! \details Inits some configs needed by class
//!
template<int TBatchSize>
inline 
void FasterRCNN<TBatchSize>::initClass(FasterRCNNParams params) {
  this->mNMSMaxOut = params.nmsMaxOut;
  this->mConfThre  = params.confThre;
  this->mNMSThre   = params.nmsThre;

  this->mParams  = std::move(params);
  this->mClasses = this->parseClassesFile();

  this->mParams.outputClsSize = this->mClasses.size();
  
  this->mNbInputs     = 2;
  this->mNetImageMean = cv::Scalar(102.9801f, 115.9465f, 122.7717f); 
}

//! \brief Processes input image/s
//!
//! \details Takes a batch of images and preprocesses them which includes
//!          resizing, converting to float and subtracting the image mean;
//!          for this overload, add im_info values
//!
template<int TBatchSize>
inline 
void FasterRCNN<TBatchSize>::processInput(
  const samplesCommon::BufferManager& buffers, const std::array<cv::Mat, TBatchSize>& images) {

  const int inputH = this->mInputDims.d[1];
  const int inputW = this->mInputDims.d[2];

  // Fill im_info buffer
  float* hostImInfoBuffer = static_cast<float*>(buffers.getHostBuffer("im_info"));
  for (int i = 0; i < TBatchSize; ++i) {      
    hostImInfoBuffer[i * 3]     = float(inputH);     // Number of rows
    hostImInfoBuffer[i * 3 + 1] = float(inputW); // Number of columns
    hostImInfoBuffer[i * 3 + 2] = 1;                 // Image scale
  }

  Caffe<TBatchSize>::processInput(buffers, images);
}

//! \brief Get labels and bounding boxes found in image
//!
//! \details A super function that interprets results from resulting execution and
//!          returns information about the image, like the class and bounding box
//!
//! \return Returns DetectionObject struct containing info about the image/s
//!
template<int TBatchSize>
inline 
std::array<std::vector<DetectionObject>, TBatchSize> FasterRCNN<TBatchSize>::getDetections(
  const samplesCommon::BufferManager& buffers, const std::array<cv::Mat, TBatchSize>& images, 
  const float nmsThreshold, const float scoreThreshold) {

  const int batchSize      = TBatchSize;
  const int outputClsSize  = this->mParams.outputClsSize;
  const int outputBBoxSize = this->mParams.outputClsSize * 4;

  const float* imInfo      = static_cast<const float*>(buffers.getHostBuffer("im_info"));
  const float* deltas      = static_cast<const float*>(buffers.getHostBuffer("bbox_pred"));
  const float* clsProbs    = static_cast<const float*>(buffers.getHostBuffer("cls_prob"));
  float* rois              = static_cast<float*>(buffers.getHostBuffer("rois"));

  // Unscale back to raw image space
  for (int i = 0; i < batchSize; ++i) {
    if (int(imInfo[i * 3 + 2]) != 1)
      for (int j = 0; j < mNMSMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j) {
        rois[i * mNMSMaxOut * 4 + j] /= imInfo[i * 3 + 2];
      }
  }

  std::vector<float> predBBoxes(batchSize * mNMSMaxOut * outputBBoxSize, 0);
  bboxTransformInvAndClip(rois, deltas, predBBoxes.data(), imInfo, batchSize, outputClsSize);

  const int inputH = this->mInputDims.d[1];
  const int inputW = this->mInputDims.d[2];

  // The sample passes if there is at least one detection for each item in the batch
  std::array<std::vector<DetectionObject>, TBatchSize> results;

  for (int i = 0; i < batchSize; ++i) {
    float* bbox         = predBBoxes.data() + i * mNMSMaxOut * outputBBoxSize;
    const float* scores = clsProbs + i * mNMSMaxOut * outputClsSize;

    for (int c = 1; c < outputClsSize; ++c) { // Skip the background
      std::vector<std::pair<float, int>> scoreIndex;
      for (int r = 0; r < mNMSMaxOut; ++r) {
        if (scores[r * outputClsSize + c] >= scoreThreshold) {
          scoreIndex.push_back(std::make_pair(scores[r * outputClsSize + c], r));
          std::stable_sort(scoreIndex.begin(), scoreIndex.end(),
            [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
              return pair1.first > pair2.first;
            });
        }
      }

      // Apply NMS algorithm
      const std::vector<int> indices = nonMaximumSuppression(
                                        scoreIndex, bbox, c, outputClsSize, nmsThreshold);

      const float imScaleW = float(images[i].cols) / (float)inputW;
      const float imScaleH = float(images[i].rows) / (float)inputH;

      // Show results
      for (unsigned k = 0; k < indices.size(); ++k) {
        const int idx = indices[k];
        const samplesCommon::BBox b{
          bbox[idx * outputBBoxSize + c * 4] * imScaleW, 
          bbox[idx * outputBBoxSize + c * 4 + 1] * imScaleH,
          bbox[idx * outputBBoxSize + c * 4 + 2] * imScaleW, 
          bbox[idx * outputBBoxSize + c * 4 + 3] * imScaleH
        };
        results[i].push_back({b, this->mClasses[c], scores[idx * outputClsSize + c]});
      }
    }
  }
  return results;
}

//!
//! \brief Performs inverse bounding box transform
//!
template<int TBatchSize>
inline 
void FasterRCNN<TBatchSize>::bboxTransformInvAndClip(const float* rois, 
  const float* deltas, float* predBBoxes, const float* imInfo, const int N, const int numCls) {
  for (int i = 0; i < N * mNMSMaxOut; ++i) {
    float width  = rois[i * 4 + 2] - rois[i * 4] + 1;
    float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
    float ctr_x  = rois[i * 4] + 0.5f * width;
    float ctr_y  = rois[i * 4 + 1] + 0.5f * height;

    const float* imInfo_offset = imInfo + i / mNMSMaxOut * 3;

    for (int j = 0; j < numCls; ++j) {
      float dx = deltas[i * numCls * 4 + j * 4];
      float dy = deltas[i * numCls * 4 + j * 4 + 1];
      float dw = deltas[i * numCls * 4 + j * 4 + 2];
      float dh = deltas[i * numCls * 4 + j * 4 + 3];

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;

      float pred_w = exp(dw) * width;
      float pred_h = exp(dh) * height;

      predBBoxes[i * numCls * 4 + j * 4]
        = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
      predBBoxes[i * numCls * 4 + j * 4 + 1]
        = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
      predBBoxes[i * numCls * 4 + j * 4 + 2]
        = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
      predBBoxes[i * numCls * 4 + j * 4 + 3]
        = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
    }
  }
}

//!
//! \brief Performs non maximum suppression on final bounding boxes
//!
template<int TBatchSize>
inline 
std::vector<int> FasterRCNN<TBatchSize>::nonMaximumSuppression(std::vector<std::pair<float, int>>& scoreIndex, float* bbox,
  const int classNum, const int numClasses, const float nmsThreshold) {
  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min) {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };

  auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
    float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
    float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);

    float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

    float overlap2D = overlapX * overlapY;
    float u         = area1 + area2 - overlap2D;

    return u == 0 ? 0 : overlap2D / u;
  };

  std::vector<int> indices;
  for (auto i : scoreIndex) {
    const int idx = i.second;
    bool keep = true;
    for (unsigned k = 0; k < indices.size(); ++k) {
      if (keep) {
        const int kept_idx = indices[k];
        float overlap = computeIoU(
          &bbox[(idx * numClasses + classNum) * 4], &bbox[(kept_idx * numClasses + classNum) * 4]);
        keep = overlap <= nmsThreshold;
      } else break;
    }
    if (keep) {
      indices.push_back(idx);
    }
  }
  return indices;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample.
//!          It sets inputs and executes the engine.
//!
//! \return returns an array of detection objects which include bounding box info
//!         and labels related to the object
//!
template<int TBatchSize>
inline 
std::array<std::vector<DetectionObject>, TBatchSize> FasterRCNN<TBatchSize>::infer(
    const std::array<cv::Mat, TBatchSize>& images, const float confThre, const float nmsThre) {

  using namespace std::chrono_literals;
  // Create RAII buffer manager object, move away from RAII

  assert(this->mParams.inputTensorNames.size() == this->mNbInputs);

  // Read the input data into the managed buffers
  inferTimer.start();
  this->processInput(*buffers.get(), images);
  inferTimer.stop();
  if (inferTimer.elapsed() >= 20000) {
    gLogWarning << "Timeout while pre-processing image" << '\n';
    throw CaffeRuntimeException("Timed out while doing inference");
  }

  cudaEventSynchronize(inputReady);
  buffers->copyInputToDeviceAsync(stream);

  bool status = context->enqueue(TBatchSize, buffers->getDeviceBindings().data(), stream, &inputReady);

  if (!status)
    return {};

  const auto streamSyncWithTimeout = [this] () {
    int counter         = 0;
    const int loopLimit = 60;
    
    while (counter++ < loopLimit) {
      const cudaError_t err = cudaStreamQuery(stream);
      switch (err) {
        case cudaSuccess: 
          return true;           // now we are synchronized
        case cudaErrorNotReady: 
          std::this_thread::sleep_for(500us);
          break;                  // continue waiting
        default: CHECK(err);      // unexpected error: throw
      }
    }

    gLogWarning << "Timeout while enqueueing inference job" << '\n';
    return false;
  };

  
  const auto enqueueResult = streamSyncWithTimeout();

  if (!enqueueResult)
    throw CaffeRuntimeException("Timed out while doing inference");

  buffers->copyOutputToHostAsync(stream);
  cudaStreamSynchronize(stream);

  // Post-process detections and verify results
  auto detections = getDetections(*buffers.get(), images, nmsThre, confThre);

  return detections;
}

//!
//! \brief Parse Python list of classnames
//!
//! \details This function parses a text file containing a a python tuple 
//!          of all objects to be detected
//!
//! \return Vector of objects that need to be detected
//!
template<int TBatchSize>
inline 
std::vector<std::string> FasterRCNN<TBatchSize>::parseClassesFile(void) {
  std::vector<std::string> classesList;
  classesList.push_back("background");

  std::string tmpStr;
  std::ifstream classesFile;

  classesFile.open(this->mParams.classesFileName);

  std::getline(classesFile, tmpStr);

  classesFile.close();

  const int tmpStrLen = tmpStr.length();

  if (tmpStrLen <= 0)
    throw CaffeRuntimeException("classes file is empty");

  std::stringstream strStream(tmpStr.substr(1, tmpStrLen-3)); //create string stream from the string

  while(strStream.good()) {
    std::string substr;
    getline(strStream, substr, ','); //get first string delimited by comma

    substr.erase( // remove white space + single quote
      std::remove_if(substr.begin(), substr.end(), 
        [](char c) -> bool
        { 
          return std::isspace<char>(c, std::locale::classic()) ||
            c == '\'';
        }), 
      substr.end());

    classesList.push_back(substr);
  }

  if (classesList.size() > 1) {
    return classesList;
  } else {
    throw CaffeRuntimeException("classes file is empty");
  }
}

}
#endif // FRCNN_TRT