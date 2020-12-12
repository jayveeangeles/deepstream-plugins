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

#ifndef SSD_TRT
#define SSD_TRT

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "caffetrt.h"
#include "caffehelper.h"

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
#include <string>
#include <algorithm>

//! \brief  The SSD class implements the SSD sample
//!
//! \details It creates the network using a caffe model
//!
namespace trt 
{
//!
//! \brief The SSDParams structure groups the additional parameters required by
//!         the SSD sample.
//!
struct SSDParams : public CaffeNetworkParams
{
  float confThre;
  int keepTopK = 200;
};

template<int TBatchSize>
class SSD : public Caffe <TBatchSize>
{
  template <typename T>
  using SampleUniquePtr = typename Caffe<TBatchSize>::template SampleUniquePtr<T>;
public:
  SSD(SSDParams params) {
    initClass(std::forward<SSDParams>(params));
  }

  virtual ~SSD();

  virtual bool initEngine();

  virtual std::array<std::vector<DetectionObject>, TBatchSize> infer(
    const std::array<cv::Mat, TBatchSize>& images) override { 
      
    return infer(images, mConfThre); 
  }

  std::array<std::vector<DetectionObject>, TBatchSize> infer(
    const std::array<cv::Mat, TBatchSize>& images, const float confThre);

private:
  int mKeepTopK;
  float mConfThre;

  cudaStream_t stream;
  cudaEvent_t inputReady;

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
  //! \brief Init Class, set params
  //!
  void initClass(SSDParams params);

  //!
  //! \brief Parse Python list of classnames
  //!
  std::vector<std::string> parseClassesFile(void);

  //!
  //! \brief Filters output detections, handles post-processing of bounding boxes and verify results
  //!
  std::array<std::vector<DetectionObject>, TBatchSize> getDetections(
    const samplesCommon::BufferManager& buffers, 
    const std::array<cv::Mat, TBatchSize>& images, const float scoreThreshold);
};

//!
//! \brief Destructor
//!
template<int TBatchSize>
SSD<TBatchSize>::~SSD() {
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
bool SSD<TBatchSize>::initEngine() {
  auto successInit = Caffe<TBatchSize>::initEngine();

  // https://forums.developer.nvidia.com/t/tensorrt-7-1-0-dp-segfault-when-deserailizing-the-priorbox-plugin/124111/8
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
void SSD<TBatchSize>::initClass(SSDParams params) {
  this->mKeepTopK  = params.keepTopK;
  this->mConfThre  = params.confThre;

  this->mParams  = std::move(params);
  this->mClasses = this->parseClassesFile();

  this->mParams.outputClsSize = this->mClasses.size();
  
  this->mNbInputs     = 1;
  this->mNetImageMean = cv::Scalar(104.0f, 117.0f, 123.0f); 
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
std::array<std::vector<DetectionObject>, TBatchSize> SSD<TBatchSize>::getDetections(
  const samplesCommon::BufferManager& buffers, 
  const std::array<cv::Mat, TBatchSize>& images, const float scoreThreshold) {

  const int batchSize      = TBatchSize;
  const int keepTopK       = this->mKeepTopK;
  const int outputClsSize  = this->mParams.outputClsSize;

  const float* detectionOut = static_cast<const float*>(buffers.getHostBuffer("detection_out"));
  const int* keepCount      = static_cast<const int*>(buffers.getHostBuffer("keep_count"));

  // The sample passes if there is at least one detection for each item in the batch
  std::array<std::vector<DetectionObject>, TBatchSize> results;

  for (int p = 0; p < batchSize; ++p) {
    for (int i = 0; i < keepCount[p]; ++i) {
      const float* det = detectionOut + (p * keepTopK + i) * 7;
      if (det[2] < scoreThreshold)
        continue;

      assert((int) det[1] < outputClsSize);

      const samplesCommon::BBox b{
        det[3] * float(images[p].cols), 
        det[4] * float(images[p].rows),
        det[5] * float(images[p].cols), 
        det[6] * float(images[p].rows)
      };
      
      results[p].push_back( {b, this->mClasses[(int) det[1]], det[2] } );
    }
  }
  return results;
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
std::array<std::vector<DetectionObject>, TBatchSize> SSD<TBatchSize>::infer(
    const std::array<cv::Mat, TBatchSize>& images, const float confThre) {

  using namespace std::chrono_literals;
  // Create RAII buffer manager object, move away from RAII

  assert(this->mParams.inputTensorNames.size() == this->mNbInputs);

  // Read the input data into the managed buffers
  inferTimer.start();
  this->processInput(*buffers.get(), images);
  inferTimer.stop();
  if (inferTimer.elapsed() >= this->mParams.preprocessDeadline) {
    gLogWarning << "Timeout while pre-processing image" << '\n';
    throw CaffeRuntimeException("Timed out while doing inference");
  }

  cudaEventSynchronize(inputReady);
  buffers->copyInputToDeviceAsync(stream);

  bool status = context->enqueue(TBatchSize, buffers->getDeviceBindings().data(), stream, &inputReady);
  if (!status)
    return {};

  const auto streamSyncWithTimeout = [this] () {
    uint counter         = 0;
    
    while (counter++ < this->mParams.inferLoopLimit) {
      const cudaError_t err = cudaStreamQuery(stream);
      switch (err)
      {
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

  auto detections = getDetections(*buffers.get(), images, confThre);

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
std::vector<std::string> SSD<TBatchSize>::parseClassesFile(void) {
  std::vector<std::string> classesList(1);
  std::vector<int> order;

  std::string tmpStr;
  std::ifstream classesFile;

  classesFile.open(this->mParams.classesFileName);

  int count = 0;
  uint index = 0;

  while (std::getline(classesFile, tmpStr)) {
    std::string key, value;
    std::istringstream iss(tmpStr);

    if (count == 2) {
      iss >> key >> value;
      index = atoi(value.c_str());

      if ((index + 1) > classesList.size()) {
        classesList.resize(index + 1);        
      }
    } else if (count == 3) {
      iss >> key >> value;
          
      value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
      classesList[index] = value;
      index = 0;
    }
    count = (count + 1) % 5;
  }
  
  classesFile.close();

  if (classesList.size() > 1) {
    return classesList;
  } else {
    throw CaffeRuntimeException("classes file is empty");
  }
}

}
#endif // SSD_TRT