#include "caffehelper.h"

//!
//! \brief Hack to Parse Prototxt file
//!
string buildEngineName(std::string modelPath, cudaDeviceProp& prop) {
  std::string line, height, width;
  std::smatch m;
  std::regex e("^\\W+dim:\\W([0-9]{3,4})$");

  std::ifstream caffePrototxtFile(modelPath + "/test_trt.prototxt");
  bool firstDim = true, continueReading = true;

  if (caffePrototxtFile.is_open()){
  while (std::getline(caffePrototxtFile, line) && continueReading) {
    while (std::regex_search(line, m, e)) {
    if (firstDim) {
      height = m[1];
      firstDim = false;
      break;
    } else {
      width = m[1];
      continueReading = false;
      break;
    }
    }
  }
  caffePrototxtFile.close();
  }

  assert(height != ""); assert(width != "");

  return modelPath + "/model-" + std::to_string(prop.major) + \
  std::to_string(prop.minor) + "-" + height + "x" + width + ".engine";
}

//!
//! \brief Initializes parameters for Faster RCNN engine
//!
void initializeParams(
  trt::FasterRCNNParams& params, cudaDeviceProp& prop, std::string& modelPath, std::string& modelName) {

  params.dataDirs.push_back(modelPath.c_str());
  params.serializedWeightsFileName = buildEngineName(modelPath, prop);
  params.prototxtFileName = "test_trt.prototxt";
  params.weightsFileName  = modelName;
  params.classesFileName  = locateFile("classname.txt", params.dataDirs);
  params.batchSize        = BATCH_SIZE;
  
  params.inputTensorNames.push_back("data");
  params.inputTensorNames.push_back("im_info");
  params.outputTensorNames.push_back("bbox_pred");
  params.outputTensorNames.push_back("cls_prob");
  params.outputTensorNames.push_back("rois");
}

//!
//! \brief Initializes parameters for SSD engine
//!
void initializeParams(
  trt::SSDParams& params, cudaDeviceProp& prop, std::string& modelPath, std::string& modelName) {
  params.dataDirs.push_back(modelPath.c_str());
  params.serializedWeightsFileName = buildEngineName(modelPath, prop);

  params.prototxtFileName = "deploy_trt.prototxt";
  params.weightsFileName  = modelName;
  params.classesFileName  = locateFile("labelmap.prototxt", params.dataDirs);
  params.batchSize        = BATCH_SIZE;

  params.inputTensorNames.push_back("data");
  params.outputTensorNames.push_back("detection_out");
  params.outputTensorNames.push_back("keep_count");
}

//!
//! \brief Get GCD for two FPS
//!
int getGCD(int n1, int n2) {
  int hcf = 1;

  for (int i = 1; i <=  n2; ++i) {
    if (n1 % i == 0 && n2 % i ==0) {
      hcf = i;
    }
  }

  return hcf;
}