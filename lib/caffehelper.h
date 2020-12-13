#ifndef CAFFE_HELPER_H
#define CAFFE_HELPER_H

#include <sstream>
#include <tuple>
#include <iostream>
#include <fstream>
#include <regex>

#include "frcnntrt.h"
#include "ssdtrt.h"
#include "logger.h"

#ifndef BATCH_SIZE
#define BATCH_SIZE  1
#endif

//!
//! \brief Hack to Parse Prototxt file
//!
std::string buildEngineName(std::string, cudaDeviceProp&);

//!
//! \brief Initializes members of the params using env variables
//!
void initializeParams(trt::FasterRCNNParams&, cudaDeviceProp&, std::string&, std::string&);

//!
//! \brief Initializes parameters for SSD engine
//!
void initializeParams(trt::SSDParams&, cudaDeviceProp&, std::string&, std::string&);

//!
//! \brief Get GCD for two FPS
//!
int getGCD(int, int);

#endif // CAFFE_HELPER_H