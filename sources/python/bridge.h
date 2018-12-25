#ifndef __BRIDGE_H__
#define __BRIDGE_H__

// #ifdef __cplusplus
// extern "C" {
// #endif
// Include darknet as a C Library
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov3.h"
#include "ds_image.h"
#include "calibrator.h"
#include "plugin_factory.h"

using namespace cv;
using namespace std;

class DsImageSingle : public DsImage
{
public:
    DsImageSingle();
    DsImageSingle(const Mat&, const string&, const int&, const int&);

// protected:
//     int m_Height;
//     int m_Width;
//     int m_XOffset;
//     int m_YOffset;
//     float m_ScalingFactor;
//     string m_ImagePath;
//     RNG m_RNG;
//     string m_ImageName;
//     vector<BBoxInfo> m_Bboxes;

//     // unaltered original Image
//     Mat m_OrigImage;
//     // letterboxed Image given to the network as input
//     Mat m_LetterboxImage;
//     // final image marked with the bounding boxes
//     Mat m_MarkedImage;
};

Mat blobFromDsImage(const DsImageSingle& inputImage, const int& inputH,
                         const int& inputW);


// #ifdef __cplusplus
// }
// #endif

#endif