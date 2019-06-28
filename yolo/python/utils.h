#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <cassert>

using namespace cv;
using namespace std;

static inline Mat getLetterBoxBlob(const Mat& origImage, const int& inputH,
                         const int& inputW)
{
  int m_Height;
  int m_Width;
  int m_XOffset;
  int m_YOffset;
  float m_ScalingFactor;

  // letterboxed Image given to the network as input
  cv::Mat m_LetterboxImage;

  if (!origImage.data || origImage.cols <= 0 || origImage.rows <= 0)
  {
    cout << "Image is not valid" << endl;
    assert(0);
  }

  if (origImage.channels() != 3)
  {
    cout << "Non RGB images are not supported " << endl;
    assert(0);
  }

  m_Height = origImage.rows;
  m_Width = origImage.cols;

  // resize the DsImage with scale
  float dim = max(m_Height, m_Width);
  int resizeH = ((m_Height / dim) * inputH);
  int resizeW = ((m_Width / dim) * inputW);
  m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);

  // Additional checks for images with non even dims
  if ((inputW - resizeW) % 2) resizeW--;
  if ((inputH - resizeH) % 2) resizeH--;
  assert((inputW - resizeW) % 2 == 0);
  assert((inputH - resizeH) % 2 == 0);

  m_XOffset = (inputW - resizeW) / 2;
  m_YOffset = (inputH - resizeH) / 2;

  assert(2 * m_XOffset + resizeW == inputW);
  assert(2 * m_YOffset + resizeH == inputH);

  // resizing
  resize(origImage, m_LetterboxImage, Size(resizeW, resizeH), 0, 0, INTER_CUBIC);
  // letterboxing
  copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
                      m_XOffset, BORDER_CONSTANT, Scalar(128, 128, 128));
  // converting to RGB
  cvtColor(m_LetterboxImage, m_LetterboxImage, CV_BGR2RGB);

  return dnn::blobFromImage(m_LetterboxImage, 1.0, Size(inputW, inputH),
                                  Scalar(0.0, 0.0, 0.0), false, false);                             
}

#endif