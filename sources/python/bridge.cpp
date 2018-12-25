#include "bridge.h"
using namespace cv;
using namespace std;

// DsImage::DsImage(const cv::Mat& origImage, const std::string imageName, const int& inputH, const int& inputW) :
//     m_Height(0),
//     m_Width(0),
//     m_XOffset(0),
//     m_YOffset(0),
//     m_ScalingFactor(0.0),
//     m_RNG(cv::RNG(unsigned(std::time(0)))),
//     m_ImageName()
// {
//     m_ImageName = imageName;
//     // m_OrigImage = origImage.clone();
//     m_OrigImage = origImage;

//     if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
//     {
//         std::cout << "Image is not valid" << std::endl;
//         assert(0);
//     }

//     if (m_OrigImage.channels() != 3)
//     {
//         std::cout << "Non RGB images are not supported " << std::endl;
//         assert(0);
//     }

//     m_OrigImage.copyTo(m_MarkedImage);
//     m_Height = m_OrigImage.rows;
//     m_Width = m_OrigImage.cols;

//     // resize the DsImage with scale
//     float dim = std::max(m_Height, m_Width);
//     int resizeH = ((m_Height / dim) * inputH);
//     int resizeW = ((m_Width / dim) * inputW);
//     m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);

//     // Additional checks for images with non even dims
//     if ((inputW - resizeW) % 2) resizeW--;
//     if ((inputH - resizeH) % 2) resizeH--;
//     assert((inputW - resizeW) % 2 == 0);
//     assert((inputH - resizeH) % 2 == 0);

//     m_XOffset = (inputW - resizeW) / 2;
//     m_YOffset = (inputH - resizeH) / 2;

//     assert(2 * m_XOffset + resizeW == inputW);
//     assert(2 * m_YOffset + resizeH == inputH);

//     // resizing
//     cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_CUBIC);
//     // letterboxing
//     cv::copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
//                        m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
//     // converting to RGB
//     cv::cvtColor(m_LetterboxImage, m_LetterboxImage, CV_BGR2RGB);
// }

DsImageSingle::DsImageSingle(){}

DsImageSingle::DsImageSingle(const Mat& origImage, const string& imageName, const int& inputH, const int& inputW) {
    m_ImageName = imageName;
    // m_OrigImage = origImage.clone();
    m_OrigImage = origImage;

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
    {
        cout << "Image is not valid" << endl;
        assert(0);
    }

    if (m_OrigImage.channels() != 3)
    {
        cout << "Non RGB images are not supported " << endl;
        assert(0);
    }

    m_OrigImage.copyTo(m_MarkedImage);
    m_Height = m_OrigImage.rows;
    m_Width = m_OrigImage.cols;

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
    resize(m_OrigImage, m_LetterboxImage, Size(resizeW, resizeH), 0, 0, INTER_CUBIC);
    // letterboxing
    copyMakeBorder(m_LetterboxImage, m_LetterboxImage, m_YOffset, m_YOffset, m_XOffset,
                       m_XOffset, BORDER_CONSTANT, Scalar(128, 128, 128));
    // converting to RGB
    cvtColor(m_LetterboxImage, m_LetterboxImage, CV_BGR2RGB);
}

Mat blobFromDsImage(const DsImageSingle& inputImage, const int& inputH,
                         const int& inputW)
{
    return dnn::blobFromImage(inputImage.getLetterBoxedImage(), 1.0, Size(inputW, inputH),
                                   Scalar(0.0, 0.0, 0.0), false, false);
}