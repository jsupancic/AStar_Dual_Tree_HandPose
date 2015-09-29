/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_KD_TRESS
#define DD_KD_TRESS

#include <memory>
#include <opencv2/opencv.hpp>

#include "Detector.hpp"

namespace deformable_depth
{
  typedef cv::flann::Index KDTree;
  
  shared_ptr<KDTree> kdTreeXYZS(DetectionSet&index_detections,ImRGBZ&im);
  shared_ptr<KDTree> kdTreeXYWH(DetectionSet&detections);
}

#endif
