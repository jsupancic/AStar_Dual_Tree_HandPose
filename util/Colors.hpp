/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_COLORS
#define DD_COLORS

#include <opencv2/opencv.hpp>

namespace deformable_depth
{
  // define various color constants
  namespace 
  {
    cv::Vec3b WHITE(255,255,255);
    cv::Vec3b BLACK(0,0,0);
    cv::Vec3b INVALID_COLOR(127,180,255);    
    cv::Vec3b INF_COLOR(127,0,0);
    cv::Vec3b NINF_COLOR(0,255,0);
    cv::Vec3b RED(0,0,255);
    cv::Vec3b GREEN(0,255,0);
    cv::Vec3b BLUE(255,0,0);
    cv::Vec3b YELLOW(0,255,255);
    cv::Vec3b ORANGE(0,165,255);
    cv::Vec3b DARK_ORANGE(0,140,255);
    cv::Vec3b CYAN(255,255,0);
  }

  cv::Scalar toScalar(cv::Vec3b);
  cv::Vec3b  toVec3b(cv::Scalar s);
  namespace Colors
  {
    cv::Scalar invalid();
    cv::Scalar inf();
    cv::Scalar ninf();
  }
}

#endif

