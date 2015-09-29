/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_SEGMENT
#define DD_SEGMENT

#define use_speed_ 0
#include <cv.h>

#include "util.hpp"

namespace deformable_depth
{
  Mat ZDT(cv::Mat Zcap);
  void segment(int argc, char**argv);  
}

#endif

