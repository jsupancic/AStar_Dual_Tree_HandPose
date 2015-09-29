/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_CAPTURER
#define DD_CAPTURER

#define use_speed_ 0

#include <util.hpp>
#include <cv.h>
#include <highgui.h>

namespace deformable_depth
{
  using namespace cv;
  using cv::VideoCapture;
  using cv::Mat;
  using namespace std;
  
  class Capturer
  {
  private:
    VideoCapture capture;
    MatND skin_hist, bg_hist;    
  public:
    Capturer();
    void doIteration(Mat&Zcap,Mat&RGBcap);
  protected:  
  };
}

#endif
