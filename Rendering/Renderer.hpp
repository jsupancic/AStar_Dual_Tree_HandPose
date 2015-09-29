/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_RENDERER
#define DD_RENDERER

#define use_speed_ 0

#include <cv.h>
#include <highgui.h>
#include "HandSynth.hpp"

namespace deformable_depth
{
  using cv::VideoCapture;
  using cv::Mat_;
  using cv::Mat;
  using cv::MatND;
  
  class Renderer
  {
  private:
    HandRenderer hand;
  public:
    Renderer();
    HandData doIteration(Mat&Zrnd,Mat&RGBrnd);
    HandRenderer& getHand();
  protected:
    void draw();
    void doOpenGL(Mat&Zrnd,Mat&RGBrnd);
  };
 
  // GL functions to read the rendered data back to OpenCV.
  void gl_get_camera_parameter_from_perspective_matrix(
    double & fovy_rad,
    double & fovx_rad,
    double & clip_min,
    double & clip_max);
  void gl_cv_read_buffers(Mat&Zrnd,Mat&RGBrnd,int W,int H);
  void gl_cv_read_buffers_RGB(Mat&RGBrnd,int W,int H);
  void gl_cv_read_buffers_Z(Mat&Zrnd,int W,int H);
  // GLUT utility functions
  void reshape(int width, int height);
  void init_glut(int argc, char**argv, void (* fn_render)( void ));
}

#endif
