/**
 * Copyright 2012: Jürgen Brauer
 **/

#ifndef VIS_HOG
#define VIS_HOG

#define use_speed_ 0

#include <cv.h>
#include <iostream>

using cv::Mat;
using std::vector;

Mat get_hogdescriptor_visu(Mat& origImg, vector<float>& descriptorValues);

#endif
