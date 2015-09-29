/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_COLOR_MODEL
#define DD_COLOR_MODEL

#include <opencv2/opencv.hpp>
#include <vector>

namespace deformable_depth
{
  using cv::Mat;
  using std::vector;
  
  class ColorModel_Gaussian
  {
  protected:
    double fg_samples, bg_samples;
    Mat fg_mu, fg_sigma;
    Mat bg_mu, bg_sigma;
    
  public:
    ColorModel_Gaussian();
    void accumulate(const Mat&image,const Mat&mask);
    void normalize();
    Mat likelihood_ratio(const Mat&image);    
  };
  
  class ColorModel_Histogram
  {
  public:
    Mat fg_hist, bg_hist;
    vector<int> channels;
    vector<float> colRange;
    vector<float*> ranges;
    int histSz_1;
    vector<int> histSz;
    
  public:
    ColorModel_Histogram();
    void accumulate(const Mat&image,const Mat&mask);
    void normalize();
    Mat likelihood_ratio(const Mat&image);
    
  protected:
    int* get_channels();
    float * get_colRange();
    const float** get_ranges();
    int * get_histSz();
  };
}

#endif
