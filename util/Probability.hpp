/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_PROBABILITY
#define DD_PROBABILITY

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <functional>

namespace deformable_depth
{
  using std::vector;
  
  // multinomial
  cv::Point rnd_multinom(const cv::Mat&prob_image);
  int rnd_multinom(vector<double>&thetas);
  
  // MV Gaussian
  double mv_gaussian_pdf(const cv::Mat&x,const cv::Mat&mu, const cv::Mat&sigma,const cv::Mat&sigmaInv);
  double mv_gaussian_support(double sigma);
  cv::Mat mv_gaussian_kernel(const cv::Mat&sigma, const cv::Mat&sigmaInv);
  
  class EmpiricalDistribution
  {
  public:
    vector<double> data;
    // 
    double cdf(double x) const; 
    double quant(double p);
  };
  
  // 1D Gaussian
  class Gaussian
  {
  protected:
    double mu;
    double sigma;
    
  public:
    Gaussian(double mu, double sigma);
    Gaussian(const Gaussian&);
    double pdf(double x) const;
    
    friend std::ostream& operator << (std::ostream&os, const Gaussian&g);
  };

  // entropy functions
  double shannon_entropy(double p);
  double shannon_entropy(vector<double> ps);
  typedef std::function<cv::Mat (int)> ExampleFn;
  double entropy_gaussian(ExampleFn exampleFn,int nExamples,
			  cv::Mat&mean,cv::Mat&cov,int feat_len = 2);
}

#endif

