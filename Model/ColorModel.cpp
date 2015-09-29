/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ColorModel.hpp"
#include "Probability.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  ///
  /// SECTION: Gaussian Color Model
  ///
  void ColorModel_Gaussian::accumulate(const Mat& image, const Mat& mask)
  {
    for(int rIter = 0; rIter < image.rows; rIter++)
      for(int cIter = 0; cIter < image.cols; cIter++)
      {
	Vec3b pixel = image.at<Vec3b>(rIter,cIter);
	Mat pix_color(1,3,DataType<double>::type);
	pix_color.at<double>(0) = pixel[0];
	pix_color.at<double>(1) = pixel[1];
	pix_color.at<double>(2) = pixel[2];
	
	uint8_t m = (mask.at<uint8_t>(rIter,cIter));
	if(m)
	{
	  fg_samples++;
	  fg_mu += pix_color;
	  fg_sigma += pix_color.t() * pix_color;
	}
	else
	{
	  bg_samples++;
	  bg_mu += pix_color;
	  bg_sigma += pix_color.t() * pix_color;
	}
      }
  }

  ColorModel_Gaussian::ColorModel_Gaussian()
  {
    bg_samples = fg_samples = 0;
    bg_mu = fg_mu = Mat(1,3,DataType<double>::type,Scalar::all(0));
    bg_sigma = fg_sigma = Mat(3,3,DataType<double>::type,Scalar::all(0));
  }

  void ColorModel_Gaussian::normalize()
  {
    fg_mu /= fg_samples;
    fg_sigma /= fg_samples;
    
    bg_mu /= bg_samples;
    bg_sigma /= bg_samples;
  }

  Mat ColorModel_Gaussian::likelihood_ratio(const Mat& image)
  {
    Mat pFG(image.rows,image.cols,DataType<double>::type,Scalar::all(0));
    Mat pBG(image.rows,image.cols,DataType<double>::type,Scalar::all(0));
    Mat fg_sigma_inv = fg_sigma.inv();
    Mat bg_sigma_inv = bg_sigma.inv();
    
    for(int rIter = 0; rIter < image.rows; rIter++)
      for(int cIter = 0; cIter < image.cols; cIter++)
      {
	Vec3b pixel = image.at<Vec3b>(rIter,cIter);
	Mat pix_color(1,3,DataType<double>::type);
	pix_color.at<double>(0) = pixel[0];
	pix_color.at<double>(1) = pixel[1];
	pix_color.at<double>(2) = pixel[2];
	
	pFG.at<double>(rIter,cIter) = mv_gaussian_pdf(pix_color,fg_mu, fg_sigma,fg_sigma_inv);
	pFG.at<double>(rIter,cIter) = mv_gaussian_pdf(pix_color,bg_mu, bg_sigma,bg_sigma_inv);
      }
    
    pFG += 1e-6;
    pBG += 1e-6;
    
    // Pr(skin | color) = Pr(color | skin) * Pr(Skin)/Pr(Color)
    // compute the likelihood ratio assumming equal likelihood
    return pFG/pBG;      
  }
  
  ///
  /// SECTION: ColorHistogram
  /// 
  
  ColorModel_Histogram::ColorModel_Histogram() : 
    channels{0,1,2},
    colRange{0,255},
    ranges{&colRange[0], &colRange[0], &colRange[0]},
    histSz_1(16),
    histSz{histSz_1,histSz_1,histSz_1}
  {
  }
  
  void ColorModel_Histogram::accumulate(const Mat& image, const Mat& mask_in)
  {
    Mat mask, inv_mask = mask_in;
    threshold(inv_mask,mask,254,255,cv::THRESH_BINARY_INV);
    threshold(mask,inv_mask,254,255,cv::THRESH_BINARY_INV);
    
    // add to the histogram...
    calcHist(&image,1,get_channels(),mask,fg_hist,3,get_histSz(),get_ranges(),true,true);
    calcHist(&image,1,get_channels(),inv_mask,bg_hist,3,get_histSz(),get_ranges(),true,true);    
  }

  Mat ColorModel_Histogram::likelihood_ratio(const Mat& image)
  {
    Mat fRGB; image.convertTo(fRGB,DataType<Vec3f>::type);
    Mat pFG; calcBackProject(&fRGB,1,get_channels(),fg_hist,pFG,get_ranges());
    Mat pBG; calcBackProject(&fRGB,1,get_channels(),bg_hist,pBG,get_ranges());
    pFG.convertTo(pFG,DataType<double>::type);
    pBG.convertTo(pBG,DataType<double>::type);
    pFG += 1e-6;
    pBG += 1e-6;
    
    // Pr(skin | color) = Pr(color | skin) * Pr(Skin)/Pr(Color)
    // compute the likelihood ratio assumming equal likelihood
    return pFG/pBG;  
  }

  void ColorModel_Histogram::normalize()
  {
    cv::normalize(fg_hist,fg_hist,1,0,cv::NORM_L1);
    cv::normalize(bg_hist,bg_hist,1,0,cv::NORM_L1);    
  }
  
  int* ColorModel_Histogram::get_channels()
  {
    return &channels[0];
  }

  float* ColorModel_Histogram::get_colRange()
  {
    return &colRange[0];
  }

  int* ColorModel_Histogram::get_histSz()
  {
    return &histSz[0];
  }

  const float** ColorModel_Histogram::get_ranges()
  {
    return const_cast<const float**>(&ranges[0]);
  }
}
