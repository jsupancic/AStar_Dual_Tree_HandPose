/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_SKIN
#define DD_SKIN

#include <opencv2/opencv.hpp>
#include "DepthFeatures.hpp"

namespace deformable_depth
{
  using cv::Mat;
  
  void skin_make_hist();
  double skin_likelihood(Vec3b color);
  Mat  skin_detect(const Mat&RGB, Mat&FGProbs, Mat&BGProbs);
  
#ifdef DD_CXX11
  class SkinFeatureComputer : public DepthFeatComputer
  {
  protected:
    Size cell_size;
    Size win_size;
    static constexpr int BG_BINS = 4;
    static constexpr int FG_BINS = 4;
    static constexpr double weight = 1;
    
  public: // RAII CTORS
    SkinFeatureComputer(
		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8));
    SkinFeatureComputer(SkinFeatureComputer&other);
    
  public: // Implmeentation of HOGComputer
    virtual int getNBins() ;
    virtual Size getBlockStride() ;
    virtual Size getCellSize() ;
    virtual Size getWinSize() ;
    virtual void compute(ImRGBZ&im,vector<float>&feats) ;
    virtual size_t getDescriptorSize();   
    virtual Mat show(const string&title,vector<double> feat);
    virtual int cellsPerBlock();
    virtual Size getBlockSize();
    virtual string toString() const;
    virtual vector<FeatVis> show_planes(vector<double> feat);
  };
#endif
}

#endif

