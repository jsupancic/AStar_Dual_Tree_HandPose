/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_GLOBAL_FEATS
#define DD_GLOBAL_FEATS

#include "DepthFeatures.hpp"
#include "Faces.hpp"
#include "SubTrees.hpp"

namespace deformable_depth
{
  double subtree_box_consistancy(const ImRGBZ&im,const Rect_<double>&bb);
  
  class SubTreeConsistancyFeature
  {
  protected:
    static constexpr size_t bb_bins = 5;
    static constexpr size_t color_bins = 8;
    static constexpr double BETA_COLOR = 10;
    static constexpr double BESTS_BOX_CONSIST = 10;
    
  public:
    static constexpr size_t length = 0; //color_bins + bb_bins;
    static SparseVector calculate(const ImRGBZ&im,const Rect_<double>&bb);
    
  protected:
    static vector<double> calculate_color_consistancy(const ImRGBZ&im,const Rect_<double>&bb);
    static vector<double> calculate_subtree_box_consistancy(const ImRGBZ&im,const Rect_<double>&bb);
  };
  
  class MetricFeature
  {
  public:
    static constexpr size_t length = 0; //color_bins + bb_bins;
    static SparseVector calculate(const ImRGBZ&im,const Rect_<double>&bb,int nx, int ny);   
    static Mat show(const SparseVector&feat,int nx, int ny);
  };
  
  typedef MetricFeature GlobalFeature;
  
  double colorConst(const ImRGBZ&im, Rect&bb);
}

#endif
