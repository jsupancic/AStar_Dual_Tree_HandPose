/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_MANIFOLD_SELECTION
#define DD_MANIFOLD_SELECTION

#include <vector>
#include <util_mat.hpp>

namespace deformable_depth
{
  typedef function<vector<float> (const ImRGBZ& im, const Rect_< double >& bb)> ManifoldFn;
  
  vector<float> manifoldFn_boxMedian(const ImRGBZ&im, const Rect_<double> & bb);
  vector<float> manifoldFn_all(const ImRGBZ& im, const Rect_< double >& bb);
  vector<float> manifoldFn_kmax(const ImRGBZ& im, const Rect_< double >& bb);
  vector<float> manifoldFn_telescope(const ImRGBZ& im, const Rect_< double >& bb);
  vector<float> manifoldFn_default(const ImRGBZ&im,const Rect_<double>&bb);
  vector<float> manifoldFn_prng(const ImRGBZ&im,const Rect_<double>&bb, int samples = 10);
  vector<float> manifoldFn_min(const ImRGBZ&im,const Rect_<double>&bb);
  vector<float> manifoldFn_apxMin(const ImRGBZ&im,const Rect_<double>&bb);
  vector<float> manifoldFn_ordApx(const ImRGBZ&im,const Rect_<double>&bb,double ord);
  vector<float> manifoldFn_discrete_sparse(const ImRGBZ&im, const Rect_<double>&bb,float step = 10);
  
  vector<float> sort_depths_by_im_area(const vector<float>&depths,const ImRGBZ&im,const Rect_<double>&bb);
}

#endif
