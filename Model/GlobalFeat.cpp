/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "GlobalFeat.hpp"
#include <boost/multi_array.hpp>

namespace deformable_depth
{
  double colorConst(const ImRGBZ& im, Rect& bb)
  {
    // round it int
    Rect query = bb;
    
    // init counts to zero
    int hist_res = 3;
    boost::multi_array<double,3 > histogram(boost::extents[hist_res][hist_res][hist_res]);
    for(int rIter = 0; rIter < hist_res; rIter++)
      for(int gIter = 0; gIter < hist_res; gIter++)
	for(int bIter = 0; bIter < hist_res; bIter++)
	  histogram[rIter][gIter][bIter] = 0;
    
    // bin the colors
    int bin_size = 256/hist_res;
    double num_data = 0;
    for(int xIter = query.tl().x; xIter < query.br().x; xIter++)
      for(int yIter = query.tl().y; yIter < query.br().y; yIter++)
      {
	Vec3b pixel = im.RGB.at<Vec3b>(yIter,xIter);
	int rIdx = clamp(0, pixel[0] / bin_size,hist_res - 1);
	int gIdx = clamp(0, pixel[1] / bin_size,hist_res - 1);
	int bIdx = clamp(0, pixel[2] / bin_size,hist_res - 1);
	
	histogram[rIdx][gIdx][bIdx]++;
	num_data++;
      }
      
    // compute the entropy
    double entropy = 0;
    double h_max = 0;
    for(int rIter = 0; rIter < hist_res; rIter++)
      for(int gIter = 0; gIter < hist_res; gIter++)
	for(int bIter = 0; bIter < hist_res; bIter++)
	{
	  double p_max = 1.0/(hist_res*hist_res*hist_res);
	  h_max += p_max * std::log2(1/p_max);
	  
	  double p_i = histogram[rIter][gIter][bIter]/num_data;
	  if(p_i > 0)
	    entropy += p_i * std::log2(1/p_i);
	}
	
    return entropy/h_max;
  }
  
  //
  // SECTION: SubTreeConsistancyFeature
  //
  vector<double> SubTreeConsistancyFeature::calculate_color_consistancy(const ImRGBZ& im, const Rect_< double >& bb)
  {
    Rect query = rectResize(bb,.5,.5);
    double entropy = colorConst(im,query);
    
    vector<double> feat(color_bins,0);
    int bin = clamp<int>(0,  (int)entropy*color_bins ,color_bins-1);
    feat[bin] = BETA_COLOR;
    return feat;
  }
  
  double subtree_box_consistancy(const ImRGBZ& im, const Rect_< double >& query)
  {
    // if the OL is NaN we must compute it
    PixelAreaDecorator&pixAreas = pixelAreas_cached(im);
        
    double best_ol = -inf;
    for(int xIter = query.tl().x; xIter < query.br().x; ++xIter)
      for(int yIter = query.tl().y; yIter < query.br().y; ++yIter)
      {
	if(xIter < 0 || yIter < 0 || xIter >= im.cols() || yIter >= im.rows())
	  continue;
	
	for(Direction cur_dir : card_dirs())
	{
	  // check the query dimensions
	  int pa_grid_height = pixAreas.bbs.size();
	  bool pay_good = 0 <= yIter && yIter < pa_grid_height;
	  if(!pay_good)
	  {
	    cout << "pay bad: " << yIter << " " << pa_grid_height << endl;
	    assert(false);
	  }
	  int pa_grid_width  = pixAreas.bbs[yIter].size();
	  bool pax_good = 0 <= xIter && xIter < pa_grid_width;	  
	  if(!pax_good)
	  {
	    cout << "pax bad: " << xIter << " " << pa_grid_width << endl;
	    assert(false);
	  }
	  
	  Rect&candidate = pixAreas.bbs[yIter][xIter][cur_dir];
	  double ol = rectIntersect(candidate,query);
	  if(ol > best_ol)
	  {
	    best_ol = ol;
	  }
	}
      }    
      
    assert(goodNumber(best_ol));
    return best_ol;
  }

  vector<double> SubTreeConsistancyFeature::calculate_subtree_box_consistancy(const ImRGBZ& im, const Rect_< double >& bb)
  {
    Rect query = bb;
    
    // truncate to int
    double best_ol = subtree_box_consistancy(im,bb);

    vector<double> feat_vec(bb_bins,0);
    int bin = clamp<int>(0, (int)best_ol*bb_bins ,bb_bins-1);
    assert(0 <= bin && bin < feat_vec.size());
    feat_vec[bin] = BESTS_BOX_CONSIST;
    return feat_vec;
  }
  
  SparseVector SubTreeConsistancyFeature::calculate(
    const ImRGBZ& im, const Rect_<double>&bb)
  {
    return vector<double>{};
    
//     vector<double> box_feat = calculate_subtree_box_consistancy(im,bb);
//     vector<double> color_feat = calculate_color_consistancy(im,bb);
//     
//     box_feat.insert(box_feat.end(),color_feat.begin(),color_feat.end());
//     return box_feat;
  }
  
  ///
  /// SECTION: Deva's Metric Feature
  ///
  SparseVector MetricFeature::calculate(
    const ImRGBZ& im, const Rect_< double >& bb, int nx, int ny)
  {
    // TODO
    return SparseVector(0);
  }
  
  Mat MetricFeature::show(const SparseVector& feat, int nx, int ny)
  {
    // TODO
    return Mat();
  }
}

