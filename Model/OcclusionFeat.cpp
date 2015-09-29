/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "OcclusionFeat.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
 
namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  ///
  /// SECTION: CellDepths
  ///
  // protected CTOR
  CellDepths::CellDepths()
  {
  }
  
  CellDepths::CellDepths(const ImRGBZ& im, int nx, int ny)
  { 
    upper_manifold = Mat(ny,nx,DataType<float>::type,Scalar::all(0));
    double block_width = im.Z.cols/static_cast<double>(nx);
    double block_height = im.Z.rows/static_cast<double>(ny);
    assert(im.Z.cols % nx == 0);
    assert(im.Z.rows % ny == 0);
    double block_area = block_width*block_height;
    for(int block_y = 0; block_y < upper_manifold.rows; block_y++)
    {
      raw_cell_depths.push_back(vector<vector<float> >());
      
      for(int block_x = 0; block_x < upper_manifold.cols; block_x++)
      {
	vector<float> depths(block_width*block_height);
	int idx = 0;
	for(int rIter = block_y*block_height; rIter < (block_y+1)*block_height; rIter++)
	  for(int cIter = block_x*block_width; cIter < (block_x+1)*block_width; cIter++)
	  {
	    assert(block_y < upper_manifold.rows);
	    assert(block_x < upper_manifold.cols);
	    assert(rIter < im.Z.rows);
	    assert(cIter < im.Z.cols);
	    depths[idx] = im.Z.at<float>(rIter,cIter);
	    idx++;
	  }
	std::sort(depths.begin(),depths.end());
	// median
	// upper_manifold.at<float>(block_y,block_x) = depths[depths.size()/2];
	// min
	upper_manifold.at<float>(block_y,block_x) = depths[0];
	
	// can't compute the depth histogram yet
	raw_cell_depths[block_y].push_back(depths);
      }
    }
  }
  
  CellDepths CellDepths::operator()(const Rect& roi) const
  {
    CellDepths result;
    
    // copy the upper_manifold roi
    result.upper_manifold = upper_manifold(roi).clone();
    
    // copy the raw depth roi
    for(int yIter = roi.y; yIter < roi.y + roi.height; yIter++)
    {
      result.raw_cell_depths.push_back(vector<vector<float>>());
      for(int xIter = roi.x; xIter < roi.x + roi.width; xIter++)
      {
	assert(0 <= yIter && yIter < raw_cell_depths.size());
	assert(0 <= xIter && xIter < raw_cell_depths[0].size());
	result.raw_cell_depths.back().push_back(raw_cell_depths[yIter][xIter]);
      }
      assert(result.raw_cell_depths.back().size() == result.upper_manifold.cols);
    }
    assert(result.raw_cell_depths.size() == result.upper_manifold.rows);
    
    return result;
  }
  
  vector< double > CellDepths::depth_histogram(
    int row, int col, float min_depth, float max_depth) const
  {
    vector<double> histogram(METRIC_DEPTH_BINS,0);
    
    require_gt(max_depth,min_depth);
    assert(row < raw_cell_depths.size());
    assert(col < raw_cell_depths[0].size());
    
    // compute the histogram from the data
    for(const float & depth : raw_cell_depths[row][col])
    {
      float metric_depth = depth - min_depth;
      
      if(depth < min_depth)
	histogram[BIN_OCC]++;      // Occlusion
      else if(depth > max_depth || depth >= params::MAX_Z())
	histogram[BIN_BG]++;      // BG
      else
      {
	double bin_size = (max_depth-min_depth)/(METRIC_DEPTH_BINS-2);
	int inner_bin = clamp<int>(0,metric_depth/bin_size,4);
	histogram[2+inner_bin]++;
      }
    }
    
    // now, normalize the histogram
    double sum = 0;
    for(double & p : histogram)
      sum += p;
    for(int iter = 0; iter < histogram.size(); ++iter)
    {
      if(iter == BIN_OCC)
	histogram[iter] = histogram[iter]*BETA_OCC/sum;
      else
	histogram[iter] = histogram[iter]*BETA_HISTOGRAM/sum;
    }
    
    return histogram;
  }
  
  ///
  /// SECTION: OccSliceFeature
  ///
  CellDepths OccSliceFeature::mkCellDepths(const ImRGBZ& im) const
  {
    return CellDepths(im,subordinate->blocks_x(),subordinate->blocks_y());
  }
  
  int OccSliceFeature::cellsPerBlock()
  {
    return subordinate->cellsPerBlock();
  }
    
  std::vector< float > OccSliceFeature::decorate_feat(
    const std::vector< float >& raw_pos,
    const CellDepths&cellDepths,
    float min_depth, 
    float max_depth)
  {
    int nbins = subordinate->getNBins();
    int ncells = num_cells();    
    
    std::vector< float > feats = vector<float>(ncells*(nbins+DECORATION_LENGTH));
    int occIter = 0, rawIter = 0; // the order of these for loops supports these iterators
    for(int xIter = 0; xIter < subordinate->blocks_x(); xIter++)
      for(int yIter = 0; yIter < subordinate->blocks_y(); yIter++)
      {
	vector<double> depths_histogram = cellDepths.depth_histogram(
	  yIter, xIter, min_depth, max_depth);
	
	for(int bin = 0; bin < nbins + DECORATION_LENGTH; bin++)    
	{
	  assert(yIter < cellDepths.upper_manifold.rows && xIter < cellDepths.upper_manifold.cols);
	  float cell_depth = cellDepths.upper_manifold.at<float>(yIter,xIter);
	  if(bin < CellDepths::METRIC_DEPTH_BINS)
	  {
	    feats[occIter] = depths_histogram[bin];
	  }
	  else
	  {
	    if(rawIter >= raw_pos.size())
	    {
	      cout << "blocks_x = " << subordinate->blocks_x() << endl;
	      cout << "blocks_y = " << subordinate->blocks_y() << endl;
	      cout << "nbins = " << nbins << endl;
	      cout << "occ_pos.size = " << feats.size() << endl;
	      cout << "raw_pos.size = " << raw_pos.size() << endl;	
	      cout << "rawIter = " << rawIter << endl;
	      cout << "occIter = " << occIter << endl;
	    }
	    assert(occIter < feats.size());
	    assert(rawIter < raw_pos.size());
	    bool occluded = (cell_depth < min_depth);
	    if(!occluded /*&& cell_depth <= max_depth*/)
	      feats[occIter] = raw_pos[rawIter];
	    else
	      feats[occIter] = 0;
	    rawIter++;
	  }
	  occIter++;
	}
      }
    return feats;
  }
  
  void OccSliceFeature::compute(const ImRGBZ& im, std::vector< float >& feats)
  {
    // ideally, we'll pad occluded cells with a "1"
    vector<float> raw_pos;
    subordinate->compute(im,raw_pos);
    if(raw_pos.size() == 0)
      feats = raw_pos;
    
    //get the depth for each cell
    CellDepths cellDepths = mkCellDepths(im);   
    
    float min_depth,max_depth;
    comp_depths(cellDepths,min_depth,max_depth);
    max_depth = std::min<float>(max_depth,params::MAX_Z());
    if(min_depth >= max_depth)
      feats.clear();
    else
    {
      feats = decorate_feat(raw_pos,cellDepths,min_depth,max_depth);
      assert(feats.size() == getDescriptorSize());
    }
  }
  
  void OccSliceFeature::setDepthFn(OccSliceFeature::DepthFn depthFn)
  {
    comp_depths = depthFn;
  }

  OccSliceFeature::OccSliceFeature(shared_ptr< DepthFeatComputer > subordinate) : 
    subordinate(subordinate)
  {
  }
  
  string OccSliceFeature::toString() const
  {
    return string("OccSliceFeature_") + subordinate->toString();
  }
  
  Size OccSliceFeature::getBlockSize()
  {
    return subordinate->getBlockSize();
  }

  Size OccSliceFeature::getBlockStride()
  {
    return subordinate->getBlockStride();
  }

  Size OccSliceFeature::getCellSize()
  {
    return subordinate->getCellSize();
  }
  
  int OccSliceFeature::num_cells() 
  {
    int sub_length = subordinate->getDescriptorSize();
    if(subordinate->getNBins() > 0)
      return sub_length/subordinate->getNBins();
    else if(subordinate->cellsPerBlock() == 1)
    {
      int bx = blocks_x(), by = blocks_y();
      assert(bx > 0);
      assert(by > 0);
      return bx*by;
    }
    else
    {
      assert(false);
      return -1;
    }
  }
  
  size_t OccSliceFeature::getDescriptorSize()
  {
    int sub_length = subordinate->getDescriptorSize();
    int length = (sub_length+DECORATION_LENGTH*num_cells());
    assert(length > 0);
    return length;
  }

  void OccSliceFeature::setRanges(float min_depth, float max_depth)
  {
    assert(min_depth < max_depth);
    comp_depths = [min_depth,max_depth](const CellDepths&,float&min,float&max)
    {
      min = min_depth;
      max = max_depth;
    };
  }
  
  int OccSliceFeature::getNBins()
  {
    return subordinate->getNBins() + DECORATION_LENGTH;
  }

  Size OccSliceFeature::getWinSize()
  {
    return subordinate->getWinSize();
  }
  
  template<int BIN>
  static double feature_bin_match_ratio(OccSliceFeature&src,const vector<double>&feat,double BETA)
  {
    assert(feat.size() == src.getDescriptorSize());
    double totalMatch = 0;
    for(int block_y = 0; block_y < src.blocks_y(); block_y++)
    {
      for(int block_x = 0; block_x < src.blocks_x(); block_x++)
      {
	float bin_p = feat.at(src.getIndex(block_x,block_y,0/*1 cell per block*/,BIN));
	totalMatch += bin_p/BETA;
      }
    }    
    totalMatch = totalMatch/(src.blocks_x()*src.blocks_y());
    assert(0 <= totalMatch && totalMatch <= 1);
    return totalMatch;    
  }
  
  double OccSliceFeature::occlusion(const vector<double>&feat)
  {
    return feature_bin_match_ratio<CellDepths::BIN_OCC>(*this,feat,CellDepths::BETA_OCC);
  }  
  
  double OccSliceFeature::background(const vector< double >& orig)
  {
    return feature_bin_match_ratio<CellDepths::BIN_BG>(*this,orig,CellDepths::BETA_HISTOGRAM);
  }

  double OccSliceFeature::real(const vector< double >& orig)
  {
    return 1.0 - occlusion(orig) - background(orig);
  }
  
  void OccSliceFeature::mark_context(Mat&vis,vector<double> feat,bool lrflip)
  {
    require_equal<size_t>(feat.size(),getDescriptorSize());
    assert(vis.type() == DataType<Vec3b>::type);
    double block_width = vis.cols/static_cast<double>(blocks_x());
    double block_height = vis.rows/static_cast<double>(blocks_y());
    for(int rIter = 0; rIter < vis.rows; rIter++)
    {
      int block_y = rIter/block_height; // C++ rounds to zero
      for(int cIter = 0; cIter < vis.cols; cIter++)
      {
	int block_x = cIter/block_width;
	int read_x = block_x, read_y = block_y;
	if(lrflip)
	{
	  read_x = blocks_x() - 1 - block_x;
	}
	
	// colorize the pixel
	Vec3b color(255,255,255);
	float strongest_bin = -inf;
	for(int metric_depth_bin = 0; metric_depth_bin < CellDepths::METRIC_DEPTH_BINS; metric_depth_bin++)
	{
	  float bin_p = feat.at(getIndex(read_x,read_y,0/*1 cell per block*/,metric_depth_bin));
	  
	  if(metric_depth_bin == 0)
	    bin_p /= CellDepths::BETA_OCC;
	  else
	    bin_p /= CellDepths::BETA_HISTOGRAM;
	  
	  if(bin_p > strongest_bin)
	  {
	    strongest_bin = bin_p;
	    
	    // assign color
	    if(metric_depth_bin == 1)
	      color = Vec3b(255,0,0); // FGBG
	    else if(metric_depth_bin == 0)
	      color = Vec3b(0,0,255); // occlusion
	    else
	    {
	      //color = getColor(metric_depth_bin);
	      double g = interpolate_linear(
		metric_depth_bin,0,CellDepths::METRIC_DEPTH_BINS,0,255);
	      color = Vec3b(0,g,0);
	    }
	  }
	}
	vis.at<Vec3b>(rIter,cIter) = max(color,vis.at<Vec3b>(rIter,cIter)); 
      }
    }    
  }
  
  vector< FeatVis > OccSliceFeature::show_planes(vector< double > feat)
  {
    vector<double> raw_pos = undecorate_feat(feat);
    vector<FeatVis> planes = subordinate->show_planes(raw_pos);    
    return planes;    
  }
  
  std::vector< double > OccSliceFeature::undecorate_feat(
    const std::vector< double >&orig)
  {
    vector<double> raw_pos(subordinate->getDescriptorSize());
    int nbins = subordinate->getNBins();
    int ncells = num_cells();
    
    int occIter = 0, rawIter = 0; // the order of these for loops supports these iterators
    for(int xIter = 0; xIter < subordinate->blocks_x(); xIter++)
      for(int yIter = 0; yIter < subordinate->blocks_y(); yIter++)
      {
	for(int bin = 0; bin < nbins + DECORATION_LENGTH; bin++)    
	{
	  if(bin >= DECORATION_LENGTH)
	  {
	    raw_pos[rawIter] = orig[occIter];
	    rawIter++;
	  }
	  occIter++;
	}    
      }
    return raw_pos;
  }
    
  Mat OccSliceFeature::show(const string& title, std::vector< double > occ_pos)
  {
    // get the plnaes
    vector<double> raw_pos = undecorate_feat(occ_pos);
    vector<FeatVis> planes = subordinate->show_planes(raw_pos);
    cout << "OccSliceFeature::show plane count = " << planes.size() << endl;
    
    // combine them into a visualization
    vector<Mat> viss;
    for(int iter = 0; iter < planes.size(); ++iter)
    {
      Mat vis = planes[iter].getPos();
      if(iter == 0)
	mark_context(vis,occ_pos,false);
      viss.push_back(imVGA(vis));
    }
    return tileCat(viss);
  }    
}
