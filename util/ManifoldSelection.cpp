/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ManifoldSelection.hpp"
#include "util.hpp"
#include <queue>

namespace deformable_depth
{
  
  static vector<float> filter_similar_depths(const vector<float>&depths)
  {
    vector<float> sorted_depths(depths.begin(),depths.end());
    std::sort(sorted_depths.begin(),sorted_depths.end());
    
    vector<float> outDepths;
    for(float depth : sorted_depths)
      if(outDepths.empty())
	outDepths.push_back(depth);
      else if(std::abs(outDepths.back() - depth) > params::DEPTHS_SIMILAR_THRESAH)
	outDepths.push_back(depth);
    return outDepths;
  }
  
  vector<float> manifoldFn_prng(const ImRGBZ&im,const Rect_<double>&bb_src, int max_samples)
  {
    Rect bb = clamp(im.Z,bb_src);

    // setup the sample distribution
    std::mt19937 sample_seq;
    sample_seq.seed(bb.x + bb.y*bb.width);
    std::uniform_int_distribution<int> x_dist(bb.x,bb.x+bb.width-1);
    std::uniform_int_distribution<int> y_dist(bb.y,bb.y+bb.height-1);
    
    // sample some depths
    int samples = std::min<int>(max_samples,bb.size().area());
    vector<float> depths;
    depths.reserve(samples);
    for(int iter = 0; iter < samples; iter++)
    {
      int sample_x = x_dist(sample_seq);
      int sample_y = y_dist(sample_seq);    
      float sampleZ = im.Z.at<float>(sample_y,sample_x);
      if(goodNumber(sampleZ))
	depths.push_back(sampleZ);
    }
    return filter_similar_depths(depths);    
  }  
  
  vector< float > manifoldFn_apxMin(const ImRGBZ& im, const Rect_< double >& bb)
  {
    return manifoldFn_ordApx(im,bb,.1);
  }

  vector< float > manifoldFn_default(const ImRGBZ& im, const Rect_< double >& bb)
  {
    string fnName = g_params.require("MANIFOLD_FN");
    
    if(fnName == "min")
      return manifoldFn_min(im,bb);
    else if(fnName == "ApxMin")
      return manifoldFn_apxMin(im,bb);
    else
      assert(false);
  }
  
  // return another manifold function called on zoomed subimages
  vector<float> manifoldFn_telescope(const ImRGBZ& im, const Rect_< double >& bb)
  {
    vector<float> zs;
    for(double zf : vector<double>{.5,1})
    {
      vector<float> z = manifoldFn_ordApx(im,rectResize(bb,zf,zf),.1);
      zs.insert(zs.end(),z.begin(),z.end());
    }
    return filter_similar_depths(zs);
  }

  vector<float> manifoldFn_all(const ImRGBZ& im, const Rect_< double >& bb)
  {
    vector<float> all_depths;    
    Rect_<double> bb_clamped = clamp(im.Z,bb);
    const Mat z_roi = im.Z(bb_clamped);
    for(int rIter = 0; rIter < z_roi.rows; ++rIter)
      for(int cIter = 0; cIter < z_roi.cols; ++cIter)
      {
	const float & here = z_roi.at<float>(rIter,cIter);
	if(goodNumber(here) && here < params::MAX_Z())
	  all_depths.push_back(here);
      }
    std::sort(all_depths.begin(),all_depths.end());

    return all_depths;
  }
  
  // split with a MinSpanTree [just sort and remove longest distances to cluster]
  // then take mins
  vector<vector<float> > kmax(const ImRGBZ&im, const Rect_<double> & bb, int K = 5)
  {
    struct DepthInterval
    {
      double z1, z2;
      int index;
      bool operator < (const DepthInterval&other) const
      {
	return dist() < other.dist();
      }
      double dist() const
      {
	return std::abs(z1 - z2);
      }
      double begin() const
      {
	return std::min<double>(z1,z2);
      }
      double end() const
      {
	return std::max<double>(z1,z2);
      }
    };

    // build a priority queue of the intervals that we can choose to cut.
    std::priority_queue<DepthInterval> intervals;
    vector<float> all_depths = manifoldFn_all(im, bb); // all_depths will be returned sorted.
    if(all_depths.empty())
      return vector<vector<float>>{};
    for(int iter = 1; iter < all_depths.size(); ++iter)
      intervals.push(DepthInterval{all_depths.at(iter),all_depths.at(iter-1),iter});

    // construct a map (Balanced BST) 
    map<float/*depth*/,int/*k*/> cluster_starts;
    cluster_starts[all_depths.at(0)] = 0;
    float last_pw_dist = inf;
    for(int k = 0; cluster_starts.size() < K and !intervals.empty(); ++k)
    {
      // get the next largest gap
      auto interval = intervals.top();
      intervals.pop();
      // check the interval
      assert(last_pw_dist >= interval.dist());
      last_pw_dist = interval.dist();
      // 
      if(interval.dist() < params::MIN_Z_STEP)
	break;
      cluster_starts[interval.end()] = k + 1;
    }
      
    // cut the intervals to construct the tree.
    vector<vector<float> > clusters(K);
    for(int iter = 0; iter < all_depths.size(); ++iter)
    {
      float depth = all_depths.at(iter);
      int k = cluster_starts.lower_bound(depth)->second;
      clusters.at(k).push_back(depth);
    }
    return clusters;
  }

  vector<float> manifoldFn_kmax_medians(const ImRGBZ&im, const Rect_<double> & bb)
  {
    // return the median for each kmax cluster
    vector<vector<float> > clusters = kmax(im,bb);
    vector<float> use_dists;
    for(auto && cluster : clusters)
    {
      use_dists.push_back(cluster.at(cluster.size()/2));
    }
    return use_dists;
  }

  vector<float> manifoldFn_kmax_starts(const ImRGBZ& im, const Rect_< double >& bb)
  {
    // return the beginning for each kmax cluster
    vector<vector<float> > clusters = kmax(im,bb);
    vector<float> use_dists;    
    for(auto && cluster : clusters)
      if(not cluster.empty())
	use_dists.push_back(cluster.front());

    std::sort(use_dists.begin(),use_dists.end());
    vector<float> final_dists;
    for(auto && depth : use_dists)
      if(final_dists.empty() or final_dists.back() + params::MIN_Z_STEP <= depth)
	final_dists.push_back(depth);
    return final_dists;
  };

  vector<float> manifoldFn_kmax(const ImRGBZ& im, const Rect_< double >& bb)
  {
    //return manifoldFn_kmax_medians(im,bb);
    return manifoldFn_kmax_starts(im,bb);
  }

  vector< float > manifoldFn_ordApx(const ImRGBZ& im, const Rect_< double >& bb, double ord)
  {
    return vector<float>{medianApx(im.Z,bb,ord)};
  }
  
  vector< float > manifoldFn_min(const ImRGBZ& im, const Rect_< double >& bb)
  {
    // compute the min z for each window
//     Rect_<double> bb0 = bb;
//     Size sz(bb0.size().width + 2, bb0.size().height + 2); // +2 for rounding errors
//     Mat struct_elem = getStructuringElement(MORPH_RECT,sz);
//     Mat z_mins; erode(im.Z.clone(),z_mins,struct_elem,Point(0,0)/*anchor*/,
//       1/*iters*/,BORDER_REPLICATE);
    
    //float min = extrema(im.Z(bb)).min;
    float min = inf;
    const Mat z_roi = im.Z(bb);
    for(int rIter = 0; rIter < z_roi.rows; ++rIter)
      for(int cIter = 0; cIter < z_roi.cols; ++cIter)
      {
	const float & here = z_roi.at<float>(rIter,cIter);
	if(goodNumber(here) && here < min)
	  min = here;
      }
    
    
    return vector<float>{min};
  }
  
  vector<float> manifoldFn_discrete_sparse(const ImRGBZ&im, const Rect_<double>&bb, float step)
  {      
    vector<float> all_depths = manifoldFn_all(im, bb);

    vector<float> depths;
    for(float & depth : all_depths)
      if(depths.empty() || std::abs(depths.back() - depth) >= step)
	depths.push_back(depth);
    return depths;
  }

  // floating box median...
  vector<float> manifoldFn_boxMedian(const ImRGBZ&im, const Rect_<double> & bb)
  {
    // debug
    return manifoldFn_kmax(im,bb);

    vector<float> depths = manifoldFn_kmax(im, bb);
    vector<float> allDepths = manifoldFn_all(im, bb);

    bool changed_depths = true;
    while(changed_depths)
    {
      changed_depths = false;
      for(int cluster_iter = 0; cluster_iter < depths.size(); ++cluster_iter)
      {
	// compute the range
	float begin_depth = depths.at(cluster_iter);
	auto begin_iter = std::lower_bound(allDepths.begin(),allDepths.end(),begin_depth);
	size_t begin_index = begin_iter - allDepths.begin();
	// end of the range
	float end_depth   = begin_depth + params::obj_depth();
	auto end_iter   = std::lower_bound(allDepths.begin(),allDepths.end(),end_depth);
	size_t end_index = end_iter - allDepths.begin();
	// take the median
	auto median_index = (begin_index + end_index)/2;
	require_in_range<size_t>(0,median_index,allDepths.size()-1);
	float median = allDepths.at(median_index);
	// update the depth
	float new_depth = median - params::obj_depth()/4;
	float&old_depth = depths.at(cluster_iter);
	if(new_depth != old_depth and std::abs(new_depth - old_depth) < params::obj_depth()/2 )
	{
	  changed_depths = true;
	  old_depth = new_depth;
	}
      }
    }
    
    // sort the list
    std::sort(depths.begin(),depths.end());
    // remove duplicates
    vector<float> final_depths;
    for(auto && depth : depths)
      if(final_depths.empty() or final_depths.back() < depth)
	final_depths.push_back(depth);

    return final_depths;
  }

  ////
  /// SECTION: SORTING
  ////
  vector<float> sort_depths_by_im_area(const vector<float>&depths,const ImRGBZ&im,const Rect_<double>&bb)
  {
    vector<float> all_depths = manifoldFn_all(im, bb);
    std::sort(all_depths.begin(),all_depths.end());	      
    
    // first compute an area for each depth.
    struct ImAreaForDepth
    {
      float depth;
      double area;

      bool operator< (const ImAreaForDepth&o) const
      {
	return area > o.area;
      }
    };
    vector<ImAreaForDepth> areas;
    for(float depth : depths)
    {
      auto lower = std::lower_bound(all_depths.begin(),all_depths.end(),depth);
      auto upper = std::upper_bound(all_depths.begin(),all_depths.end(),depth+params::obj_depth());
      double area = upper - lower;
      areas.push_back({depth,area});
    }

    // convert to output
    std::sort(areas.begin(),areas.end());
    vector<float> out_depths;
    for(auto & area : areas)
      out_depths.push_back(area.depth);
    return out_depths;
  }
}
