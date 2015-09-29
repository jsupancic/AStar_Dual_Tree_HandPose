/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "Poselet.hpp"
#include "Detector.hpp"
#include "MetaData.hpp"
#include "Eval.hpp"
#include "main.hpp"
#include "Log.hpp"
#include <vector>
#include <bits/unordered_set.h>
#include <boost/graph/graph_concepts.hpp>
#include "Annotation.hpp"

namespace deformable_depth
{
  namespace Poselet
  {
    using namespace std;
    
    static constexpr size_t MIN_POSELET_SIZE = 5;
    // threshold which controls how similar examples need to be 
    // to be in the same poselet
    // to high: 
    // to low : 
    //
    // With BB keypoints 500 is ideal but doesn't work well enough
    static constexpr double MAX_POSELET_RADIUS = 500;
    
    struct ClusterDist
    {
      shared_ptr<MetaData> source;
      shared_ptr<MetaData> dest;
      double dist;
      bool operator< (const ClusterDist&other) const
      {
	return other.dist > this->dist;
      }
    };

    class Poselet : public vector<ClusterDist>
    {
    protected:
      unordered_set<string> covered_dests;
      
    public:
      Poselet(const vector<ClusterDist>&dists) : 
	vector<ClusterDist>(dists)
	{
	};
	
      Poselet(const Poselet& copyme) : 
	vector<ClusterDist>(copyme)
      {
	for(auto&elem : copyme.covered_dests)
	  covered_dests.insert(elem);
      }
      
      bool operator< (const Poselet&other) const
      {
	// reverse order is better because we remove from end of vectors...
	return this->uncovered_size() < other.uncovered_size();
      }
      
      size_t uncovered_size() const
      {
	return size() - covered_dests.size();
      }
      
      void cover(string id)
      {
	covered_dests.insert(id);
      }
    };
    
    struct Error
    {
      double squared, absolute, max;
    };
    
    Error comp_error(const vector<Point2d>&xs/*src*/,const vector<Point2d>&ys/*dst*/,
		    double sx, double sy, double tx, double ty)
    {
      double sse = 0;
      double sae = 0;
      double max_error = -inf;
      
      for(int iter = 0; iter < xs.size(); iter++)
      {
	double x1 = xs[iter].x;
	double x2 = xs[iter].y;
	double y1 = ys[iter].x;
	double y2 = ys[iter].y;
	
	// apply the affine transform
	x1 = sx*x1 + tx;
	x2 = sy*x2 + ty;
	
	// update sse
	double e1 = (x1 - y1)*(x1 - y1);
	double e2 = (x2 - y2)*(x2 - y2);
	double se = e1 + e2;
	sse += se;
	
	// update absolute error
	sae += std::abs(x1 - y1) + std::abs(x2 - y2);
	
	// update max error
	max_error = std::max(max_error,se);
      }
      
      Error error;
      error.squared = sse;
      error.absolute = sae;
      error.max = max_error;
      return error;     
    }
    
    double comp_sse(const vector<Point2d>&xs/*src*/,const vector<Point2d>&ys/*dst*/,
		    double sx, double sy, double tx, double ty)
    {
      return comp_error(xs,ys,sx,sy,tx,ty).squared;
    }
    
    struct ProcrustianMoments
    {
    public:
      ProcrustianMoments(const vector<Point3d>&xs/*src*/,const vector<Point3d>&ys/*dst*/,
			vector<double> weights) : 
			weights(weights)
      {
	// normalize the weights s.t. thresholds are comparable?
	weights = normalize(weights);
	
	x1_sq = 0, x1 = 0, x2_sq = 0, x2 = 0, 
	N = 0, y1x1 = 0, y2x2 = 0, y1 = 0, y2 = 0;
	x3_sq = 0, x3 = 0, y3x3 = 0, y3 = 0;
	
	for(int iter = 0; iter < xs.size(); iter++)
	{
	  x1_sq += xs[iter].x*xs[iter].x*weights[iter];
	  x1    += xs[iter].x*weights[iter];
	  x2_sq += xs[iter].y*xs[iter].y*weights[iter];
	  x2    += xs[iter].y*weights[iter];      
	  N += 1.0*weights[iter];
	  //y1_sq += ys[iter].x*ys[iter].x*weights[iter];
	  y1    += ys[iter].x*weights[iter];
	  //y2_sq += ys[iter].y*ys[iter].y*weights[iter];
	  y2    += ys[iter].y*weights[iter];      
	  //
	  y1x1  += ys[iter].x*xs[iter].x*weights[iter];
	  y2x2  += ys[iter].y*xs[iter].y*weights[iter];
	  
	  // for the z moments.
	  x3 += xs[iter].z;
	  y3 += ys[iter].z;
	  x3_sq += xs[iter].z * xs[iter].z;
	  y3x3 += xs[iter].z * ys[iter].z;
	}      
      }
      
      // input
      double 
	x1_sq, x1, x2_sq, x2, N, y1x1, y2x2, y1, y2;    
      double x3_sq, x3, y3x3, y3;
      vector<double> weights;
	
      // output
      double sx, sy, tx, ty, tz;
    };
    
    void min_dist_1scale(ProcrustianMoments&m)
    { 
      // (1) find a psudo-affine (not rotation) transform from kp_src to kp_dest
      // which minimizes the weighted sum of squared error
      Mat A(4,4,DataType<double>::type,Scalar::all(0));
      Mat b(4,1,DataType<double>::type,Scalar::all(0));
      // scale constraints
      A.at<double>(0,0) = m.x1_sq + m.x2_sq + m.x3_sq;
      A.at<double>(0,1) = m.x1;
      A.at<double>(0,2) = m.x2;
      A.at<double>(0,3) = m.x3;
      // tx 
      A.at<double>(1,0) = m.x1;
      A.at<double>(1,1) = m.N;
      // ty
      A.at<double>(2,0) = m.x2;
      A.at<double>(2,2) = m.N;
      // tz 
      A.at<double>(3,0) = m.x3;
      A.at<double>(3,3) = m.N;
      // setup B
      b.at<double>(0,0) = m.y1x1 + m.y2x2 + m.y3x3;
      b.at<double>(1,0) = m.y1;
      b.at<double>(2,0) = m.y2;
      b.at<double>(3,0) = m.y3;
      // commit to the solver
      Mat x; cv::solve(A,b,x);
      
      // write the output
      m.sx = x.at<double>(0);
      m.sy = x.at<double>(0);
      m.tx = x.at<double>(1);
      m.ty = x.at<double>(2);    
      m.tz = x.at<double>(3);
    }
    
    void min_dist_2scales(ProcrustianMoments&m)
    {
      assert(false);
      Mat A(4,4,DataType<double>::type,Scalar::all(0));
      Mat b(4,1,DataType<double>::type,Scalar::all(0));    
      
      // construct the A and B matrices
      A.at<double>(0,0) = m.x1_sq;
      A.at<double>(0,2) = m.x1;
      A.at<double>(1,1) = m.x2_sq;
      A.at<double>(1,3) = m.x2;
      A.at<double>(2,0) = m.x1;
      A.at<double>(2,2) = m.N;
      A.at<double>(3,1) = m.x2;
      A.at<double>(3,3) = m.N;
      b.at<double>(0,0) = m.y1x1;
      b.at<double>(1,0) = m.y2x2;
      b.at<double>(2,0) = m.y1;
      b.at<double>(3,0) = m.y2;
      cout << "A = " << A << endl;
      cout << "b = " << b << endl;
      // commit to the solver
      Mat x; cv::solve(A,b,x);
      m.sx = x.at<double>(0);
      m.sy = x.at<double>(1);
      m.tx = x.at<double>(2);
      m.ty = x.at<double>(3);
    }
    
    Procrustean2D min_dist(const vector<Point2d>&xs/*dst*/,const vector<Point2d>&ys/*src*/,vector<double> weights)
    {
      assert(xs.size() == ys.size());
      Procrustean2D result;
      
      // (0) assign weights to each keypoint
      
      // (1) find a psudo-affine (not rotation) transform from kp_src to kp_dest
      // which minimizes the weighted sum of squared error
      // compute the values needed by the matrices
      vector<Point3d> xs3, ys3;
      for(const Point2d & pt : xs)
	xs3.push_back(Point3d(pt.x,pt.y,0));
      for(const Point2d & pt : ys)
	ys3.push_back(Point3d(pt.x,pt.y,0));
      ProcrustianMoments m(xs3,ys3,weights);      
      // proc
      //min_dist_2scales(m);
      min_dist_1scale(m);
      
      //printf("min_dist sx = %f sy = %f tx = %f ty %f\n",m.sx,m.sy,m.tx,m.ty);
      if(m.sx < 0 || m.sy < 0)
      {
	result.min_ssd = inf;
	return result; // very bad solution.
      }
      
      // (2) for this affine transform, return the minimal weighted sum of sq. error
      double old_sse = comp_sse(xs,ys,1, 1, 0, 0);
      double sse = comp_sse(xs,ys,m.sx, m.sy, m.tx, m.ty);
      //cout << printfpp("SSE %f => %f",old_sse,sse) << endl;
      result.min_ssd = sse;
      result.s = std::min(m.sx,m.sy);
      result.tx = m.tx;
      result.ty = m.ty;
      return result;    
    }
    
    double min_dist_simple(const vector< Vec3d >& xs, const vector< Vec3d >& ys)
    {
      vector<double> std_x1, std_x2, std_x3, std_y1, std_y2, std_y3;
      
      for(Vec3d p : xs)
      {
	std_x1.push_back(p[0]);
	std_x2.push_back(p[1]);
	std_x3.push_back(p[2]);
      }
      for(Vec3d p : ys)
      {
	std_y1.push_back(p[0]);
	std_y2.push_back(p[1]);
	std_y3.push_back(p[2]);
      }
      
      standardize(std_x1);
      standardize(std_y1);
      standardize(std_x2);
      standardize(std_y2);
      standardize(std_x3);
      standardize(std_y3);
      
      double ssd = 0;
      for(int iter = 0; iter < ys.size();++iter)
      {
	ssd += 
	  (std_x1[iter]-std_y1[iter])*(std_x1[iter]-std_y1[iter]) + 
	  (std_x2[iter]-std_y2[iter])*(std_x2[iter]-std_y2[iter]) + 
	  (std_x3[iter]-std_y3[iter])*(std_x3[iter]-std_y3[iter]);
      }
      
      return ssd;
    }
    
    vector<Point2d> keypoints(MetaData&metadata)
    {
      vector<Point2d> kp;
      
      // extract the annotations
      for(string essential_keypoint : essential_keypoints())
      {
	Point2d keypoint = take2(metadata.keypoint(essential_keypoint).first);
	//cout << printfpp("keypoint (%s,%s): ",metadata.get_filename().c_str(),
	  //	       essential_keypoint.c_str()) 
	  //<< keypoint.x << " " << keypoint.y << endl;
	kp.push_back(keypoint);
      }
      
      // add a keypoint for each bb corner
      Rect_<double> handBB = metadata.get_positives()["HandBB"];
      kp.push_back(handBB.tl());
      kp.push_back(handBB.br());
      kp.push_back(Point2d(handBB.x+handBB.width,handBB.y)); 
      kp.push_back(Point2d(handBB.x,handBB.y+handBB.height));
      
      return kp;
    }
    
    double min_dist_procrustian(MetaData& dst, DetectorResult& src)
    {
      vector<Point2d> dst_keypoints;
      vector<Point2d> src_keypoints;
      
      // perlim check
      auto dst_poss = dst.get_positives();
      set<string> dst_names;
      for(auto && part : dst_poss)
	dst_names.insert(part.first);
      
      // add HandBB
      src_keypoints.push_back(src->BB.tl());
      dst_keypoints.push_back(dst_poss["HandBB"].tl());
      src_keypoints.push_back(src->BB.br());
      dst_keypoints.push_back(dst_poss["HandBB"].br());
      
      // add the finger keypoints
      set<string> common_part_names;
      set<string> src_parts(src->part_names().begin(),src->part_names().end());
      set<string> dst_parts(dst_names.begin(),dst_names.end());
      assert(src_parts.size() > 0);
      assert(dst_parts.size() > 0);
      std::set_intersection(src_parts.begin(),src_parts.end(),
			    dst_parts.begin(),dst_parts.end(),
	std::inserter(common_part_names,common_part_names.begin()));
      if(common_part_names.size() < 1)
      {
	log_once(printfpp("warning: min_dist_procrustian common_part_names size < 1"));
	return inf;
      }
      for(string part_name : common_part_names)
      {
	src_keypoints.push_back(rectCenter(src->getPart(part_name).BB));
	dst_keypoints.push_back(rectCenter(dst_poss[part_name]));
      }
      
      return min_dist(dst_keypoints,src_keypoints,vector<double>(src_keypoints.size(),1)).min_ssd;
    }
    
    vector<double> make_weights(string partname, 
				shared_ptr<MetaData> source,
				vector<Point2d>&kp_src)
    {
      // the root is a special case where everyone matters...
      if(partname == "HandBB")
	return vector<double>(kp_src.size(),1);
      
      // for parts we want to emphasize local context
      vector<double> weights(kp_src.size(),1);
      
      // TODO
      
      return weights;
    }
    
    Procrustean2D min_dist(const map< string, Vec3d >& in_xs, 
			   const map< string, Vec3d >& in_ys, bool do_clamp)
    {
      vector<Point2d> xs;
      vector<Point2d> ys;
      vector<double> weights;
      
      for(auto keypoint : in_xs)
	if(in_ys.find(keypoint.first) != in_ys.end())
	{
	  auto x = in_xs.at(keypoint.first);
	  auto y = in_ys.at(keypoint.first);
	  xs.push_back(Point2d(x[0],x[1]));
	  ys.push_back(Point2d(y[0],y[1]));
	  weights.push_back(1);
	}
      
      Procrustean2D dist = min_dist(xs,ys,weights);
      if(dist.s <= 0)
	dist.min_ssd = inf;
      else
      {
	if(do_clamp)
	  dist.s = clamp<double>(1/4.0,dist.s,4);
	auto errors = comp_error(xs,ys,dist.s, dist.s, dist.tx, dist.ty);
	dist.min_ssd = errors.squared;
	dist.max_se  = errors.max;
      }
      
      return dist;
    }

    Procrustean2D min_dist(const map<string,AnnotationBoundingBox>&xs,const map<string,AnnotationBoundingBox>&ys,bool clamp)
    {
      map<string,Vec3d> xpts, ypts;
      for(auto & x : xs)
      {
	Point2d p = rectCenter(x.second);
	xpts[x.first] = Vec3d(p.x,p.y,0);
      }
      for(auto & y : ys)
      {
	Point2d p = rectCenter(y.second);
	ypts[y.first] = Vec3d(p.x,p.y,0);
      }

      return min_dist(xpts,ypts,clamp);
    }
    
    // return a sorted list of examples by proximity to the target.
    vector<ClusterDist> cluster_of(
      string partname, shared_ptr<MetaData> source,
      vector<shared_ptr<MetaData> > examples)
    {
      vector<ClusterDist> dists;
      
      // compute the distance to each examples
      // random order to avoid lock contention in image loading
      random_shuffle(examples.begin(),examples.end()); 
      for(auto && dest : examples)
      {
	// compute the dist from source to dest.
	auto kp_src = keypoints(*source);
	auto kp_dst = keypoints(*dest);
	vector<double> weights = make_weights(partname,source,kp_src);
	double dist = min_dist(kp_dst,kp_src,weights).min_ssd;
	dists.push_back(ClusterDist{source,dest,dist});
      }
      std::sort(dists.begin(),dists.end());
      log_file << "cluster_of: computed cluster... now visualizing" << endl;
      
      // apply the MAX_POSELET_RADIUS to the cluster
      int iter = 0;
      for(iter = 0; ; iter++)
      {
	if(dists[iter].dist > MAX_POSELET_RADIUS)
	  break; 
	cout << "dist: " << dists[iter].dist << endl;
      }
      dists.erase(dists.begin()+iter,dists.end());
      cout << "allocated a cluster of size: " << dists.size() << endl;
      
      // check the result
      for(ClusterDist & elem : dists)
      {
	assert(elem.dest != nullptr);
	assert(elem.source != nullptr);
      } 
      
      return dists;
    }
    
    static void log_poselet(vector<ClusterDist> poselet_config)
    {
      vector<Mat> vis{show_one(*poselet_config[0].source,Scalar(255,0,0))};
      std::sort(poselet_config.begin(),poselet_config.end());
      TaskBlock log_poselet("log_poselet");
      for(ClusterDist& elem : poselet_config)
	log_poselet.add_callee([&,elem]()
	{
	  Mat vis1 = show_one(*elem.dest);
	  static mutex m; unique_lock<mutex> l(m);
	  vis.push_back(vis1);
	});
      log_poselet.execute();
      log_im("poselet",tileCat(vis));
    }
    
    static vector<Poselet> poselet_set_cover_nms(
      vector<shared_ptr<MetaData> >&all_examples,vector<Poselet> poselets)
    {
      // check preconditions
      for(Poselet & poselet : poselets)
	for(ClusterDist & elem : poselet)
	{
	  assert(elem.dest != nullptr);
	  assert(elem.source != nullptr);
	}
      
      // construct the cover map, for the greedy algorithm
      map<string,bool> cover;
      for(auto && example : all_examples)
      {
	assert(cover.find(example->get_filename()) == cover.end());
	cover[example->get_filename()] = false;
      }
	  
      vector<Poselet> final_poslets;
      while(poselets.size() > 0)
      {
	// sort the poselets by their uncovered size
	std::sort(poselets.begin(),poselets.end());      
	
	// get the next poselet
	Poselet cur_poselet = poselets.back();
	if(cur_poselet.uncovered_size() < MIN_POSELET_SIZE)
	  break;
	final_poslets.push_back(cur_poselet);
	poselets.pop_back();
	
	// for each poselet which becomes covered
	for(ClusterDist&elem : cur_poselet)
	{
	  assert(elem.dest != nullptr);
	  string elem_name = elem.dest->get_filename();
	  if(!cover[elem_name])
	  {
	    cover[elem_name] = true;
	    // elem_name has become covered
	    
	    // mark as covered in all other poselets
	    for(Poselet & other_poselet : poselets)
	      other_poselet.cover(elem_name);
	  }
	}
	
	printf("another poselet passed supression\n");
      }
      
      return final_poslets;
    }
    
    static void poselet_part(string partname,vector<shared_ptr<MetaData> > examples)
    {
      // get a cluster for each example
      vector<Poselet> poselets;
      TaskBlock poselet_part("poselet_part");
      for(auto& example : examples)
      {
	poselet_part.add_callee([&,example]()
	{
	  // get the large clusters
	  Poselet poselet{cluster_of(partname,example,examples)};
	  if(poselet.size() >= MIN_POSELET_SIZE)
	  {
	    // store the resutls
	    static mutex m; unique_lock<mutex> l(m);
	    poselets.push_back(poselet);
	  }
	});
      }
      poselet_part.execute();
      
      // try to solve set cover problem to supress duplicate poselets
      poselets = poselet_set_cover_nms(examples,poselets);
      
      // visualize the final selection
      for(Poselet&poselet : poselets)
	log_poselet(poselet);
    }
  } // end namespace Poselet
  
  void poselet()
  {
    // load the metadata
    vector<shared_ptr<MetaData> > set  = metadata_build_all(default_train_dirs());
    random_shuffle(set.begin(),set.end());
    
    // for each part
    Poselet::poselet_part("HandBB",set);
  }
}
