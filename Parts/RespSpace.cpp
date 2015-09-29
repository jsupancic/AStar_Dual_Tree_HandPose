/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "RespSpace.hpp"
#include "util_real.hpp"
#include <functional>
#include "Orthography.hpp"
#include "Log.hpp"
#include "util_file.hpp"
#include "Poselet.hpp"
#include "LibHandSynth.hpp"
#include "util_rect.hpp"
#include "util.hpp"
#include "OcclusionReasoning.hpp"

namespace deformable_depth
{
  using params::RESP_ORTHO_X_RES;
  using params::RESP_ORTHO_Y_RES;
  
  ///
  /// SECTION: ManifoldCell_Tree
  /// 
  double ManifoldCell_Tree::size() const
  {
    return contents.size();
  }
  
  const DetectorResult& ManifoldCell_Tree::deref(
    const std::pair< const float, DetectorResult >& iter)
  {
    return iter.second;
  }

  DetectorResult& ManifoldCell_Tree::deref(std::pair< const float, DetectorResult >& iter)
  {
    return iter.second;
  }

  deformable_depth::ManifoldCell_Tree::iterator ManifoldCell_Tree::begin(float min_area)
  {
    return contents.lower_bound(min_area);
  }

  deformable_depth::ManifoldCell_Tree::iterator ManifoldCell_Tree::end(float max_area)
  {
    return contents.upper_bound(max_area);
  }

  void ManifoldCell_Tree::merge(const ManifoldCell_Tree& other)
  {
    for(auto iter = other.contents.begin(); iter != other.contents.end(); ++iter)
      contents.insert(*iter);
  }

  float ManifoldCell_Tree::max_score()
  {
    float max_score = -inf;
    for(auto iter = contents.begin(); iter != contents.end(); ++iter)
      max_score = std::max(max_score, iter->second->resp);
    return max_score;
  }

  void ManifoldCell_Tree::emplace(const DetectorResult& det)
  {
    contents[det->BB.area()] = det;
  }

  int ManifoldCell_Tree::conflict_count(const ManifoldCell_Tree& other)
  {
    // never conflicts because it can always grow.
    return 0;
  }
  
  ///
  /// SECTION: ManifoldCell_3Occ
  ///
  ManifoldCell_3Occ::iterator ManifoldCell_3Occ::begin()
  {
    return contents.begin();
  }

  ManifoldCell_3Occ::iterator ManifoldCell_3Occ::end()
  {
    return contents.end();
  }

  void ManifoldCell_3Occ::emplace(const DetectorResult& det)
  {
    // ignore empty detections
    if(det == nullptr)
      return;
    
    // find a spot
    Cell_Type ct = cellType(det);
    while(ct >= contents.size())
      contents.push_back(DetectorResult());
    
    // add if better
    DetectorResult&here = contents[ct];
    if(here == nullptr || here->resp < det->resp)
      contents[ct] = det;
    
    assert(contents.size() <= 3);
  }

  float ManifoldCell_3Occ::max_score()
  {
    float max_score = -inf;
    for(DetectorResult & result : contents)
      if(result != nullptr)
	max_score = std::max<float>(max_score,result->resp);
    
    return max_score;
  }

  void ManifoldCell_3Occ::merge(const ManifoldCell_3Occ& from_other)
  {
    for(const DetectorResult & other_result : from_other.contents)
    {
      emplace(other_result);
    }
  }
  
  int ManifoldCell_3Occ::conflict_count(const ManifoldCell_3Occ& other)
  {
    int conflicts = 0;
    
    for(int iter = 0; iter < std::min(contents.size(),other.contents.size()); ++iter)
      if(contents[iter] != nullptr && other.contents[iter] != nullptr)
	++conflicts;
    
    return conflicts;
  }
  
  ManifoldCell_3Occ::Cell_Type ManifoldCell_3Occ::cellType(const DetectorResult& result)
  {
    if(result->occlusion <= 2.0/3.0)
    {
      // the object is fully visible
      return Cell_Type::visible;
    }
    else if(result->occlusion <= 1.0/3.0)
    {
      // the object is partially occluded
      return Cell_Type::part_occ;
    }
    else
    {
      // the object is fully occluded
      return Cell_Type::full_occ;
    }    
  }
  
  ///
  /// SECTION: Manifold
  /// 
  
  void manifold_side_lengths(const ImRGBZ&im,float&min_side,float&max_side)
  {
    float cm_to_pixels = ::sqrt(
      (RESP_ORTHO_X_RES*RESP_ORTHO_Y_RES)/(
	im.camera.widthAtDepth(params::MAX_Z())*im.camera.heightAtDepth(params::MAX_Z())));    
    min_side = 10, max_side = 20; // in cm
    min_side *= cm_to_pixels; max_side *= cm_to_pixels;
  }
  
  shared_ptr< DetectionManifold > init_manifold(
    const ImRGBZ&im)
  {
    shared_ptr<DetectionManifold> result(new DetectionManifold(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES));
    return result;
  }
  
  void manifold_emplace_one(
    shared_ptr<Detection>&detection,const ImRGBZ&im,DetectionManifold&result,
    float min_side,float max_side)
  {
    // map image coordinates to manifold coordinates
    auto bb = detection->BB;
    float z = detection->getDepth(im);
    auto ortho_bb = map2ortho(im.camera,Size(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES),bb,z);
    //if(ortho_bb.area() < min_side*min_side || ortho_bb.area() > max_side*max_side)
      //return;
    //float x = bb.x + bb.width/2;
    //float y = bb.y + bb.height/2;
    //if(x < 0 || y < 0 || x >= im.Z.cols || y >= im.Z.rows)
      //return;
    Point2d ortho_center = rectCenter(ortho_bb);
    float ox = ortho_center.x, oy = ortho_center.y;
    if(ox < 0 || oy < 0 || ox >= RESP_ORTHO_X_RES || oy >= RESP_ORTHO_Y_RES)
      return;
    
    // actually write the manifold.
    (result)[ox][oy].emplace(detection);
  }
  
  shared_ptr< DetectionManifold > create_manifold(const ImRGBZ&im, DetectionSet& detections)
  { 
    float min_side, max_side;
    manifold_side_lengths(im,min_side,max_side);
    shared_ptr<DetectionManifold> result = init_manifold(im);
    
    for(auto &&detection : detections)
      manifold_emplace_one(detection,im,*result,min_side,max_side);
    
    return result;
  }
  
  map<string, DetectionManifold > create_sort_manifolds(const ImRGBZ& im, DetectionSet& detections)
  {
    map<string, DetectionManifold > result;
    float min_side, max_side;
    manifold_side_lengths(im,min_side,max_side);    
    
    for(auto && detection : detections)
    {
      if(result.find(detection->pose) == result.end())
	result.insert(pair<string,DetectionManifold>(detection->pose,*init_manifold(im)));
      manifold_emplace_one(detection,im,result.at(detection->pose),min_side,max_side);
    }
        
    return result;
  }
  
  Mat draw_manifold(DetectionManifold& manifold)
  {
    // create an image of the max resp per cell
    int X_RES = manifold.shape()[0];
    int Y_RES = manifold.shape()[1];
    Mat manifoldImage(Y_RES,X_RES,DataType<float>::type);
    for(int yIter = 0; yIter < Y_RES; yIter++)
      for(int xIter = 0; xIter < X_RES; xIter++)
	manifoldImage.at<float>(yIter,xIter) = manifold[xIter][yIter].max_score();
    return imageeq("",manifoldImage,false,false);
  }
  
  DetectorResult merge_manifolds_one_pair(
    Vec2d scales,
    const Detection& root,const Detection& part,
    pw_cost_fn pwResp,
    long&pairs, string part_name, EPM_Statistics&stats,
    DetectionFilter&filter
  )
  { 
    // check that scale constraints are satisifed.
    double current_scale = ::sqrt((double)part.BB.area())/::sqrt((double)root.BB.area());
    if(current_scale < scales[0] || current_scale > scales[1])
      return stats.count("reject_scale"), nullptr;
    
    // compute pairwise cost and possibly take max
    bool good_input = goodNumber(root.resp) && goodNumber(part.resp);
    float newResp = pwResp(root,part); 
    if(newResp == -inf)
      return nullptr;
    pairs++;
    bool good_output = goodNumber(newResp);
    bool good_pw_resp = !good_input || good_output;// good_input => good_output
    if(!good_pw_resp)
    {
      log_file << "f(" << root.resp << ", " << part.resp << ") = "<< newResp << endl;
      log_file << "good_input" << good_input << endl;
      log_file << "good_output" << good_output << endl;
      log_file << "failure: bad pw resp function!" << endl;
      assert(good_pw_resp); // fail
    }

    // get the joint detection object
    DetectorResult joint(new Detection());

    // update the joint detection object
    *joint = root;
    joint->resp = newResp;
    joint->emplace_part(part_name,part,false); // was filter.testing_mode 
    
    if(part.occlusion > .5)
      stats.count("occluded_parts_accepted");
    else
      stats.count("unoccluded_parts_accepted");
    
    return joint;
  }
  
  /// SECTION: DetectionManifold  
  std::size_t deformable_depth::DetectionManifold::xSize()
  {
    return (*this).shape()[0];
  }

  std::size_t deformable_depth::DetectionManifold::ySize()
  {
    return (*this).shape()[1];
  }

  DetectionManifold DetectionManifold::max_pool(int amount)
  {
    DetectionManifold pooled_manifold(this->xSize(),this->ySize());
    
    // for each cell
    for(int yTo = 0; yTo < ySize(); yTo++)
      for(int xTo = 0; xTo < xSize(); xTo++)
	// look over a neighbourhood
	for(int xFrom = xTo - amount; xFrom <= xTo + amount; xFrom++)
	  for(int yFrom = yTo - amount; yFrom <= yTo + amount; yFrom++)
	  {
	    bool valid_loc = yFrom >= 0 && yFrom < ySize() && xFrom >= 0 && xFrom < xSize();
	    if(!valid_loc)
	      continue;
	    bool will_overwrite = 
	      pooled_manifold[xTo][yTo].conflict_count((*this)[xFrom][yFrom]) > 0;
	    if(valid_loc && !will_overwrite)
	      pooled_manifold[xTo][yTo].merge((*this)[xFrom][yFrom]);
	  }
	  
    return pooled_manifold;
  }
  
  DetectionManifold::DetectionManifold(int xres, int yres) : 
    boost::multi_array<ManifoldCell,2>(boost::extents[xres][yres])
  {
  }
    
  void DetectionManifold::concat(const DetectionManifold& other)
  {
    for(int xIter = 0; xIter < xSize(); xIter++)
      for(int yIter = 0; yIter < ySize(); yIter++)
      {
	auto &here  = (*this)[xIter][yIter];
	auto &there = other  [xIter][yIter];
	here.merge(there);
      }
  }
  
  void part_offset(
    Rect_<double> bb_root, 
    Rect_<double> bb_part,
    const shared_ptr<const ImRGBZ>&im,
    Vec2d&current_offset,
    Vec2d&cur_perspective_offset,
    double&current_scale,
    Rect_<double>&ortho_root, 
    Rect_<double>&ortho_part
  )
  {
    //Rect_<double> bb_root = root->BB;
    //Rect_<double> bb_part = part->BB;
    
    // to ortho: OBJECT
    float z1 = extrema(im->Z(bb_root)).min;
    ortho_root = 
      map2ortho(im->camera,
		Size(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES),
		bb_root,z1);
    // to ortho: PART
    float z2 = extrema(im->Z(bb_part)).min;
    ortho_part   = 
      map2ortho(im->camera,
		Size(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES),
		bb_part,z2);
      
    // find the current offset
    Vec2d object_center(ortho_root.x+ortho_root.width/2,ortho_root.y+ortho_root.height/2);
    Vec2d part_center(ortho_part.x + ortho_part.width/2,ortho_part.y+ortho_part.height/2);
    current_offset = part_center - object_center;   
    
    // update the perspective offset
    Point cpo = rectCenter(bb_part) - rectCenter(bb_root);
    cur_perspective_offset = Vec2d((double)cpo.x,(double)cpo.y);
    cur_perspective_offset[0] /= bb_root.width;
    cur_perspective_offset[1] /= bb_root.height;
    
    current_scale = ::sqrt(ortho_part.area())/::sqrt(ortho_root.area());
  }
}
