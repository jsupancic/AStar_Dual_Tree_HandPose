/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "PairwiseDeformationWindow.hpp"
#include "Log.hpp"
#include "util.hpp"
#include "Orthography.hpp"
#include "Log.hpp"

namespace deformable_depth
{
  using params::RESP_ORTHO_X_RES;
  using params::RESP_ORTHO_Y_RES;  
  
  ///
  /// SECTION: Deformation Model
  ///
  shared_ptr<DetectionManifold> FixedWindowDeformationModel::merge(
    string root_pose,
    shared_ptr< DetectionManifold > root_manifold,
    PartMessages& part_manifolds,
    pw_cost_fn pwResp, const ImRGBZ&im,DetectionFilter&filter
  ) const
  {
    shared_ptr<DetectionManifold> joint_manifold = root_manifold;
    // loop over parts
    for(auto&&part_manifold_pair : part_manifolds)
    {
      map<string, DetectionManifold >&part_mixture = part_manifold_pair.second;
      const FixedWindowPairwiseModel&window = 
	pw_models.at(root_pose+"-"+part_manifold_pair.first);
      // loop over mixture elements for one part
      bool skiped_all_elements = true;
      for(auto&&mixture_pair : part_mixture)
      {
	// TODO: for now, only consider mixture elements compatible with the root?
	if(mixture_pair.first/*part's pose*/ != root_pose)
	  continue;
	skiped_all_elements = false;
	
	auto&part_manifold = mixture_pair.second;
	joint_manifold = 
	  window.merge_manifolds(
	    *joint_manifold,part_manifold,
	    pwResp,part_manifold_pair.first,filter);
      }
      if(skiped_all_elements)
	cout << "warning: skiped all elements" << endl;
    }
    
    return root_manifold;
  }
  
  void FixedWindowDeformationModel::train_pair(
    vector< shared_ptr< MetaData > > train_set, 
    string pose_name, string root_name, string part_name)
  {
    parts.insert(part_name);
    FixedWindowPairwiseModel pw_model(train_set,root_name,part_name); 
    unique_lock<mutex> l(m);
    pw_models.insert(
      pair<string,FixedWindowPairwiseModel>
      (string(pose_name+"-"+part_name),pw_model));
  }
  
  Vec2d FixedWindowDeformationModel::get_card_position_normalized(string pose, string part) const
  {
    string pw_model_name = pose+"-"+part;
    const FixedWindowPairwiseModel&pw_model = pw_models.at(pw_model_name);
    
    Vec2d offset = pw_model.getPersOffset();
    
    return offset;
  }
  
  Mat FixedWindowDeformationModel::vis_model(
    Mat& background, DetectorResult& top_det) const
  {
    Mat vis = background.clone();
    
    // draw the detected BBs in blue
    for(string part_name : top_det->part_names())
    {
      Rect_<double> bb = top_det->getPart(part_name).BB;
      rectangle(vis,bb.tl(),bb.br(),Scalar(255,0,0));
    }
    
    // draw the anchor positions in green
    cout << "top_det->scale_factor: " << top_det->scale_factor << endl;
    for(auto && part : parts)
    {
      Vec2d offset = get_card_position_normalized(top_det->pose,part);
      cout << "offset: " << offset << endl;
      // root + offset = pw_pos
      Point part_center(
	rectCenter(top_det->BB).x + offset[0]*top_det->BB.width,
	rectCenter(top_det->BB).y + offset[1]*top_det->BB.height);
      circle(vis,part_center,5,Scalar(0,255,0));
    }
    
    return vis;
  }
  
  void read(cv::FileNode node, deformable_depth::FixedWindowDeformationModel&model)
  {
    read(node["displacement_model"],model.pw_models);
  }
  
  void write(cv::FileStorage&fs, 
	     std::string, 
	     const deformable_depth::FixedWindowDeformationModel&model)
  {
    fs << "displacement_model"; write(fs,model.pw_models);
  }   
  
  /// SECTION: FixedWindowPairwiseModel
  FixedWindowPairwiseModel::FixedWindowPairwiseModel(
    SupervisedMixtureModel::TrainingSet& training_set,
    string part1, string part2)
  {
    Vec2d root(0,0);
    Rect_<double> window;
    
    // init 
    offset = pers_offset = Vec2d(0,0);
    min_offset_x = min_offset_y = min_scale = +inf;
    max_offset_x = max_offset_y = max_scale = -inf;
    int num_visible = 0;
    
    for(shared_ptr<MetaData>&metadata : training_set)
    {
      // get the bbs
      shared_ptr<const ImRGBZ> im = metadata->load_im();
      auto parts = metadata->get_positives();
      
      // check for cases of occlusion
      if(parts.find(part1) == parts.end() || parts.find(part2) == parts.end())
	continue;
      
      // get the offset
      // and current scale
      Vec2d current_offset;
      Vec2d cur_perspective_offset;
      Rect_<double> ortho_root, ortho_part;
      double current_scale;
      part_offset(parts.at(part1), parts.at(part2),
		  im,current_offset,cur_perspective_offset,current_scale,
		  ortho_root,ortho_part);
      
      // update statistics
      offset += current_offset;
      pers_offset += cur_perspective_offset;
      cout << "pers_offset: " << pers_offset << endl;
      min_scale = std::min(min_scale,current_scale);
      min_offset_x = std::min(min_offset_x,current_offset[0]);
      min_offset_y = std::min(min_offset_y,current_offset[1]);
      max_scale = std::max(max_scale,current_scale);
      max_offset_x = std::max(max_offset_x,current_offset[0]);
      max_offset_y = std::max(max_offset_y,current_offset[1]);
      num_visible++;
    }
    
    offset[0] /= training_set.size();
    offset[1] /= training_set.size();
    pers_offset[0] /= training_set.size();
    pers_offset[1] /= training_set.size();
    
    #pragma omp critical(LOG)
    {
      log_file << "=====" << part1 << " , " << part2 << "=====" << endl;
      log_file << "FixedWindowPairwiseModel trained, offset = " << 
	offset[0] << ", " << offset[1] << endl;
      log_file << "FixedWindowPairwiseModel " << num_visible << " visible parts" << endl;
      log_file << 
	printfpp("FixedWindowPairwiseModel window: x: [%f %f] y: [%f %f]",
		  min_offset_x,max_offset_x, min_offset_y, max_offset_y) << endl;
      log_file << "area = " << (max_offset_x - min_offset_x)*(max_offset_y - min_offset_y) << endl;
    }
  }

  FixedWindowPairwiseModel::~FixedWindowPairwiseModel()
  {
  }
  
  Vec2i FixedWindowPairwiseModel::getOffset() const
  {
    return Vec2i(offset[0],offset[1]);
  }

  Vec2d FixedWindowPairwiseModel::getPersOffset() const
  {
    return pers_offset;
  }
  
  Vec2d FixedWindowPairwiseModel::getScales() const
  {
    return Vec2d(.75*min_scale,1.25*max_scale);
  }

  Vec2i FixedWindowPairwiseModel::getWindowSize() const
  {
    double generalization_fudge = 1.5;
    double x_size = generalization_fudge*(max_offset_x - min_offset_x);
    double y_size = generalization_fudge*(max_offset_y - min_offset_y);
    return Vec2i(x_size,y_size);
  }
  
  shared_ptr< DetectionManifold > FixedWindowPairwiseModel::merge_manifolds(
    DetectionManifold& m1, DetectionManifold& m2, 
    pw_cost_fn pwResp, string part_name, DetectionFilter&filter) const
  {
    EPM_Statistics stats;
    Vec2i window = this->getWindowSize(), 
	  offset = this->getOffset(), 
	  scales = this->getScales();
    
    int x_res = m1.shape()[0];
    int y_res = m2.shape()[1];    
    assert(m1.shape()[0] == m2.shape()[0] && m1.shape()[1] == m2.shape()[1]);
    shared_ptr<DetectionManifold> result(new DetectionManifold(x_res,y_res));
    //log_file << "merge_manifolds: " << x_res << ", " << y_res << endl;
    
    long pairs = 0;
    for(int xIter = 0; xIter < x_res; xIter++)
      for(int yIter = 0; yIter < y_res; yIter++)
      {
	for(auto && det1_ptr = m1[xIter][yIter].begin(-inf); 
	    det1_ptr != m1[xIter][yIter].end(inf); ++det1_ptr)
	{
	  if(ManifoldCell::deref(*det1_ptr) == nullptr)
	    continue;
	  Detection& det1 = *ManifoldCell::deref(*det1_ptr);
	  int x_min = std::max<int>(0,xIter + offset[0] - window[0]/2);
	  int y_min = std::max<int>(0,yIter + offset[1] - window[1]/2);
	  int x_max = std::min<int>(xIter + offset[0] + window[0]/2,x_res-1);
	  int y_max = std::min<int>(yIter + offset[1] + window[1]/2,y_res-1);
	  for(int otherX = x_min; otherX < x_max; otherX++)
	    for(int otherY = y_min; otherY < y_max; otherY++)
	    {
	      for(auto && det2_ptr = m2[otherX][otherY].begin(-inf);
		  det2_ptr != m2[otherX][otherY].end(inf); ++det2_ptr)
	      {
		// merge a pair of detections into the joint manifold
		if(ManifoldCell::deref(*det2_ptr) == nullptr)
		  continue;
		Detection& det2 = *ManifoldCell::deref(*det2_ptr);
		DetectorResult joint = merge_manifolds_one_pair(
		  scales,det1,det2,pwResp,pairs,part_name,stats,filter);
		(*result)[xIter][yIter].emplace(joint);
	      }
	    }
	  }
      }
    //log_file << "merge_manifolds: compared " << pairs << " pairs of detections" << endl;
    
    assert(m1.shape()[0] == result->shape()[0] && m1.shape()[1] == result->shape()[1]);
    return result;
  }    
  
  /// SECTION: Serialization
  void write(FileStorage& fs, string& , const FixedWindowPairwiseModel&model)
  {
    fs << "{";
    fs << "offset" << model.offset;
    fs << "min_scale" << model.min_scale;
    fs << "max_scale" << model.max_scale;
    fs << "min_offset_x" << model.min_offset_x;
    fs << "min_offset_y" << model.min_offset_y;
    fs << "max_offset_x" << model.max_offset_x;
    fs << "max_offset_y" << model.max_offset_y;   
    fs << "}";
  }
  
  void read(FileNode node, FixedWindowPairwiseModel& model, FixedWindowPairwiseModel )
  {
    read<double>(node["offset"],model.offset);
    node["min_scale"] >> model.min_scale;
    node["max_scale"] >> model.max_scale;
    node["min_offset_x"] >> model.min_offset_x;
    node["min_offset_y"] >> model.min_offset_y;
    node["max_offset_x"] >> model.max_offset_x;
    node["max_offset_y"] >> model.max_offset_y;       
  }  
}
 