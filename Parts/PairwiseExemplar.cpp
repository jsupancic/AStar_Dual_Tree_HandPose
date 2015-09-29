/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "PairwiseExemplar.hpp"
#include "Orthography.hpp"
#include "OcclusionReasoning.hpp"
#include "Log.hpp"
#include "RespSpace.hpp"
#include "Poselet.hpp"
#include "LibHandSynth.hpp"
#include "util_file.hpp"
#include "util.hpp"
#include "InverseKinematics.hpp"

namespace deformable_depth 
{
  using params::RESP_ORTHO_X_RES;
  using params::RESP_ORTHO_Y_RES;  
  
  ///
  /// SECTION: 
  ///
  void ExemplarPairwiseModel::compute_scale_offset(
    MetaData&exemplar,
    DetectionFilter&filter,
    string&part_name,
    Rect_<double> bb_joint_detection_root,
    Vec2d&scale_range,
    double&raw_part_min_area,double&raw_part_max_area, bool&part_occluded
  ) const
  {
    AnnotationBoundingBox bb_root = exemplar.get_positives()[root_name];
    AnnotationBoundingBox bb_part = exemplar.get_positives()[part_name];    
    double raw_scale = ::sqrt(bb_part.area())/::sqrt(bb_root.area());
    scale_range = filter.testing_mode?
      Vec2d(2.0/3.0*raw_scale,3.0/2.0*raw_scale):Vec2d(2.0/3.0*raw_scale,3.0/2.0*raw_scale);
    raw_part_min_area = std::pow(scale_range[0],2)*bb_joint_detection_root.area();
    raw_part_max_area = std::pow(scale_range[1],2)*bb_joint_detection_root.area();
    assert(raw_part_min_area < raw_part_max_area);
    //cout << "scale_range: " << scale_range << endl;
    //cout << printfpp("raw pix range = %f to %f",raw_part_min_area,raw_part_max_area) << endl;
    part_occluded = !bb_part.visible;
  }
  
  void ExemplarPairwiseModel::compute_part_offset_ortho(
    MetaData&exemplar,
    DetectionFilter&filter,const string&part_name,
    Rect_<double> root_ortho_bb,
    double&part_x,double&part_y) const
  {
    Point2d ortho_root_center = rectCenter(root_ortho_bb);
    Rect_<double> ortho_root, bb_root = exemplar.get_positives()[root_name];
    Rect_<double> ortho_part, bb_part = exemplar.get_positives()[part_name];
    Vec2d current_offset;
    Vec2d cur_perspective_offset;
    double ortho_scale;
    part_offset(bb_root, bb_part,
		exemplar.load_im(),
		current_offset,
		cur_perspective_offset,
		ortho_scale,
		ortho_root, ortho_part);
    Point2d ortho_center = rectCenter(ortho_root);
    part_x = current_offset[0] + ortho_root_center.x;
    part_y = current_offset[1] + ortho_root_center.y;    
  }
  
  void ExemplarPairwiseModel::compute_part_offset_affine(
    MetaData&exemplar,
    DetectionFilter&filter,const string&part_name,
    Rect_<double> root_ortho_bb,
    double&part_x,double&part_y) const  
  {
    // compute an affine transform from the exemplar root bb to root_ortho_bb
    Rect_<double> bb_root = exemplar.get_positives()[root_name];
    Rect_<double> bb_part = exemplar.get_positives()[part_name];
    Mat transform = affine_transform(bb_root,root_ortho_bb);
    
    // get the part_x and part_y by applying said transform the the part bb
    Rect_<double> ortho_part = rect_Affine(bb_part, transform);
    Point2d ortho_part_center = rectCenter(ortho_part);
    part_x = ortho_part_center.x;
    part_y = ortho_part_center.y;
  }
  
  void ExemplarPairwiseModel::merge_check_part_candidate
  (
    DetectorResult&exemplars_result,
    DetectorResult&best_merged,
    const ManifoldCell_Tree::iterator::value_type*part_candidate,
    pw_cost_fn pwResp,
    string&part_name,
    DetectionFilter&filter,
    EPM_Statistics&stats,long&pairs,
    double&part_candidates,
    Vec2d scale_range,bool part_occluded
  ) const
  {
    // shouldn't happen
    if(ManifoldCell::deref(*part_candidate) == nullptr) 
      return;
    assert(exemplars_result != nullptr);
    assert(ManifoldCell::deref(*part_candidate) != nullptr);
    
    // update stats for debugging and error checking
    if(ManifoldCell::deref(*part_candidate)->occlusion > .5)
      stats.count("occluded_parts_considered");
    stats.count("total_parts_considered");
    part_candidates++;
    
    // try to filter out problem cases.
    float z_offset = std::abs(exemplars_result->depth - ManifoldCell::deref(*part_candidate)->depth);
    if(z_offset > (filter.testing_mode?
      exemplars_result->z_size : exemplars_result->z_size /*cm*/))
    {
      stats.count("continue z_offset");
      return;
    }
    if(rectIntersect(exemplars_result->BB,ManifoldCell::deref(*part_candidate)->BB) <= 0)
    {
      stats.count("continue rect_intersect");
      return;
    }
    if(part_occluded != ManifoldCell::deref(*part_candidate)->is_occluded())
    {
      stats.count("continue occlusion");
      return;
    }
      
    // merge if valid
    DetectorResult merged = merge_manifolds_one_pair(
      scale_range,*exemplars_result,*ManifoldCell::deref(*part_candidate),
      pwResp,pairs, part_name,stats,filter); 
    if(merged == nullptr)
    {
      stats.count("continue merge failed");
      return;
    }
    if(best_merged == nullptr || best_merged->resp < merged->resp)
      best_merged = merged;    
  }
  
  // try to merge all the parts into a root_candidate for a given
  // exemplar.
  DetectorResult ExemplarPairwiseModel::merge_one_exemplar_position
  (
    shared_ptr<MetaData>&exemplar,
    DetectorResult&root_candidate,
    PartMessages& part_manifolds,
    pw_cost_fn pwResp,
    string root_pose,
    long&pairs, EPM_Statistics&stats,
   const ImRGBZ&im,DetectionFilter&filter
  ) const
  {
    DetectorResult exemplars_result(new Detection);
    *exemplars_result = *root_candidate;
    Rect_<double> root_BB = exemplars_result->BB;
    float root_z = exemplars_result->getDepth(im);
    Rect_<double> root_ortho_bb = 
      map2ortho(im.camera,
		Size(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES),
		root_BB,root_z);    
	    
    // for each part of exemplar
    for(string part_name : parts)
    {
      // compute the search location for the part
      Vec2d scale_range;
      double raw_part_min_area, raw_part_max_area;
      bool part_occluded;
      compute_scale_offset(*exemplar,filter,part_name,root_BB,scale_range,
			   raw_part_min_area, raw_part_max_area,part_occluded);
      double part_x, part_y;
      compute_part_offset_affine(*exemplar,filter,part_name,root_ortho_bb,part_x,part_y);
      if(part_x >= RESP_ORTHO_X_RES || part_x < 0)
	return stats.count("reject_window"), nullptr;
      if(part_y >= RESP_ORTHO_Y_RES || part_y < 0)
	return stats.count("reject_window"), nullptr;
      
      DetectorResult best_merged;
      auto&manifold_cell = part_manifolds.at(part_name).at(root_pose)[part_x][part_y];
      double part_candidates = 0; 
      for(auto&&part_candidate = manifold_cell.begin(raw_part_min_area); 
	  part_candidate != manifold_cell.end(raw_part_max_area); ++part_candidate)
      {
	merge_check_part_candidate(
	  exemplars_result,best_merged,&*part_candidate,
	  pwResp,part_name,filter,stats,pairs,part_candidates,scale_range,part_occluded);
      }
      stats.count("continue scale",manifold_cell.size() - part_candidates);   
      
      if(best_merged == nullptr)
	return stats.count("reject_merge"), nullptr;
      int best_merged_part_count = best_merged->part_names().size();
      int exemplars_result_count = exemplars_result->part_names().size();
      assert(best_merged_part_count - 1 == exemplars_result_count);
      exemplars_result = best_merged;
    }
    
    //assert(parts.size() == 5);
    assert(exemplars_result->part_names().size() > 0);
    //assert(exemplars_result->parts->size() == 5);
    return stats.count("valid"), exemplars_result;
  }
  
  shared_ptr<DetectionManifold> ExemplarPairwiseModel::merge_prep_manifold
  (string root_pose,
   shared_ptr< DetectionManifold > root_manifold,
    PartMessages& part_manifolds,DetectionFilter&filter,
   EPM_Statistics&stats,const ImRGBZ&im
  ) const
  {
    // allocate our result
    int x_res = root_manifold->xSize();
    int y_res = root_manifold->ySize();
    shared_ptr<DetectionManifold> joint_manifold
      (new DetectionManifold(root_manifold->xSize(),root_manifold->ySize()));
    // DEBUG
    for(int xIter = 0; xIter < x_res; xIter++)
      for(int yIter = 0; yIter < y_res; yIter++)
	ManifoldCell&cell = (*joint_manifold)[xIter][yIter];
      
    // apply max pooling to the part manifolds.
    double pool_param = filter.testing_mode?3:3;
    Vec2d pool_win = cm2ortho(Vec2d(pool_param,pool_param),Size(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES),im.camera);
    double pool_dist = std::sqrt(pool_win.dot(pool_win));
    log_once(printfpp("pool dist = %f",pool_dist));
    for(string part_name : parts)
    {
      if(part_manifolds.find(part_name) == part_manifolds.end() ||
	 part_manifolds.at(part_name).find(root_pose) == part_manifolds.at(part_name).end())
      {
	string err_message = printfpp("warning %s: couldn't find %s %s",
				      im.filename.c_str(),part_name.c_str(),root_pose.c_str());
	cout << err_message << endl;
	log_file << err_message << endl;
	DetectionSet no_dets{};
	return nullptr;
// 	part_manifolds[part_name].insert(
// 	  std::pair<string,DetectionManifold>(root_pose,*create_manifold(im,no_dets))); 
      }
      else
      {
	DetectionManifold&cur_man = part_manifolds.at(part_name).at(root_pose);
	cur_man = cur_man.max_pool(pool_dist);
      }
    }    
    
    return joint_manifold;
  }
  
  void ExemplarPairwiseModel::merge_per_position
  (shared_ptr<DetectionManifold> joint_manifold,
   string root_pose,
   shared_ptr< DetectionManifold > root_manifold,
   PartMessages& part_manifolds,pw_cost_fn pwResp,
   DetectionFilter&filter,EPM_Statistics&stats,const ImRGBZ&im
  ) const
  {
    int x_res = root_manifold->xSize();
    int y_res = root_manifold->ySize();    
    
    // for each position
    long pairs = 0;
    for(int xIter = 0; xIter < x_res; xIter++)
      for(int yIter = 0; yIter < y_res; yIter++)
      {
	auto & root_cell = (*root_manifold)[xIter][yIter];
	for(auto&&root_candidate = root_cell.begin(-inf);
	    root_candidate != root_cell.end(inf); ++ root_candidate)
	{    
	  root_candidate->second->pw_debug_stats.reset(new EPM_Statistics);
	  if(ManifoldCell::deref(*root_candidate) == nullptr)
	    continue;
	  
	  // try to merge each exemplar
	  for(shared_ptr<MetaData> exemplar : 
	      examples.at(root_pose).at(root_name).at(*parts.begin()))
	  {
	    if(root_pose != exemplar->get_pose_name() || !exemplar->use_positives())
	      continue;
	    
	    DetectorResult exemplars_result = 
	      merge_one_exemplar_position(exemplar,
					  ManifoldCell::deref(*root_candidate),
					  part_manifolds,
					  pwResp,root_pose,pairs,
					  *root_candidate->second->pw_debug_stats,im,filter);
	      
	    if(exemplars_result != nullptr)
	    {
	      assert(xIter < joint_manifold->xSize());
	      assert(yIter < joint_manifold->ySize());
	      //cout << "pos: " << xIter << ", " << yIter << endl;
	      ManifoldCell&cell = (*joint_manifold)[xIter][yIter];
	      exemplars_result->exemplar = exemplar;
	      cell.emplace(exemplars_result);  
	    }
	  }
	  stats.add(*root_candidate->second->pw_debug_stats);
	}
      }
    stats.print();
  }
  
  shared_ptr< DetectionManifold > ExemplarPairwiseModel::merge(
    string root_pose, 
    shared_ptr< DetectionManifold > root_manifold, 
    PartMessages& part_manifolds, pw_cost_fn pwResp,
    const ImRGBZ&im,DetectionFilter&filter, Statistics&stats) const
  {
    auto start_pw_time = std::chrono::system_clock::now();
    
    // preparse the manifolds (max pooling)
    shared_ptr<DetectionManifold> joint_manifold;
    joint_manifold = 
      merge_prep_manifold(root_pose,root_manifold,part_manifolds,filter,stats,im);
    if(!joint_manifold)
      return nullptr;
      
    // merge the parts
    if(poses.find(root_pose) != poses.end())
      merge_per_position(joint_manifold,root_pose,root_manifold,part_manifolds,
			 pwResp,filter,stats,im);
      
    auto end_pw_time = std::chrono::system_clock::now();
    std::chrono::milliseconds pw_time = 
      std::chrono::duration_cast<std::chrono::milliseconds>(end_pw_time - start_pw_time);
    stats.count("PairwiseTime",(long)pw_time.count());
    return joint_manifold;    
  }

  void ExemplarPairwiseModel::train_pair(vector< shared_ptr< MetaData > > train_set, 
					 string pose_name, 
					 string root_name, 
					 string part_name)
  {
    unique_lock<mutex> l(monitor);
    // train/load a pairwise model for a pair
    examples[pose_name][root_name][part_name] = train_set;
    poses.insert(pose_name);
    parts.insert(part_name);
    this->root_name = root_name;
  }

  void ExemplarPairwiseModel::cluster_exemplars(
    map<int,std::tuple<shared_ptr<MetaData>,string,string,string> >&metadata_enumerations,
    Mat&data, Mat&data_centers
  )
  {
    DetectionFilter filter;
    
    // run k-means
    int exemplar_limit = fromString<int>(g_params.get_value("EXEMPLAR_LIMIT"));
    // convert the metadata instances to numbers
    data = Mat(0,parts.size()*2,DataType<float>::type,Scalar::all(0));
    vector<vector< float> > vec_data;
    int next_id = 0;
    for(auto&&iter : examples)
      for(auto && jter : iter.second)
	for(auto && kter : jter.second)
	  for(auto lter : kter.second)
	  {
	    metadata_enumerations[next_id++] = 
	      make_tuple(lter,iter.first,jter.first,kter.first);
	    // should be centered with fixed area but aspect ratio from 
	    // exemplar.
	    Rect exemplarRootBB = lter->get_positives()[root_name];
	    double aspect = (double)exemplarRootBB.width/(double)exemplarRootBB.height;
	    Rect_<double> root_ortho_bb = rectFromCenter(
	      Point(0,0),sizeFromAreaAndAspect(1000,aspect));
	    
	    vector<float> exemplar_feat;
	    int rowPos = 0;
	    Mat row_feat(1,parts.size()*2,DataType<float>::type,Scalar::all(0));
	    for(const string&part_name : parts)
	    {
	      double part_x, part_y;
	      compute_part_offset_affine(
		*lter,filter,part_name,root_ortho_bb,part_x,part_y);
	      exemplar_feat.push_back(part_x); row_feat.at<float>(rowPos++) = part_x;
	      exemplar_feat.push_back(part_y); row_feat.at<float>(rowPos++) = part_y;
	    }
	    require_equal<int>(exemplar_feat.size(),data.cols);
	    data.push_back(row_feat);
	    vec_data.push_back(exemplar_feat);
	  }
    const int kmeans_attempts = 10;
    TermCriteria term_crit(cv::TermCriteria::MAX_ITER,1000,qnan);
    Mat labels;
    cout << printfpp("kmeans: %d %d",(int)data.rows,(int)data.cols) << endl;
    cv::kmeans(data,exemplar_limit,labels,term_crit,
		kmeans_attempts,cv::KMEANS_PP_CENTERS,data_centers);    
  }
  
  int ExemplarPairwiseModel::size() const
  {
    int sz = 0;
    for(auto&&iter : examples)
      for(auto && jter : iter.second)
	for(auto && kter : jter.second)
	  for(auto lter : kter.second)
	    sz++;
	  
    return sz;
  }
  
  void ExemplarPairwiseModel::optimize()
  {
    log_file << "original_exemplar_count: " << size() << endl;
    if(g_params.has_key("EXEMPLAR_LIMIT"))
    {
      int exemplar_limit = fromString<int>(g_params.get_value("EXEMPLAR_LIMIT"));
      
      map<int,std::tuple<shared_ptr<MetaData>,string,string,string> > metadata_enumerations;
      Mat data, data_centers;
      cluster_exemplars(metadata_enumerations,data, data_centers);
      
      // keep the cluster medoids
      ExemplarSet newExemplarSet;
      set<string> newPoses, newParts;
      for(int kter = 0; kter < exemplar_limit; kter++)
      {
	// get the cluster center
	Mat center = data_centers.row(kter);
	
	// find the closeset feature
	double min_dist = inf;
	int idx_closest = -1;
	for(int candidate_iter = 0; candidate_iter < data.rows; candidate_iter++)
	{
	  double distance = Mat(data.row(candidate_iter) * center.t()).at<float>(0);
	  if(distance < min_dist)
	  {
	    min_dist = distance;
	    idx_closest = candidate_iter;
	  }
	}
	
	// now we know the medoid
	shared_ptr<MetaData> medoid = get<0>(metadata_enumerations[idx_closest]);
	log_im("Medoid",imageeq("Medoid",medoid->load_im()->Z,false,false));
	newExemplarSet
	  [get<1>(metadata_enumerations[idx_closest])]
	  [get<2>(metadata_enumerations[idx_closest])]
	  [get<3>(metadata_enumerations[idx_closest])].push_back(
	  get<0>(metadata_enumerations[idx_closest]));
	newPoses.insert(get<1>(metadata_enumerations[idx_closest]));
      }
      
      examples = newExemplarSet;
      poses = newPoses;
    }
  }
  
  Mat ExemplarPairwiseModel::vis_model(Mat& background, DetectorResult& det) const
  {
    Mat vis = background.clone();

    // get the src bb and dst bb
    auto poss = det->exemplar->get_positives();
    Rect_<double> src_bb = poss["HandBB"];
    Rect_<double> dst_bb = det->BB;
    
    // derive an affine transform 
    Mat transform = affine_transform(src_bb,dst_bb);
    
    // for each part, compute it's location given by the exemplar
    rectangle(vis,dst_bb.tl(),dst_bb.br(),Scalar(0,0,255));
    for(const string&part_name : det->part_names())
    {
      Rect_<double> part_src = poss[part_name];
      Rect_<double> part_dst = rect_Affine(part_src,transform);
      rectangle(vis,part_dst.tl(),part_dst.br(),Scalar(0,0,255));
    }
    
    return vertCat(image_text("Exemplar"),vis);
  }
  
  Mat ExemplarPairwiseModel::vis_model_offline(Mat& background, DetectorResult& top_det) const
  { 
#ifdef DD_ENABLE_HAND_SYNTH
    unique_lock<decltype(monitor)> l(monitor);
    
    // find the nearest hand params in our regression database
    map<string,Point2d> keypoints;
    vector<Point2d> vec_det = top_det->keypoints();
    PoseRegressionPoint match = libhand_regress(vec_det,
      [&](PoseRegressionPoint&rp){ 
	vector<Point2d> vec_candidate;
	
	write_corners(vec_candidate,top_det->BB);
	for(const string & part_name : top_det->parts_flat())
	  vec_candidate.push_back(rectCenter(rp.parts[part_name]));
	
	return vec_candidate;
      });
    
    // render an image and return it.
    LibHandSynthesizer synther(params::out_dir());
    Size bgSize(params::depth_hRes,params::depth_vRes);
    synther.set_model(match,bgSize);
    shared_ptr<MetaData> match_metadata;
    while(!(match_metadata = synther.synth_one(true)))
      synther.randomize_background(bgSize);
    Mat vis_est_pose = match_metadata->load_im()->RGB;
    
    return vis_est_pose;
#else
    return background.clone();
#endif
  }
  
  void write(FileStorage& fs, string , const ExemplarPairwiseModel& epwm)
  {
    string s;
    
    fs << "{";
    fs << "root_name" << epwm.root_name;
    fs << "poses"; deformable_depth::write(fs,s,epwm.poses);
    fs << "parts"; deformable_depth::write(fs,s,epwm.parts);
    fs << "examples"; deformable_depth::write(fs,epwm.examples);
    fs << "}";
  }

  void read(FileNode fn, ExemplarPairwiseModel& epwm)
  {
    fn["root_name"] >> epwm.root_name;
    deformable_depth::read(fn["poses"],epwm.poses);
    deformable_depth::read(fn["parts"],epwm.parts);
    deformable_depth::read(fn["examples"],epwm.examples);
    epwm.optimize();
  }  
}
