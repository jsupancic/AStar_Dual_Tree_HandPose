/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "StarModel.hpp"
#include "Log.hpp"
#include "Semaphore.hpp"
#include "ThreadPool.hpp"
#include "BackgroundThreads.hpp"
#include "OcclusionReasoning.hpp"
#include "PairwiseExemplar.hpp"

namespace deformable_depth
{  
  /** SECTION: HandModel **/
  StarModel::StarModel() : 
    partFilter(-inf,numeric_limits<int>::max()),
    update_count(0)
  {
    partFilter.require_dot_resp = true;
    // set if we are extracting features...
    partFilter.supress_feature = false;
  }
  
  void StarModel::debug_incorrect_resp(SparseVector& feat, Detection&det)
  {
    vector<double> w = learner->getW();
    
    // print the feature interprestation
    log_file << "faet_interp = " << feat_interpreation->toString() << endl;
    log_file << "debug_incorrect_resp: " << det.src_filename << endl;
    log_file << "debug_incorrect_resp: det depth = " << det.depth << endl;
    
    for(int strip_id = 0; strip_id < feat.strip_count(); strip_id++)
    {
      // get a little info about the strip
      vector<float> strip = feat.get(strip_id);
      int strip_position = feat.strip_start(strip_id);
      string interp = feat_interpreation->interp_at_pos(strip_position);
      if((*feat_interpreation)[interp][1] != strip.size())
      {
	log_file << "stripId = " << strip_id << 
	  " interp = " << interp << 
	  " strip_position = " << strip_position << endl;
	log_file << printfpp("interp.size() = %d strip.size() = %d",
			     (int)(*feat_interpreation)[interp][1],(int)strip.size()) << endl;
      }
      assert((*feat_interpreation)[interp][1] == strip.size());
      
      // compute the resp = w'x for this part.
      vector<double> w = feat_interpreation->select(interp,w);
      double strip_resp = dot<double>(w,strip);
      
      // print what we know
      log_file << printfpp("interp = %s w(interp)'*feat(interp) = %f",
			   interp.begin(),strip_resp) << endl;
      
    }
  }
  
  DetectionSet StarModel::detect(const ImRGBZ& im, DetectionFilter filter) const
  {    
    // run a detection algorithm
    //DetectionSet detections = detect_flann(im,filter);
    Statistics stats;
    DetectionSet detections = detect_manifold(im,filter,stats);
    if(filter.testing_mode)
      assert(detections.part_candidates.size() > 0);
    
    // DEBUG mode
    static const bool DEBUG_FEAT_RESP = false;
    
    // filter the detections as needed
    if(!DEBUG_FEAT_RESP)
      filter.apply(detections);
    if(!filter.testing_mode)
      log_once("WARNING: NO DETECTIONS!");
    
    // extract the features
    if(!filter.supress_feature)
      for(auto && detection : detections)
      {
	auto root_feature = detection->feature;
	map<string,function<vector<float> ()> > part_features;
	for(string part_name : detection->part_names())
	  part_features[part_name] = detection->getPart(part_name).feature;
	string pose = detection->pose;
	detection->feature = [this,pose,root_feature,part_features]() -> SparseVector 
	{
	  SparseVector joint_feature(this->feat_interpreation->total_length);
	  // set the root feature
	  SparseVector root_feat = root_feature();
	  this->feat_interpreation->set("root",root_feat,joint_feature);
	  // set the part features
	  for(auto && part_faeture : part_features)
	  {
	    SparseVector part_feat = part_faeture.second();
	    this->feat_interpreation->set("part_" + part_faeture.first,
				    part_feat,joint_feature);	  
	  }
	  return joint_feature;
	};
      }    
    
    // DEBUG: check the w'*feature() == resp correectness before supression
    if(DEBUG_FEAT_RESP)
      for(auto && detection : detections)
      {
	SparseVector sparse_feat = detection->feature();
	double model_score = learner->predict(sparse_feat);
	bool good_scores = (goodNumber(model_score) && goodNumber(detection->resp));
	if(!good_scores || ::abs(model_score-detection->resp)>1e-4) 
	{
	  cout << printfpp("debug error: model_score(%f) != detector_score(%f)",
		  model_score,detection->resp) << endl;
	  assert(false);
	}
      }    
      
    // print statistics
    sort(detections);
    if(!filter.testing_mode)
      log_once("WARNING: NO DETECTIONS!");
    else
    {
      string message =printfpp("Time = %d Exemplars = %d MaxResp = %f",
		      (int)stats.getCounter("PairwiseTime"),
		      (int)def_model_exemplar.size(),
		      detections.size() > 0?(double)detections[0]->resp:qnan
		      );
      cout << message << endl;
      log_file << message << endl;
    }
    
    return detections;
  }

  pw_cost_fn combine_wgm = [](const Detection&hand_det,const Detection&finger_det)
  {
    float prob = weighted_geometric_mean({hand_det.resp,finger_det.resp},{.9,.1});
    return ::isnan(prob)?-inf:prob;
  };
  
  pw_cost_fn combine_linear = [](const Detection&hand_det,const Detection&finger_det)
  {
    return hand_det.resp + finger_det.resp;
  };
  

  
  pw_cost_fn combine_w_occlusion_reasoning = [](const Detection&hand_det,const Detection&finger_det)
  {
    // DEBUG alteration
    //return hand_det.real + finger_det.resp;
    
    // reason about occlusion
    // by rejecting any pairing with conflicting occlusion states.
    bool overlaps_visible = false;
    for(string part_name : hand_det.part_names())
    {
      // two conditions should be met.
      bool parts_overlap = occlusion_intersection(hand_det.getPart(part_name).BB,finger_det.BB);
      bool other_visible = is_visible(hand_det.getPart(part_name).occlusion);
      if(parts_overlap && other_visible)
	// our part overlaps a visible object and should be occluded?
	overlaps_visible = true;
    }
      
    bool part_is_visible = is_visible(finger_det.occlusion);
    
    if(valid_configuration(part_is_visible,overlaps_visible))
      return hand_det.resp + finger_det.resp;
    else
      return -inf;
  };
  
  shared_ptr< DetectionManifold > StarModel::detect_gbl_mixture(
    const ImRGBZ& im, 
    string pose, shared_ptr<FeatPyr> rootPyr,
    PartMessages& part_manifolds,
    DetectionFilter&filter,
    DetectionSet&joint_detections,
    Statistics&stats) const
  {
    // detect the root candidates
    DetectionFilter live_filter = partFilter;
    live_filter.feat_pyr = rootPyr;
    live_filter.allow_occlusion = false;
    live_filter.testing_mode = filter.testing_mode;
    live_filter.is_root_template = filter.is_root_template;
    DetectionSet glbDetsPerPose = 
      root_mixture->detect_mixture(im,live_filter,root_mixture->mixture_id(pose));
    log_file << printfpp("Detected %d root templates",glbDetsPerPose.size()) << endl;
    for(auto && detection : glbDetsPerPose)
      detection->pose = pose;
    
    // 
    DetectionSet visible_glbDetsPerPose;
    for(auto && detection : glbDetsPerPose)
      if(detection->real > 0)
	visible_glbDetsPerPose.push_back(detection);
    
    // create the root manifold
    shared_ptr< DetectionManifold > root_manifold = 
      create_manifold(im,visible_glbDetsPerPose);
  
    // DEBUG store all of the original detections and use them to store rejction
    // reasoning...
    if(filter.testing_mode)
    {
      // TODO: BUG: should merge not take the most recent mixture's root bbs
      static mutex m; unique_lock<mutex> l(m);
      auto & handBBcandidates = joint_detections.part_candidates["HandBB"];
      handBBcandidates.insert(handBBcandidates.end(),
			      visible_glbDetsPerPose.begin(),visible_glbDetsPerPose.end());
    }
    
    // join them with the appropriate pairwise model
    shared_ptr<DetectionManifold> joint_manifold = 
      def_model_exemplar.merge(pose,root_manifold,part_manifolds,
			      combine_linear,im,filter,stats);
    // combine_w_occlusion_reasoning OR combine_linear
      
    // return the result;
    return joint_manifold;
  }
  
  void StarModel::extract_detections_from_manifold
    (const ImRGBZ& im, 
     DetectionFilter&filter,
     DetectionSet&joint_detections,
      map<string,shared_ptr<DetectionManifold>>&global_manifolds
    ) const
  {
    // convert manifold to image
    #pragma omp critical(MANIFOLD_LOG)
    if(filter.verbose_log)
    {
      // serial merge with zero displacement allowed.
      DetectionSet nodets;
      shared_ptr<DetectionManifold> full_joint_manifold = create_manifold(im, nodets); 
      for(auto &&iter = global_manifolds.begin(); 
	  iter != global_manifolds.end(); 
	  iter++)
	  if(iter->second)
	    full_joint_manifold->concat(*iter->second);      
      
      // show the joint manifold
      Mat joint_vis = draw_manifold(*full_joint_manifold);
      log_im("manifolds",joint_vis);
    }
    
    // extract the detections from the manifold.
    for(auto&&gm_manifold : global_manifolds)
      if(gm_manifold.second)
	for(int yIter = 0; yIter < gm_manifold.second->ySize(); yIter++)
	  for(int xIter = 0; xIter < gm_manifold.second->xSize(); xIter++)
	  {
	    auto&gm_cell = (*gm_manifold.second)[xIter][yIter];
	    for(auto && curDetection = gm_cell.begin(-inf);
		curDetection != gm_cell.end(inf); ++curDetection)
	    {
	      if(ManifoldCell::deref(*curDetection) == nullptr)
		continue;
	      ManifoldCell::deref(*curDetection)->resp += learner->getB();
	      // at the first step, check that the filter matches
	      if(ManifoldCell::deref(*curDetection)->resp > filter.thresh)
	      {
		joint_detections.push_back(ManifoldCell::deref(*curDetection));
	      }
	    }
	  }    
  }
  
  bool StarModel::part_is_root_concreate(string part_name) const
  {
    return false;
  }
  
  DetectionSet StarModel::detect_manifold(const ImRGBZ& im, DetectionFilter filter, Statistics&stats) const
  {    
    // the output we are building
    DetectionSet joint_detections;
    
    // detect the parts
    TaskBlock part_detection_tasks("StarModel::detect_manifold: parts");
    DetectionFilter live_part_filter = partFilter;
    live_part_filter.feat_pyr = shared_ptr<FeatPyr>(new FeatPyr(im));
    live_part_filter.testing_mode = filter.testing_mode;
    live_part_filter.is_root_template = false;
    mutex manifold_guard;
    PartMessages part_manifolds;
    for(auto&&part_pair : parts)
    {
      auto  &part_name = part_pair.first;
      auto  &model     = part_pair.second;
      part_detection_tasks.add_callee([&,part_name,model]()
      {
	DetectionFilter active_part_filter = live_part_filter;
	active_part_filter.is_root_template = part_is_root_concreate(part_name);
	DetectionSet detections = model->detect(im,active_part_filter);
	log_file << printfpp("Detected %d parts",(int)detections.size()) << endl;
	map<string/*pose*/,DetectionManifold> manifold = create_sort_manifolds(im,detections);
	
	// CRITICAL SECTION
	unique_lock<mutex> exclusion(manifold_guard);
	part_manifolds.insert(
	  pair<string/*part name*/,map<string/*pose*/,DetectionManifold>>(
	    part_name,std::move(manifold)));
	if(g_params.has_key("DEBUG_PAIRWISE_BIG_MEMORY") && filter.testing_mode)
	  joint_detections.part_candidates[part_name] = detections;
      });
    }
    part_detection_tasks.execute(*default_pool);
    if(g_params.has_key("DEBUG_PAIRWISE_BIG_MEMORY") && filter.testing_mode)
      assert(joint_detections.part_candidates.size() > 0);
    
    // detect the root templates
    // in parallel, detect for each mixture
    shared_ptr<FeatPyr> rootPyr(new FeatPyr(im));
    TaskBlock root_detection_tasks("StarModel::detect_manifold: roots");
    map<string,shared_ptr<DetectionManifold>> global_manifolds;
    filter.is_root_template = true;
    for(int iter = 0; iter < root_mixture->get_models().size(); iter++)
      root_detection_tasks.add_callee([&,iter]()
      {
	// get the pose name
	string pose_name = root_mixture->ith_pose(iter);
	
	shared_ptr< DetectionManifold > pose_manifold = 
	  detect_gbl_mixture(im,pose_name,rootPyr,
			     part_manifolds,filter,
			      joint_detections,stats);
	static mutex m; lock_guard<mutex> l(m);
	global_manifolds[pose_name] = pose_manifold;	
      });
    root_detection_tasks.execute(*default_pool);       
    
    extract_detections_from_manifold(im, filter,joint_detections,global_manifolds);
      
    return joint_detections;
  }
  
  void StarModel::save()
  {
    // don't allow changes while saving
    boost::shared_lock<boost::shared_mutex> read_monitor(monitor);    
    
    string hostname = get_hostname();
    string filename = printfpp("cache/StarModel%s:%s:%d.yml.gz",
			       hostname.c_str(),getRootPartName().c_str(),
			       update_count);
    FileStorage cache(filename,FileStorage::WRITE);
    cache << "StarModel" << *this;
    cache.release();
  }
  
  Mat StarModel::show(const string& title)
  {    
    // don't allow changes while showing
    boost::shared_lock<boost::shared_mutex> read_monitor(monitor);
    
    // show the model
    // show the root templates
    Mat vis = root_mixture->show(title);
    // show the part templates
    //for(auto&&part_pair : parts)
      //vis = vertCat(part_pair.second->show(title),vis);
    image_safe((string("HandModel") + title).c_str(),vis);
    return vis; 
  }
  
  void StarModel::train_init_joint_interpretation()
  {
    feat_interpreation.reset(new FeatureInterpretation());
    
    // collect the feature sizes for the root_mixture
    feat_interpreation->init_include_append_model(
      *root_mixture,"root");
    
    // collect the feature sizes for the finger tip.
    for(auto&&part_pair : parts)
    {
      feat_interpreation->init_include_append_model
	(*part_pair.second,"part_" + part_pair.first);
    }
  }
  
  void StarModel::train_pairwise()
  {
    TaskBlock train_pairwise("StarModel::train_pairwise");
    for(int iter = 0; iter < root_mixture->get_models().size(); iter++)
    {
      train_pairwise.add_callee([&,iter]()
      {
	string pose_name = root_mixture->ith_pose(iter);
	for(auto&&part_pair : parts)
	{
	  def_model_exemplar.train_pair(
	    root_mixture->get_training_sets().at(pose_name),
	    pose_name,getRootPartName(),part_pair.first);
	}	
      });
    }
    train_pairwise.execute();
    
    def_model_exemplar.optimize();
  }
  
  void StarModel::train_joint(vector< shared_ptr< MetaData > >& training_set, 
				       Model::TrainParams train_params)
  {
    // setup the learner
    learner.reset(new QP(getC()));
    learner->prime(feat_interpreation->total_length);
    learner->getW(); // verify this works.
    
    // train via parallel hard negative mining
    Model::TrainParams out_params;
    out_params.subset_cache_key = "StarAbout:" + getRootPartName();
    string root_part_name = getRootPartName();
    out_params.part_subset = [root_part_name](const map<string,AnnotationBoundingBox>&all_pos)
      {
	map<string,AnnotationBoundingBox> subset;
	Rect_<double> part = all_pos.at(root_part_name);
	subset[root_part_name].write(part);
	return subset;	
      };
    train_smart(*this,training_set,out_params);
    //show("StarFinal");
  }
  
  SparseVector StarModel::extractPos(MetaData&metadata, AnnotationBoundingBox bb) const
  { 
    auto pos = metadata.get_positives();
    
    SparseVector feature(feat_interpreation->total_length);
    //
    // ROOT: extract the root feature for the correct mixture.
    //
    auto root_feature = root_mixture->extractPos(metadata,pos[getRootPartName()]);
    if(root_feature.size() == 0)
    {
      log_once(printfpp("Problem with root feature in: %s",metadata.get_filename().c_str()));
      return vector<float>();
    }
    feature.set(feat_interpreation->at("root")[0],root_feature);
    
    //
    // PARTS: extract features for each of the parts,
    // 	assume visible. Consider occluded IFF an occluder is found.
    //
    vector<AnnotationBoundingBox > part_bbs;
    for(auto&&part_pair : parts)
    {
      // find wher we have occlusion conflicts
      const string&pose_name = part_pair.first;
      AnnotationBoundingBox part_bb = pos[pose_name];
      part_bb.visible = true; // initially, consider all visible.
      for(AnnotationBoundingBox & other_bb : part_bbs)
      {
	bool overlap = rectIntersect(other_bb,part_bb) > .5;
	bool occ_conflict = other_bb.visible > .5 && part_bb.visible > .5;
	if(overlap && occ_conflict)
	  part_bb.visible = false;
      }
      part_bbs.push_back(part_bb);
      
      // extract the part feature and concat to the whole feature
      auto&part_model = part_pair.second;
      auto part_feature = part_model->extractPos(metadata,part_bb);
      if(part_feature.size() == 0)
      {
	log_once(printfpp("Problem with part %s in %s",
			  pose_name.c_str(),metadata.get_filename().c_str()));
	return vector<float>();
      }
      feature.set(
	feat_interpreation->at("part_" + pose_name)[0],
	part_feature);
    }
    
    return feature;
  }

  double StarModel::min_area() const
  {
      double minArea = numeric_limits<double>::infinity();
      
      for(auto&&saved_detector : root_mixture->get_models())
	minArea = std::min<double>(minArea,saved_detector.second->min_area());
      
      return minArea;
  }
  
  void StarModel::update_model()
  {
    boost::unique_lock<boost::shared_mutex> write_monitor(monitor);
    
    // optmize the learner
    update_count++;
    assert(learner);
    cout << 
      printfpp("StarModel::update_model: w.size = %d",
	       learner->getFeatureLength()) << endl;
    learner->opt();
    vector<double> w = learner->getW();
    
    // update the root_mixture
    vector<double> mixt_wf = feat_interpreation->select("root",w);
    root_mixture->setLearner(new FauxLearner(mixt_wf,0.0));
    root_mixture->update_model();
    
    // update the finger model
    for(auto&&part_pair : parts)
    {
      vector<double> part_wf = feat_interpreation->select("part_" + part_pair.first,w);
      part_pair.second->setLearner(new FauxLearner(part_wf,0.0f));
      part_pair.second->update_model();
    }
    
    // show
    std::thread([this]()
    {
      save();
    }).detach();
  }

  LDA& StarModel::getLearner()
  {
    return *learner;
  }
  
  // pre-training check... make sure all positives are extractable
  void StarModel::check_pretraining(vector< shared_ptr< MetaData > >& training_set)
  {
    if(g_params.has_key("LOAD_MODEL"))
    {
      log_file << "Skip check_pretraining" << endl;
      return;
    }
    
    log_file << "+StarModel::check_pretraining" << endl;
    TaskBlock task("check_pretraining");
    for(int iter = 0; iter < training_set.size(); iter++)
    {
      task.add_callee([&,iter]()
      {
	assert(training_set[iter]);
	MetaData&metadata = *training_set[iter];
	if(!metadata.use_positives())
	  return;
	extractPos(metadata,metadata.get_positives()[getRootPartName()]);	
      });
    }
    task.execute();
    log_file << "-StarModel::check_pretraining" << endl;
  }
  
  void StarModel::train(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
    log_file << "+StarModel" << endl;
    prime_root(training_set,Model::TrainParams());
    log_file << "StarModel: primed root" << endl;
    prime_parts(training_set,Model::TrainParams());
    log_file << "StarModel: init'ing the joint interpreation" << endl;
    train_init_joint_interpretation();
    log_file << "StarModel: pretraining checks" << endl;
    check_pretraining(training_set);
    log_file << "StarModel: training pairwise" << endl;
    train_pairwise();
    log_file << "StarModel: Training Jointly" << endl;
    train_joint(training_set,Model::TrainParams());
    log_file << "-StarModel" << endl;
  }  
  
  Mat StarModel::vis_feats(const ImRGBZ&im,Mat& background, DetectionSet& dets) const
  {
    if(dets.size() == 0)
      return background.clone();
    Detection&det = *dets[0];
    
    // initialzie
    Mat vis = background.clone();
    vector<double> f = vec_f2d(det.feature());
    
    // draw the root
    // select root feature
    vector<double> glbl_feat = feat_interpreation->select("root",f);
    // alloc a new object to store it
    DetectorResult root_det(new Detection);
    *root_det = det;
    root_det->feature = [&](){return SparseVector(glbl_feat);};
    DetectionSet root_dets; 
    root_dets.push_back(root_det);   
    // draw it
    vis = root_mixture->vis_result(im,background,root_dets);    
    
    // draw each part
    for(string part_name : det.part_names())
    {
      // extract part feature
      vector<double> part_wf = feat_interpreation->select("part_" + part_name,f);
      // alloc
      DetectorResult sub_det(new Detection);
      *sub_det = det.getPart(part_name);
      sub_det->feature = [&](){return SparseVector(part_wf);};
      DetectionSet sub_dets; 
      sub_dets.push_back(sub_det);
      // show
      vis = parts.at(part_name)->vis_result(im,vis,sub_dets);
    }
    
    return vertCat(image_text("vis_feats"),vis);
  }
  
  Mat StarModel::vis_part_positions(Mat& background, DetectionSet& dets) const
  {
    if(dets.size() == 0)
      return background.clone();
    
    DetectorResult top_det = dets[0]; 
    Mat vis = def_model_exemplar.vis_model(background,top_det);
    
    //if(top_det->lr_flips % 2 == 1)
      //flip(vis,vis,1);
    
    return vis;
  }
  
  Mat StarModel::vis_result(const ImRGBZ&im,Mat& background, DetectionSet& dets) const
  {
    return horizCat(vis_feats(im,background,dets),vis_part_positions(background,dets));
  }
  
  /// SECTION: Serialization  
  void write(FileStorage& fs, const string& , const StarModel& star_model)
  {    
    fs << "{";
    fs << "parts"; write(fs,star_model.parts);
    fs << "root" << *star_model.root_mixture;
    fs << "wb" << FauxLearner(star_model.learner->getW(),star_model.learner->getB());
    fs << "feat_interpreation" << star_model.feat_interpreation;
    fs << "def_model_exemplar"; write(fs,star_model.def_model_exemplar);
    fs << "}";
  }
  
  bool StarModel::write(FileStorage& fs)
  {
    string s;
    deformable_depth::write(fs,s,(const StarModel&)*this);
    return true;
  }  
  
  void read(const FileNode& node, StarModel& model)
  {
    assert(!node.isNone());
    read(node["parts"],model.parts);
    node["root"] >> model.root_mixture;
    shared_ptr<FauxLearner> lda;
    node["wb"] >> lda;
    model.learner = lda;
    node["feat_interpreation"] >> model.feat_interpreation;
    read(node["def_model_exemplar"],model.def_model_exemplar);
    
    // flush the new parameters to the model
    model.update_model();
  }
}
