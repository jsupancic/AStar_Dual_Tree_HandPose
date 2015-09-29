/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "GlobalMixture.hpp"
#include "util.hpp"
#include "PXCSupport.hpp"
#include "OneFeatureModel.hpp"
#include "MetaData.hpp"
#include "omp.h"
#include "Log.hpp"
#include "KDTrees.hpp"
#include "vec.hpp"
#include "Semaphore.hpp"
#include "ThreadPool.hpp"

namespace deformable_depth
{ 
  /// SECTION: SUPERVISED
  /// Here we implement the supervised global mixture model.
  SupervisedMixtureModel::SupervisedMixtureModel(string part_name) : 
    part_name(part_name)
  {
  }
  
  DetectionSet SupervisedMixtureModel::detect_mixture(const ImRGBZ& im, DetectionFilter filter, int mixture_id) const
  {
    // grab the model
    auto modelIter = models.begin();   
    for(int jter = 0; jter < mixture_id; ++jter, ++modelIter)
      ;
    string pose_name = modelIter->first;
    
    // skip depending on hint...
    if(filter.pose_hint != "" && filter.pose_hint != pose_name)
    {
      return DetectionSet();
    }
    
    // grab the detections
    DetectionSet detections_for_mixture;
    DetectionFilter mixture_filter(-inf,numeric_limits<int>::max());
    mixture_filter.supress_feature = filter.supress_feature;
    mixture_filter.feat_pyr = filter.feat_pyr;
    mixture_filter.allow_occlusion = filter.allow_occlusion;
    mixture_filter.testing_mode = filter.testing_mode;
    mixture_filter.verbose_log = false;
    mixture_filter.is_root_template = filter.is_root_template;
    DetectionSet model_detections = 
      modelIter->second->detect(
	im,mixture_filter);
    // process the detections
    for(auto && model_detection : model_detections)
    {
      // fill in info not available in the subordinate detectors
      model_detection->pose = pose_name;
      auto raw_feat = model_detection->feature;
      string feat_name = part_name + "_" + pose_name;
      model_detection->feature = [this,feat_name,raw_feat]()
      {
	SparseVector joint_feat(this->joint_interp->total_length);
	SparseVector raw_feat_value = raw_feat();
	this->joint_interp->set(feat_name,raw_feat_value,joint_feat);
	return joint_feat;
      };
      
      // updat resp and filter locally
      if(!filter.require_dot_resp && regressers.size() > 0)
	model_detection->resp = 
	  regressers.at(pose_name)->prob(model_detection->resp);
      if(model_detection->resp >= filter.thresh)
	detections_for_mixture.push_back(model_detection);
    }
    
    return detections_for_mixture;
  }
  
  DetectionSet SupervisedMixtureModel::detect(const ImRGBZ& im, deformable_depth::DetectionFilter filter) const
  {
    // detect by calling into mixtures
    vector<DetectionSet > detections_for_mixture(models.size());
    TaskBlock subdetection("SupervisedMixtureModel::detect");
    for(int iter = 0; iter < models.size(); iter++)
      subdetection.add_callee([this,&im,&filter,iter,&detections_for_mixture]()
      {
	detections_for_mixture[iter] = detect_mixture(im,filter,iter);
      });
    subdetection.execute(*default_pool);
      
    // collect results into one vector. 
    //return nms_supress_identical(detections_for_mixture);
    DetectionSet all_dets;
    for(DetectionSet & cur_dets : detections_for_mixture)
      all_dets.insert(all_dets.end(),cur_dets.begin(),cur_dets.end());
    //vector<DetectionSet > supressed;
    //all_dets = nms(all_dets,.95,&supressed);
    return all_dets;
  }

  Mat SupervisedMixtureModel::show(const string& title)
  {
    Mat agg_vis(1,1,DataType<Vec3b>::type,Scalar::all(0));
    
    for(int iter = 0; iter < models.size(); iter++)
    {
      // grab our model and training set
      auto model_iter = models.begin();    
      for(int jter = 0; jter < iter; ++jter, ++model_iter)
	;      
      
      // get an example image
      Mat RGBExample;
      for(int iter = 0; 
	    training_sets.find(model_iter->first) != training_sets.end() 
	    && iter < training_sets[model_iter->first].size();
	  iter++)
      {
	// load the first matching image
	MetaData&first_match = *training_sets[model_iter->first][iter];
	// extract the pos BB
	auto positives = first_match.get_positives();
	if(positives.find(part_name) != positives.end())
	{
	  Rect bb = positives.at(part_name);
	  if(bb.area() <= 0)
	    continue;
	  
	  RGBExample = first_match.load_im()->RGB;
	  if(bb.x < 0 || bb.y < 0 || 
	    bb.x + bb.width > RGBExample.cols || 
	    bb.y + bb.height > RGBExample.rows)
	    continue;
	  RGBExample = RGBExample(bb);
	  // ensure reasonable resolution.
	  if(RGBExample.size().area() < 640*480)
	    RGBExample = imVGA(RGBExample);
	  break;
	}
      }
      if(RGBExample.empty())
	RGBExample = Mat(1,1,DataType<Vec3b>::type);
      assert(RGBExample.type() == DataType<Vec3b>::type);
      
      // get the model visulaization
      Mat mVis = model_iter->second->show(title+model_iter->first);
      assert(mVis.type() == DataType<Vec3b>::type);
      
      // save the visualization
      mVis = imVGA(mVis);
      Mat vis = horizCat(mVis,RGBExample);
      //log_im("GMM" + model_iter->first, vis);
      
      // append to the aggregate
      assert(agg_vis.type() == DataType<Vec3b>::type);
      assert(vis.type() == DataType<Vec3b>::type);
      agg_vis = vertCat(agg_vis,vis);
    }
    
    return agg_vis;
  }

  void SupervisedMixtureModel::train_collect_sets(
    std::vector< std::shared_ptr< deformable_depth::MetaData > >& train_files)
  {    
    // collect poses and sizes
    for(int iter = 0; iter < train_files.size(); iter++)
    {
      shared_ptr<MetaData> metadata = train_files[iter];
      if(!metadata->use_positives())
	continue;
      string pose = metadata->get_pose_name();
      Rect_<double> partBB = metadata->get_positives()[part_name];
      widths[pose].push_back(partBB.width);
      heights[pose].push_back(partBB.height);
      aspects[pose].push_back(static_cast<float>(partBB.width)/partBB.height);
      
      // figure out the pose set
      if(training_sets.find(metadata->get_pose_name()) != training_sets.end())
      {
	// add example to existing model
	training_sets[metadata->get_pose_name()].push_back(metadata);
      }
      else
      {
	bool inc_pose = true;
	  //metadata.get_pose_name() == "ASL:W";// ||
	  //metadata.get_pose_name() == "ASL:A";
	if(inc_pose)
	{
	  // allocate a new model
	  log_file << "allocated new model for pose = \"" << metadata->get_pose_name() << "\"" << endl;
	  training_sets[metadata->get_pose_name()] = 
	    vector<shared_ptr<MetaData> >(1,metadata);
	}
      }
    }
    
    // now remove all clusters which are to small.
        
    return;
  }
  
  void SupervisedMixtureModel::train_templ_sizes()
  {
    // next, compute the template sizes for each pose
    for(pair<string,vector<float> > && pose_pair : widths)
    {
      // get the pose for this iteration
      string pose = pose_pair.first;
      
      // find the median
      std::sort(widths[pose].begin(),widths[pose].end());
      std::sort(heights[pose].begin(),heights[pose].end());
      float median_width = widths[pose][widths[pose].size()/2];
      float median_height = heights[pose][heights[pose].size()/2]; 
      float median_aspect = aspects[pose][aspects[pose].size()/2];
      float mean_width = accumulate(widths[pose].begin(),widths[pose].end(),0.0f)
	/widths[pose].size();
      float mean_height = accumulate(heights[pose].begin(),heights[pose].end(),0.0f)
	/heights[pose].size();
      
      // store the TSize
      //float width = median_width;
      //float height = median_height;
	
      // base size on median aspect
      float area = median_width*median_height;
      float aspect = median_aspect;
      float width = std::sqrt(area*aspect);
      float height = width/aspect;
      
      // commit the TSize
      TSizes[pose] = Size(width,height);
      log_file << printfpp("part: %s pose %s size = %d x %d",
	part_name.c_str(),pose.c_str(),width,height
      ) << endl;
    }
  }

  void SupervisedMixtureModel::train_create_subordinates(
    vector< shared_ptr< MetaData > >& train_files, 
    Model::TrainParams& train_params)
  {
    // train the models.
    TaskBlock create_subordinates("SupervisedMixtureModel::train_create_subordinates");
    for(int iter = 0; iter < training_sets.size(); iter++)
    {
      create_subordinates.add_callee([&,iter]()
      {
	// get the training set
	auto training_set_iter = training_sets.begin();      
	for(int jter = 0; jter < iter; ++jter, ++training_set_iter)
	  ;
	string pose_name = training_set_iter->first.c_str();
	// print
	#pragma omp critical
	{
	  printf("pose = %s\n",pose_name.c_str());
	  for(shared_ptr<MetaData> & example: training_set_iter->second)
	  {
	    printf("\tpositives from %s\n",example->get_filename().c_str());
	  }
	}
	
	// determine the lead filename
	shared_ptr<ModelType> model;
	vector<shared_ptr<MetaData> >&mixture_training_set = training_set_iter->second;
	shared_ptr<MetaData> metadata = training_set_iter->second[0];
	shared_ptr<const ImRGBZ> im = metadata->load_im();
	Model*newModel;
	if(train_params.subordinate_builder)
	  newModel = train_params.subordinate_builder->build(TSizes[pose_name],im->RGB.size());
	else
	  newModel = OneFeatureModelBuilder().build(TSizes[pose_name],im->RGB.size());
	model.reset(dynamic_cast<ModelType*>(newModel));
	model->prime(mixture_training_set,train_params);
	
	// construct the model
	if(model.get() == nullptr)
	{
	  printf("Failed to find a valid positive for pose %s\n",
		pose_name.c_str());
	  assert(false);
	}
	#pragma omp critical
	models[training_set_iter->first] = model;	
      });
    }
    create_subordinates.execute();
  }
  
  void SupervisedMixtureModel::train_commit_sets(vector< shared_ptr<MetaData> >& train_files,
						       Model::TrainParams&train_params)
  {
    // train the models.
    TaskBlock train_each_model("SupervisedMixtureModel::train_commit_sets");
    for(int iter = 0; iter < training_sets.size(); iter++)
    {
      train_each_model.add_callee([&,iter]()
      {
	// get the training set
	auto training_set_iter = training_sets.begin();      
	for(int jter = 0; jter < iter; ++jter, ++training_set_iter)
	  ;
	string pose_name = training_set_iter->first.c_str();
	
	// retrieve the model
	shared_ptr<ModelType> model = models.at(training_set_iter->first);
	
	// train the model
	Model::TrainParams out_params(1,1);
	out_params.negatives_only = train_files;
	out_params.part_subset = train_params.part_subset;
	out_params.subset_cache_key = out_params.subset_cache_key + training_set_iter->first;
	((Model&)*model).train(training_set_iter->second,out_params);
	
	// now that we've trained the model.
	ostringstream oss;
	oss << "SMM_" << training_set_iter->first;
	log_im(oss.str().c_str(),model->show(oss.str().c_str()));	
      });
    }
    train_each_model.execute();
  }

  string SupervisedMixtureModel::ith_pose(int i) const
  {
    auto model = models.begin();
    for(int jter = 0; jter < i; jter++, model++){};
    return model->first;    
  }
  
  int SupervisedMixtureModel::mixture_id(string mixture_name) const
  {
    int iter = 0;
    for(auto && pose_model : models)
    {
      if(pose_model.first == mixture_name)
	return iter;
      iter++;
    }
    return -1;
  }
  
  void SupervisedMixtureModel::train_logist_platt(Model::TrainParams train_params)
  {
    active_worker_thread.V();
    #pragma omp parallel for 
    for(int iter = 0; iter < models.size(); iter++)
    {
      active_worker_thread.P();
      auto model_iter = models.begin();    
      for(int jter = 0; jter < iter; ++jter, ++model_iter)
	;      
      vector<shared_ptr<MetaData> >&training_set = training_sets[model_iter->first];
      
      Model::TrainParams plat_params;
      plat_params.part_subset = train_params.part_subset;
      plat_params.subset_cache_key = train_params.subset_cache_key + model_iter->first;
      shared_ptr<LogisticPlatt> regressor = train_platt
	(*model_iter->second,training_set,all_train_files,plat_params);
      #pragma omp critical
      regressers[model_iter->first] = regressor;
      active_worker_thread.V();
    }
    active_worker_thread.P();
  }
  
  void SupervisedMixtureModel::prime(vector< shared_ptr< MetaData > >& train_files, Model::TrainParams train_params)
  {
    all_train_files.insert(all_train_files.end(),train_files.begin(),train_files.end());
   
    train_collect_sets(train_files);
    train_templ_sizes();
    printf("Collected training poses... beginning hard negative minning\n");
    train_create_subordinates(train_files,train_params);    
    train_joint_interp();
  }
  
  void SupervisedMixtureModel::train(vector<shared_ptr<MetaData>>& train_files, Model::TrainParams train_params)
  {
    prime(train_files,train_params);
    train_commit_sets(train_files,train_params);
    train_logist_platt(train_params);
  }
  
  map< string, shared_ptr< SupervisedMixtureModel::ModelType > >&SupervisedMixtureModel::get_models()
  {
    return models;
  }

  map< string, vector< shared_ptr< MetaData > > >& SupervisedMixtureModel::get_training_sets()
  {
    return training_sets;
  }
  
  void SupervisedMixtureModel::train_joint_interp()
  {
    joint_interp.reset(new FeatureInterpretation());
    for(auto && elem_pair : models)
      joint_interp->init_include_append_model(
	*elem_pair.second,part_name + "_" + elem_pair.first);
    vector<double> w0(joint_interp->total_length,0);
    learner.reset(new FauxLearner(w0,0.0));
  }
  
  SparseVector SupervisedMixtureModel::extractPos(MetaData& metadata, AnnotationBoundingBox bb) const
  {
    assert(joint_interp->size() != 0);
    auto pos = metadata.get_positives();

    SparseVector joint_feature(joint_interp->total_length);
    auto&mixture_model = models.at(metadata.get_pose_name());
    SparseVector element_feature = mixture_model->extractPos(metadata,pos[part_name]);
    if(element_feature.size() == 0)
      return vector<float>();
    joint_feature.set(joint_interp->at(part_name + "_" + metadata.get_pose_name())[0],
		      element_feature);
    return joint_feature;
  }
  
  LDA& SupervisedMixtureModel::getLearner()
  {
    assert(learner);
    return *learner;
  }
  
  void SupervisedMixtureModel::setLearner(LDA* lda)
  {
    learner.reset(lda);
  }

  void SupervisedMixtureModel::update_model()
  {
    vector<double> w = learner->getW();
    for(auto&&element_model : models)
    {
      vector<double> elem_wf = joint_interp->select(part_name + "_" + element_model.first,w);
      element_model.second->setLearner(new FauxLearner(elem_wf,0.0));
      element_model.second->update_model();
    }
  }
  
  Mat SupervisedMixtureModel::vis_result(const ImRGBZ&im,Mat& background, DetectionSet& dets) const
  {
    if(dets.size() == 0)
      return background.clone();
    Detection&det = *dets[0];
    
    if(models.find(det.pose) == models.end())
    {
      Mat vis = background.clone();
      Point tl = det.BB.tl();
      Point br = det.BB.br();
      rectangle(vis,tl,br,Scalar(0,255,255)); 
      return vis;
    }
    else
    {
      vector<double> f = vec_f2d(det.feature());
      
      // select feature
      vector<double> mixt_feat = joint_interp->select(part_name + "_" + det.pose,f);
      // alloc a new object to store it
      DetectorResult mixt_det(new Detection);
      *mixt_det = det;
      mixt_det->feature = [&](){return SparseVector(mixt_feat);};
      DetectionSet mixt_dets; 
      mixt_dets.push_back(mixt_det);   
      
      return models.at(det.pose)->vis_result(im,background,mixt_dets);
    }
  }
  
  /// SECTION: SupervisedMixtureModel_Builder
  Model* SupervisedMixtureModel_Builder::build(Size gtSize,Size imSize) const
  {
    return new SupervisedMixtureModel();
  }

  string SupervisedMixtureModel_Builder::name() const
  {
    return "SupervisedMixtureModel_Builder";
  }
  
  /// SECTION: Serialization
  void write(FileStorage& fs, string& , const SupervisedMixtureModel& sgmm)
  {
    fs << "{";
    fs << "learner" << FauxLearner(sgmm.learner->getW(),sgmm.learner->getB());;
    fs << "joint_interp" << sgmm.joint_interp;
    fs << "part_name" << sgmm.part_name;
    fs << "regressers"; write(fs,sgmm.regressers);
    fs << "models"; write(fs,sgmm.models);    
    fs << "}";
  }
  
  void write(FileStorage& fs, string& str, const shared_ptr< SupervisedMixtureModel >& sgmm)
  {
    write(fs,str,*sgmm);
  }
  
  void read(const FileNode& node, shared_ptr< SupervisedMixtureModel >& sgmm, 
	    shared_ptr< SupervisedMixtureModel > )
  {
    sgmm.reset(new SupervisedMixtureModel());
    // the learner is bit tricky
    shared_ptr<FauxLearner> lda;
    node["learner"] >> lda;
    sgmm->learner = lda;
    //
    node["joint_interp"] >> sgmm->joint_interp;
    node["part_name"] >> sgmm->part_name;
    read(node["regressers"],sgmm->regressers);
    read(node["models"],sgmm->models);
  }
}
