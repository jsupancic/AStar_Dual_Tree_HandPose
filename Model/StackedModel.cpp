/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "StackedModel.hpp"

namespace deformable_depth
{
  /// 
  /// SECTION: Pedestrian Model
  /// 
  DetectionSet StackedModel::detect_dpm2occ(const ImRGBZ& im, DetectionFilter filter) const
  {
    DetectionFilter part_fitler = filter;
    part_fitler.nmax = numeric_limits<int>::max();
    part_fitler.thresh = -inf;
    DetectionSet occ_dets = occ_model->detect(im,part_fitler);
    DetectionSet dpm_dets = dpm_model->detect(im,part_fitler);
    
    DetectionSet merged_dets;
    for(int iter = 0; iter < dpm_dets.size(); ++iter)
    {
      DetectorResult dpm_det = dpm_dets[iter];
      DetectorResult new_det;
      //DetectorResult occ_det = nearest_neighbour(occ_dets,dpm_det->BB);
      for(DetectorResult occ_det : occ_dets)
      {
	double sf = std::sqrt(occ_det->BB.area()/dpm_det->BB.area());
	if(
	  occ_det->real >= .5 &&
	  rectIntersect(occ_det->BB,dpm_det->BB) > BaselineModel_ExternalKITTI::BB_OL_THRESH &&
	  clamp<double>(.5,sf,2) == sf)
	{
	  // update the resp
	  double new_resp = dpm_det->resp; //dpm_platt->prob();// * occ_platt->prob(occ_det->resp);
	  if(!new_det or new_det->resp < new_resp)
	  {
	    new_det.reset(new Detection(*dpm_det));
	    new_det->resp = new_resp;
	  }
	}
	if(new_det)
	  merged_dets.push_back(new_det);
      }
    }
    
    log_once(printfpp("merged %d of %d dets",(int)merged_dets.size(),(int)dpm_dets.size()));
    //filter.apply(merged_dets);
    log_once(printfpp("filt 2 %d of %d dets",(int)merged_dets.size(),(int)dpm_dets.size()));
    return merged_dets;
  }

  DetectionSet StackedModel::detect_occ2dpm(const ImRGBZ& im, DetectionFilter filter) const
  {
    DetectionFilter part_fitler = filter;
    part_fitler.nmax = numeric_limits<int>::max();
    part_fitler.thresh = -inf;
    DetectionSet occ_dets = occ_model->detect(im,part_fitler);
    DetectionSet dpm_dets = dpm_model->detect(im,part_fitler);
    
    DetectionSet merged_dets;
    for(int iter = 0; iter < occ_dets.size(); ++iter)
    {
      DetectorResult occ_det = occ_dets[iter];
      auto occFeatFn = occ_det->feature;
      DetectorResult dpm_det = nearest_neighbour(dpm_dets,occ_det->BB);
      decltype(occFeatFn) dpmFeatFn;
      if(rectIntersect(occ_det->BB,dpm_det->BB) > BaselineModel_ExternalKITTI::BB_OL_THRESH)
      {
	occ_det->resp = dpm_det->resp + learner->getB();
	//occ_det->resp = ;//dpm_platt->prob(dpm_det->resp) * occ_platt->prob(occ_det->resp);
	dpmFeatFn = dpm_det->feature;
      }
      else
      {
	occ_det->resp = -inf; //dot(dpm_model->getW(),BaselineModel_ExternalKITTI::minFeat()) + learner->getB();
	dpmFeatFn = [](){return vector<double>{BaselineModel_ExternalKITTI::minFeat()};};	
      }
      
      occ_det->feature = [this,occFeatFn,dpmFeatFn]()
      {
	map<string,SparseVector> feats{
	  {"dpm_model",dpmFeatFn()},
	  {"occ_model",occFeatFn()}};
	
	return feat_interpreation->pack(feats);
      };    
      
      merged_dets.push_back(occ_det);
    }
    
    log_once(printfpp("merged %d of %d dets",(int)merged_dets.size(),(int)occ_dets.size()));
    //filter.apply(merged_dets);
    return merged_dets;
  }
  
  DetectionSet StackedModel::detect_dpmThenOcc(const ImRGBZ& im, DetectionFilter filter) const
  {
    // get the detections from the DPM
    DetectionFilter part_fitler = filter;
    part_fitler.nmax = numeric_limits<int>::max();
    part_fitler.thresh = -inf;
    DetectionSet dpm_dets = dpm_model->detect(im,part_fitler);  
    DetectionSet dets;
    
    float min_world_area, max_world_area;
    area_model->validRange(params::world_area_variance,true,min_world_area,max_world_area);
    
    for(auto & dpm_det : dpm_dets)
    {
      AnnotationBoundingBox abb;
      abb.write(dpm_det->BB);
      
      // consider these depths...
      bool could_be_real = false;
      vector<float> depths = manifoldFn_prng(im,abb,50);
      for(float depth : depths)
      {
	// check the box's depth
	double bb_world_area = im.camera.worldAreaForImageArea(depth,abb);
	if(bb_world_area < min_world_area || bb_world_area > max_world_area)
	  continue;
	
	// check that the box is real
	abb.depth = depth;
	auto info = occ_model->extract(im, abb,false);
	//cout << printfpp("%real = %f\n",real);
	if(info.occ <= .80 && info.bg <= .5)
	  could_be_real = true;
      }
      
      if(could_be_real)
	dets.push_back(dpm_det);
    }
    
    log_once(printfpp("areaModel filtered %d => %d",(int)dpm_dets.size(),(int)dets.size()));
    return dets;
  }
  
  DetectionSet StackedModel::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    //return detect_occ2dpm(im,filter);
    //return detect_dpm2occ(im,filter);
    return detect_dpmThenOcc(im,filter);
  }

  StackedModel::StackedModel(Size TSize, Size ISize) : 
    ISize(ISize), TSize(TSize)
  {
  }

  Mat StackedModel::show(const string& title)
  {
    return dynamic_cast<Model*>(&*occ_model)->show(title);
  }

  void StackedModel::train(
    vector< shared_ptr< MetaData > >& training_set, 
    Model::TrainParams train_params)
  {
    // setup the dpm model
    dpm_model.reset(new BaselineModel_ExternalKITTI);
    
    // prime the occ model
    train_params.subset_cache_key += "occ_model";
    occ_model.reset(new OccAwareLinearModel(TSize,ISize));
    //occ_model->prime(training_set,train_params);
    //occ_model->train(training_set,train_params);
    
    // setup the area model
    area_model.reset(new AreaModel);
    area_model->train(training_set,train_params);
    
    // setup the plat system
    //train_params.subset_cache_key = "occ_platt";
    //occ_platt = train_platt(*occ_model,training_set,training_set,train_params);
    //train_params.subset_cache_key = "dpm_platt";
    //dpm_platt = train_platt(*dpm_model,training_set,training_set,train_params);
    
    // setup the feature interpretation
    feat_interpreation.reset(new FeatureInterpretation());
    feat_interpreation->init_include_append_model(
      *dpm_model,"dpm_model");
    feat_interpreation->init_include_append_model(
      *occ_model,"occ_model");    
    
    // init the QP
    learner.reset(new QP(params::C()));
    learner->prime(feat_interpreation->total_length);
    
    // train via parallel hard negative mining
    //train_smart(*this,training_set,train_params);    
  }
  
  SparseVector StackedModel::extractPos(MetaData& metadata, AnnotationBoundingBox bb) const
  {
    map<string,SparseVector> feats{
      {"dpm_model",dynamic_cast<ExposedLDAModel*>(dpm_model.get())->extractPos(metadata,bb)},
      {"occ_model",dynamic_cast<ExposedLDAModel*>(occ_model.get())->extractPos(metadata,bb)}};
    
    return feat_interpreation->pack(feats);
  }

  void StackedModel::update_model()
  {
    learner->opt();
    
    map<string,SettableLDAModel&> models{
      {"dpm_model",*dpm_model},
      {"occ_model",*occ_model}};    
    
    feat_interpreation->flush(learner->getW(),models);
  }
  
  LDA& StackedModel::getLearner()
  {
    return *learner;
  }
  
  bool StackedModel::write(FileStorage& fs)
  {
    fs << "{";
    fs << "occ_model" << *occ_model;
    fs << "}";
    return true;
  }
  
  ///
  /// SECTION: Pedestrian Model Builder
  ///
  Model* StackedModel_Builder::build(Size gtSize, Size imSize) const
  {
    return new StackedModel(gtSize,imSize);
  }

  string StackedModel_Builder::name() const
  {
    return "PedestrianModel_Builder";
  }

  StackedModel_Builder::StackedModel_Builder(double C, double area, shared_ptr< IHOGComputer_Factory > fc_factory, double minArea, double world_area_variance)
  {
  }
}

