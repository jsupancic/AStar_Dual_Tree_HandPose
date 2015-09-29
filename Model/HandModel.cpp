/**
 * Copyright 2013: James Steven Supancic III
 **/

#include <boost/multi_array.hpp>

#include "HandModel.hpp"
#include "OneFeatureModel.hpp"
#include "GlobalMixture.hpp"
#include "Log.hpp"
#include "util_real.hpp"
#include "Orthography.hpp"
#include "vec.hpp"
#include "KDTrees.hpp"
#include "RespSpace.hpp"
#include "Semaphore.hpp"
#include "MetaFeatures.hpp"
#include "Eval.hpp"
#include "Faces.hpp"

namespace deformable_depth
{  
  // SECTION: Part selectors
  // return the parts in all_pos which have a name in part_names
  map<string,AnnotationBoundingBox> select_general
  (const map<string,AnnotationBoundingBox>&all_pos,
   const set<string>&part_names)
  {
    // select the distal phalanges...
    map<string,AnnotationBoundingBox> subset;
    auto copyIfFound = [&](const string&cur_part)
    {
      if(all_pos.find(cur_part) != all_pos.end())
      {
	const AnnotationBoundingBox&src_rect = all_pos.at(cur_part);
	subset[cur_part] = src_rect;
      }	
    };
    
    for(const string&part_name : part_names)
      copyIfFound(part_name);
    
    return subset;    
  }    
  
  PartSelector select_fingers = [](const map<string,AnnotationBoundingBox>&all_pos)
    {
      set<string> part_names;
      for(int iter = 1; iter <= 5; iter++) 
      {
	string cur_part_dist = printfpp("dist_phalan_%d",iter);
	string cur_part_prox = printfpp("proxi_phalan_%d",iter);
	part_names.insert(cur_part_dist);
	part_names.insert(cur_part_prox);
      }
      
      return select_general(all_pos,part_names);
    };
    
  PartSelector select_hand = [](const map<string,AnnotationBoundingBox>&all_pos)
    {
      map<string,AnnotationBoundingBox> subset;
      AnnotationBoundingBox handBB = all_pos.at("HandBB");
      subset["HandBB"] = handBB;
      return subset;
    };
  
  /** SECTION: HandFingerTipModel **/
  void HandFingerTipModel::train(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {   
    if(g_params.has_key("LOAD_MODEL"))
    {
      FileStorage fs(g_params.require("LOAD_MODEL"),FileStorage::READ);
      assert(fs.isOpened());
      const FileNode&node = fs["model"];
      assert(!node.empty());
      node >> *this;
      fs.release();
      //log_im("HighResModel",show("loaded model")); //waitKey_safe(0);
    }
    else
    {
      train_sizes(training_set,train_params);
      deformable_depth::StarModel::train(training_set, train_params);
    }    
  }
  
  void HandFingerTipModel::train_sizes(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
    // compute mean gt size
    finger_count = 0;
    finger_gtMuX = 0;
    finger_gtMuY = 0;
    
    active_worker_thread.V();
    #pragma omp parallel for
    for(int iter = 0; iter < training_set.size(); iter++)
    {
      active_worker_thread.P();
      // load the data
      shared_ptr<MetaData>& metadata = training_set[iter];
      shared_ptr<const ImRGBZ> im = metadata->load_im();
      
      // for setting perspecitve min and max scales
      map<string,AnnotationBoundingBox> parts = select_fingers(metadata->get_positives());
      for(pair<string,AnnotationBoundingBox > part : parts)
      {
	#pragma omp critical
	{
	  finger_gtMuX += part.second.width;
	  finger_gtMuY += part.second.height;
	  finger_count += 1;
	}
      }
      active_worker_thread.V();
    }
    active_worker_thread.P();
    finger_gtMuX /= finger_count; finger_gtMuY /= finger_count;
  }    
  
  bool HandFingerTipModel::part_is_root_concreate(string part_name) const
  {
    //return true;
    //return part_name == "HandMMwristmixture";
    return false;
  }
    
  void HandFingerTipModel::prime_parts(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {      
    vector<shared_ptr<MetaData> > left_only = filterLeftOnly(training_set);
    
    // allocate the fingers
    vector<string> part_names;
    for(int iter = 0; iter < 5; ++iter)
    {
      part_names.push_back(printfpp("dist_phalan_%d",iter+1));
      //part_names.push_back(printfpp("inter_phalan_%d",iter+1));
      part_names.push_back(printfpp("proxi_phalan_%d",iter+1));
    }
    for(auto && part_name : part_names)
    {
      Model::TrainParams part_params;
      part_params.subset_cache_key = "HandMM" + part_name + "mixture";
      part_params.part_subset = select_fingers;
      part_params.subordinate_builder.reset(
	new OneFeatureModelBuilder(
	  fingerC,fingerNAREA,
	  shared_ptr<IHOGComputer_Factory>(new NullFeat_FACT()),
				   fingerMinArea,fingerAreaVariance));
      // COMBO_FACT_DEPTH(), ZHOG_FACT() NullFeat_FACT
      
      shared_ptr<SupervisedMixtureModel> fingerTipModel;
      fingerTipModel.reset(new SupervisedMixtureModel(part_name));
      fingerTipModel->prime(left_only,part_params);
      
      parts[part_name] = fingerTipModel;
    }
    
    // allocate a part for the wrist
    {
      Model::TrainParams part_params;
      string part_name = "wrist";
      part_params.subset_cache_key = "HandMM" + part_name + "mixture";
      part_params.part_subset = [=](const map<string,AnnotationBoundingBox>&all_pos)
      {
	return select_general(all_pos,set<string>{part_name});
      };
      part_params.subordinate_builder.reset(
	new OneFeatureModelBuilder(
	  wristC,wristNAREA,
	  shared_ptr<IHOGComputer_Factory>(new NullFeat_FACT()),
				   wristMinArea,wristAreaVariance));
      // COMBO_FACT_DEPTH(), ZHOG_FACT() NullFeat_FACT
      
      shared_ptr<SupervisedMixtureModel> wristModel;
      wristModel.reset(new SupervisedMixtureModel(part_name));
      wristModel->prime(left_only,part_params);
      
      parts[part_name] = wristModel;
    }
  }
  
  void HandFingerTipModel::prime_root(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
    vector<shared_ptr<MetaData> > left_only = filterLeftOnly(training_set);
    
    root_mixture.reset(new SupervisedMixtureModel());
    Model::TrainParams params;
    params.subset_cache_key = "HandGMM";
    params.part_subset = select_hand;
    params.subordinate_builder.reset(
      new OneFeatureModelBuilder(handC,handNAREA,
				 shared_ptr<IHOGComputer_Factory>(new NullFeat_FACT),
				 handMinArea,handAreaVariance));
    // COMBO_FACT_RGB_DEPTH, COMBO_FACT_DEPTH or ZHOG_FACT, NullFeat_FACT
    root_mixture->prime(left_only,params);
  }
  
  void HandFingerTipModel::filter_lr_using_face(
    const ImRGBZ& im, 
    DetectionSet& left_dets, 
    DetectionSet& right_dets) 
  {
    FaceDetector faceDetector;
    log_im("FaceDetections",faceDetector.detect_and_show(im));
    vector<Rect> faces = faceDetector.detect(im);
    if(faces.size() == 0)
      return; // nothing we can do to help.
    
    // left <===> im <===> right
    double right_of_faces = -inf, left_of_faces = inf;
    for(Rect face : faces)
    {
      right_of_faces = std::max<double>(right_of_faces,face.br().x);
      left_of_faces  = std::min<double>(left_of_faces,face.tl().x);
    }
    DetectionSet unfiltered_rights = right_dets; 
    right_dets = DetectionSet();
    DetectionSet unfiltered_lefts  = left_dets; 
    left_dets = DetectionSet();
    
    // detection of right hand must have x coord less than upper bound
    // i.e. right_of_faces
    for(DetectorResult & right_det : unfiltered_rights)
      if(rectCenter(right_det->BB).x < right_of_faces)
	right_dets.push_back(right_det);
      
    // detections of left hands must have x coordinate greater than lower bound 
    // i.e. left_of_faces
    for(DetectorResult & left_det  :unfiltered_lefts)
      if(rectCenter(left_det->BB).x > left_of_faces)
	left_dets.push_back(left_det);
  }
  
  DetectionSet HandFingerTipModel::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    // I don't want to put the LR symmetry into the star model, so I put it here.
    DetectionSet left_dets = deformable_depth::StarModel::detect(im, filter);
    auto left_det_parts = left_dets.part_candidates;
    // these need to be fliped
    assert(filter.feat_pyr == nullptr);
    const ImRGBZ imflip = im.flipLR();
    DetectionSet right_dets = deformable_depth::StarModel::detect(imflip, filter);
    // do the flipping
    auto right_det_parts = right_dets.part_candidates;
    Mat affine = affine_lr_flip(im.cols());
    for(auto && detection :right_dets)
      detection_affine(*detection,affine);
    // flip the right parts
    for(auto && right_part_set : right_det_parts)
      for(auto && right_part : right_part_set.second)
	detection_affine(*right_part,affine);

    // use a face detector, only allow right hands on the right side and left
    // hands on the left side.
    //if(filter.testing_mode)
      //filter_lr_using_face(im,left_dets,right_dets);
    
    // debug, log the LR images
    #pragma omp critical
    if(filter.verbose_log)
    {
      //log_im("DBG:LR",vertCat(im.RGB,imflip.RGB));
      //filter.apply(left_dets);
      //filter.apply(right_dets);
      //test_model_show(*this,im,left_dets,"DetLeft");
      //test_model_show(*this,imflip,right_dets,"DetRight");
    }

    // checks
    for(auto && detection :right_dets)
      assert(detection->lr_flips == 1);
    for(auto && detection : left_dets)
      assert(detection->lr_flips == 0);

    left_dets.insert(left_dets.end(),right_dets.begin(),right_dets.end());
    filter.apply(left_dets);
    // merge the detection parts for debugging
    decltype(left_det_parts) det_parts;
    for(auto && part_group : left_det_parts)
      det_parts[part_group.first].insert(
	det_parts[part_group.first].end(),part_group.second.begin(),part_group.second.end());
    for(auto && part_group : right_det_parts)
      det_parts[part_group.first].insert(
	det_parts[part_group.first].end(),part_group.second.begin(),part_group.second.end());
    if(filter.testing_mode)
      assert(det_parts.size() > 0);
    left_dets.part_candidates = det_parts;
    return left_dets;
  }

  Mat HandFingerTipModel::vis_result(const ImRGBZ&im,Mat& background, DetectionSet& dets) const
  {
    return deformable_depth::StarModel::vis_result(im,background, dets);
  }
  
  SparseVector HandFingerTipModel::extractPos(MetaData& metadata, AnnotationBoundingBox bb) const
  {
    if(metadata.leftP())
      return deformable_depth::StarModel::extractPos(metadata, bb);
    else
    {
      log_once(printfpp("reason: we don't extract from the right hand... [%s]",metadata.get_filename().c_str()));
      return vector<float>();
    }
  }
    
  double HandFingerTipModel::getC() const
  {
    return params::C();
  }

  string HandFingerTipModel::getRootPartName() const
  {
    return "HandBB";
  }
  
  /** SECTION: HandModel_Builder**/
  Model* HandFingerTipModel_Builder::build(Size gtSize, Size imSize) const
  {
    return new HandFingerTipModel();
  }

  string HandFingerTipModel_Builder::name() const
  {
    return "HandFingerTipModel";
  }
  
  /// SECTION IO:
  bool HandFingerTipModel::write(FileStorage& fs)
  {
    fs << "{";
    fs << "finger_gtMuX" << finger_gtMuX;
    fs << "finger_gtMuY" << finger_gtMuY;
    fs << "finger_count" << finger_count;
    fs << "StarModel"; assert(StarModel::write(fs));
    fs << "}";
    return true;
  }
  
  void read(const FileNode& node, HandFingerTipModel& model, 
	    const HandFingerTipModel& default_value)
  {
    assert(!node.isNone());
    node["finger_gtMuX"] >> model.finger_gtMuX;
    node["finger_gtMuY"] >> model.finger_gtMuY;
    node["finger_count"] >> model.finger_count;
    read(node["StarModel"],static_cast<StarModel&>(model));
  }
}
