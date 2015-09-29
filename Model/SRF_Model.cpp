/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "SRF_Model.hpp"
#include <boost/multi_array.hpp>
#include <boost/graph/graph_concepts.hpp>
#include "Orthography.hpp"
#include "Log.hpp"
#include "util_vis.hpp"
#include "RespSpace.hpp"
#include "Probability.hpp"
#include "util.hpp"
#include "Skin.hpp"
#include <boost/math/special_functions/round.hpp>
#include <bits/stl_queue.h>
#include "Faces.hpp"
#include "Detection.hpp"
#include "ScanningWindow.hpp"

namespace deformable_depth
{    
  ///
  /// SECTION: SRF_Model
  /// 
  SRFModel::SRFModel(Size gtSize, Size ISize) :
    gtSize(gtSize), ISize(ISize)
  {
  }
  
  SRFModel::SRFModel()
  {
  }
    
  void SRFModel::train(
    vector< shared_ptr< MetaData > >& training_set, 
    Model::TrainParams train_params)
  {
    // train the PCA
    pca_pose.train(training_set);
    
    // prime the feature extractor
    log_file << printfpp("SRFModel::train Got training set of size %d",training_set.size()) << endl;    
     
    // extract the features
    vector<StructuredExample> all_feats = extract_features(training_set);          
    log_file << "SRFModel::train all_feats.size = " << all_feats.size() << endl;
    
    // try to load Hough layer from file
    if(g_params.has_key("LOAD_MODEL"))
    {
      FileStorage model_file(g_params.get_value("LOAD_MODEL"),FileStorage::READ);
      model_file["model"] >> *this;
      model_file.release();
    } // else train the hough layer
    else
    { 
      // generate candidate splits and choose one
      log_file << printfpp("SRFModel: Training with %d examples!",(int)all_feats.size()) << endl;
      
      // train the tree
      tree_root.reset(new DiscriminativeHoughTree(*this,all_feats));      
    }
    log_file << "SRFModel::train trained Hough layer" << endl;
    
    // train the regression layer
    // (B1) : Augment the features
    // (B2) : Train the regression layer
    //vector<StructuredExample> gt_feats = extract_gt_features(all_feats);
    ///entangled_layer.reset(new DiscriminativeHoughTree(gt_feats));
    
    log_file << "-SRFModel::train" << endl;
  }
    
  DetectionSet SRFModel::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    log_file << "+SRFModel::detect" << endl;
    DetectionFilter filt(-inf);
    filt.supress_feature = true;
    filt.manifoldFn = manifoldFn_apxMin;
    DetectionSet windows = removeTruncations(im,enumerate_windows(im,filt));
    log_file << "SRFModel::Detect examining windows: " << windows.size() << endl;
    shared_ptr<const ImRGBZ> im_copy(new ImRGBZ(im));
    
    // vote the positives and negatives...
    //LatentHoughOutputSpace hand_space(getPCApose());
    HoughOutputSpace hand_space;
    //std::random_shuffle(windows.begin(),windows.end());
    //Mat vis_window = im.RGB.clone();
    //string win_name = printfpp("SlidingWindow_%s",im.filename.c_str());
    map<string,double> leaf_usages;
    for(auto && win : windows)
    {
      StructuredWindow swin;
      swin.detection = win;
      swin.im = im_copy;
      
      // vote the positive
      VoteResult vr = tree_root->vote(hand_space,swin);
      leaf_usages[vr.leaf->get_uuid()] += vr.conf;
      
      // vote the negative
      // vote_flat(hand_space, rectResize(swin.ortho_detBB(),.5,.5), conf);
      
      // show the sliding window
      //rectangle(vis_window,win->BB.tl(),win->BB.br(),Scalar(rand()%255,rand()%255,rand()%255));
      //image_safe(win_name,vis_window);
    }
    // invert and produce debug output
    multimap<double,string> inverted_leaf_usages;
    for(auto & pair : leaf_usages)
      inverted_leaf_usages.insert(std::pair<double,string>(pair.second,pair.first));
    log_file << "Most voted leaf: " << inverted_leaf_usages.rbegin()->second << endl;
    
    // visualize the Hough results
    shared_ptr<HoughLikelihoods> hough_likelihoods = hand_space.likelihood_ratio(im);
    
    // backproject from Hough space to score detections
    DetectionSet dets;
    for(auto && win : windows)
    {
      hough_likelihoods->read_detection(win,im_copy);
      if(win->resp > -inf)
	dets.push_back(win);
      //entangled_layer->predict_update(win);
    }    
    
    // for fairness
    dets = SimpleFaceDetector().filter(dets,im,.2);
    dets = nms_w_list(dets,.75);
    
    if(dets.size() > 0)
    {
      log_file << "Top Latent Position: " << dets[0]->latent_space_position << endl;
    }
    
    return dets;
  }

  bool SRFModel::write(FileStorage& fs)
  {
    fs << "{";
    fs << "Tree1" << *tree_root;
    fs << "gtSize" << gtSize;
    fs << "ISize" << ISize;
    fs << "}";
    
    return true;
  }
  
  void read(const FileNode& fn, SRFModel& srfModel, SRFModel )
  {
    srfModel.tree_root.reset(new HoughTree(srfModel));
    read(fn["Tree1"],*srfModel.tree_root);
    deformable_depth::read<int>(fn["gtSize"],srfModel.gtSize);
    deformable_depth::read<int>(fn["ISize"],srfModel.ISize);
  }
  
  Mat SRFModel::show(const string& title)
  {
    return Mat();
  }
  
  const PCAPose& SRFModel::getPCApose() const
  {
    return pca_pose;
  }
  
  PCAPose& SRFModel::getPCApose()
  {
    return pca_pose;
  }
  
  /// 
  /// Single Structural Random Tree
  ///
  HoughTree::HoughTree(SRFModel&model) : model(model)
  {
  }
  
  const SRFModel& HoughTree::getModel() const
  {
    return model;
  }

  SRFModel& HoughTree::getModel()
  {
    return model;
  }
  
  void DiscriminativeHoughTree::discriminative_try_split(
    DiscriminativeHoughTree* node,
    double&best_info_gain,
    DiscriminativeHoughTree**best_node,
    vector<StructuredExample>&exs_true,
    vector<StructuredExample>&exs_false,
    unique_ptr<RandomHoughFeature>&best_feature,
    map<string,HoughOutputSpace::Matrix3D>&target_output,
    ObjType obj_type)
  {
    vector<StructuredExample> & examples = node->training_examples;
    unique_ptr<RandomHoughFeature> rf(new RandomHoughFeature(examples));
    vector<StructuredExample> exs_true_here, exs_false_here;
    log_file << "discriminative_try_split,, examples.size() = " << examples.size() << endl;
    InformationGain info_gain = rf->info_gain(
      examples,exs_true_here,exs_false_here,node->getModel().getPCApose());
    if(info_gain.differential_info_gain == -inf || info_gain.shannon_info_gain == -inf)
    {
      log_file << "Bad split" << endl;
      return;
    }
    
    double spearman_gain = qnan;
    if(obj_type == ObjType::Spearman)
    {
      // compute the updated predictions
      map<string,HoughOutputSpace> delta_spaces;
      int cur_count = 0;
      vector<StructuredExample> voting_samples = 
	random_sample_w_replacement(examples,std::min<int>(2000,examples.size()));
      for(StructuredExample & ex : voting_samples)
      {
	// vote the sample
	string filename = ex.metadata->get_filename();
	rf->vote(delta_spaces[filename],ex,node->getModel().getPCApose());
	
	// log
	if(cur_count % std::max<int>(100,(voting_samples.size()/1000)) == 0)
	    log_file << printfpp("discriminative_try_split::vote_gaussian %d of %d",
				(int)cur_count,(int)voting_samples.size()) << endl;
	cur_count ++ ;
      }
      log_file << "discriminative_try_split: finished voting" << endl;
      
      // compute the correlation
      spearman_gain = 0;
      for(auto pair : delta_spaces)
      {
	const string&filename = pair.first;
	HoughOutputSpace&delta_space = pair.second;
	spearman_gain += HoughOutputSpace::spearman_binary_delta(
	  target_output[filename],delta_space.positive,delta_space.negative);
      }
      log_file << "spearman_gain: " << spearman_gain << endl;
    }
    
    double ig = qnan;
    switch(obj_type)
    {
      case Spearman:
	ig = spearman_gain;
	break;
      case Differential:
	ig = info_gain.differential_info_gain;
	break;
      case Shannon:
	ig = info_gain.shannon_info_gain;
	break;
      case Pose_Entropy:
	ig = info_gain.latent_structural_gain;
	break;
      default:
	assert(false);
    };
    
    // update the current best
    static mutex m; lock_guard<mutex> l(m);
    if(best_info_gain < ig)
    {
      *best_node = node;
      best_info_gain = ig;
      best_feature = std::move(rf);
      exs_true = exs_true_here;
      exs_false = exs_false_here;
    }    
  }
  
  void DiscriminativeHoughTree::discriminative_split_choose
  (list<DiscriminativeHoughTree*>&unsplit_nodes,
   map<string,HoughOutputSpace::Matrix3D>&target_output,
   DiscriminativeHoughTree** best_node,
   double&best_info_gain,
   unique_ptr<RandomHoughFeature>&best_feature,
   vector<StructuredExample>&exs_true, 
   vector<StructuredExample>&exs_false
  )
  {
    // generate the indexes to split
    vector<double> node_weights;
    for(auto node : unsplit_nodes)
      node_weights.push_back(node->training_examples.size());
    node_weights = node_weights / sum(node_weights);
    std::multiset<int> node_indexes; 
    int node_idx = rnd_multinom(node_weights);
    for(int iter = 0; iter < FEATURES_PER_SPLIT; ++iter)
    {
      node_indexes.insert(node_idx);
    }
    log_file << "node_weights: " << node_weights << endl;
    
    // choose a objective to optimize
    vector<double> obj_type_thetas{0.3,0.3,0.3,.1};
    ObjType obj_type = static_cast<ObjType>(rnd_multinom(obj_type_thetas));
    
    // generate random splits over the unsplit set
    TaskBlock compute_info_gains("compute_info_gains");
    best_info_gain = -inf;
    int node_index = 0;
    for(auto node_iter = unsplit_nodes.begin();
	node_iter != unsplit_nodes.end(); ++node_iter, ++node_index)
    {
      log_file << printfpp("queueing %d instances at index %d",
			   (int)node_indexes.count(node_index),node_index) << endl;
      for(int jter = 0; jter < node_indexes.count(node_index); jter++)
      {
	DiscriminativeHoughTree* node = *node_iter;
	compute_info_gains.add_callee([&,node]()
	{
	  discriminative_try_split(
	    node,best_info_gain,best_node,exs_true,exs_false,best_feature,target_output,obj_type);
	});
      }
    }
    compute_info_gains.execute();    
  }
  
  void DiscriminativeHoughTree::discriminative_split_update_tree
  (list<DiscriminativeHoughTree*>&unsplit_nodes,
    DiscriminativeHoughTree* best_node,
   double&best_info_gain,
   unique_ptr<RandomHoughFeature>&best_feature,
   vector<StructuredExample>&exs_true, 
   vector<StructuredExample>&exs_false
  )
  {
    // remove the best node
    log_file << "best_spearman_gain = " << best_info_gain << endl;
    //assert(false);// DEBUG: TODO Remove
    int s1 = unsplit_nodes.size();
    unsplit_nodes.remove(best_node);// remove the best node
    if(s1 - unsplit_nodes.size() != 1)
    {
      log_file << printfpp("unsplit_nodes size %d => %d",s1,(int)unsplit_nodes.size()) << endl;
      assert(false);
    }
	  
    // log the choice.
    log_file << "SELECTED FEATURE" << endl;
    log_file << *best_feature << endl;
    best_feature->log_kernels();    
    
    // enqueue valid branches for further subdivsion
    best_node->predictor = std::move(best_feature);
    if(best_node->allow_split_true())
    {
      DiscriminativeHoughTree*new_true = new DiscriminativeHoughTree(best_node->getModel(),exs_true,best_node->depth+1);
      best_node->true_branch.reset(new_true);
      unsplit_nodes.push_back(new_true);
    }
    if(best_node->allow_split_false())
    {
      DiscriminativeHoughTree* new_false = new DiscriminativeHoughTree(best_node->getModel(),exs_false,best_node->depth+1);
      best_node->false_branch.reset(new_false);
      unsplit_nodes.push_back(new_false);
    }
    
    // clear best's training set to avoid memory waste
    best_node->training_examples.clear();     
  }
  
  void DiscriminativeHoughTree::discriminative_split(
    list<DiscriminativeHoughTree*>&unsplit_nodes,
    map<string,HoughOutputSpace::Matrix3D>&target_output)
  {
    // variables
    DiscriminativeHoughTree* best_node = nullptr;
    double best_info_gain;
    unique_ptr<RandomHoughFeature> best_feature;
    vector<StructuredExample> exs_true, exs_false;
    
    // choose the best split
    discriminative_split_choose(
      unsplit_nodes,
      target_output,
      &best_node,
      best_info_gain,
      best_feature,
      exs_true, 
      exs_false); 
    
    // update the tree using the split
    discriminative_split_update_tree(
      unsplit_nodes,
      best_node,
      best_info_gain,
      best_feature,
      exs_true, 
      exs_false);   
  }
  
  DiscriminativeHoughTree::DiscriminativeHoughTree(
    SRFModel&model,
    vector< StructuredExample >& training_examples, int depth) : 
    training_examples(training_examples),
    HoughTree(model)
  {
    // subsplits will be split by the root
    this->depth = depth;
    if(depth > 0)
      return;
    
    // create a default space
    map<string,HoughOutputSpace> hand_space;
    // build the target matrices
    map<string,HoughOutputSpace::Matrix3D> output_targets;
    vector<StructuredExample> positive_examples, negative_examples;
    for(StructuredExample & ex : training_examples)
    {
      string filename = ex.metadata->get_filename();
      if(output_targets.find(filename) == output_targets.end())
	output_targets.insert(
	  std::pair<string,HoughOutputSpace::Matrix3D>(filename,hand_space[filename].positive)); 
	  // should defalut to 0
      
      if(ex.is_correct())
      {
	Point3d bb_cen = ex.ortho_detBBCenter();
	//cout << output_targets[filename].shape()[0] << endl;
	//cout << output_targets[filename].shape()[1] << endl;
	//cout << output_targets[filename].shape()[2] << endl;
	//cout << bb_cen << endl;
	double & target_here = output_targets[filename].at(bb_cen.x,bb_cen.y,0);
	target_here = 1;
	positive_examples.push_back(ex);
      }
      else
	negative_examples.push_back(ex);
    }
    //max_z(likelihood)
    
    // do the bootstrap sample
    this->training_examples = training_examples;
      //random_sample_w_replacement(training_examples,training_examples.size());
    require_gt(negative_examples.size(),positive_examples.size());
    log_file << printfpp("negative set size %d => %d",(int)negative_examples.size(),(int)positive_examples.size()) << endl;
//     negative_examples = random_sample_w_replacement(negative_examples,positive_examples.size());
//     this->training_examples.clear();
//     this->training_examples.insert(
//       this->training_examples.end(),negative_examples.begin(),negative_examples.end());
//     this->training_examples.insert(
//       this->training_examples.end(),positive_examples.begin(),positive_examples.end());
    std::random_shuffle(this->training_examples.begin(),this->training_examples.end());
    
    // iteratively construct the tree
    list<DiscriminativeHoughTree*> unsplit_nodes;
    unsplit_nodes.push_back(this);
    while(!unsplit_nodes.empty())
    {
      discriminative_split(unsplit_nodes,output_targets);
      log_file << "# unsplit nodes = " << unsplit_nodes.size() << endl;
    }
  }
  
  bool HoughTree::allow_split_true() const
  {
    return    depth < MAX_DEPTH && 
	      cv::determinant(predictor->get_cov_true()) > MIN_CoVar_DET && 
	      predictor->get_N_true() > MIN_EXAMPLES &&
	      predictor->get_N_true_pos() > MIN_POS_EXAMPLES && 
	      predictor->get_N_true_neg() > MIN_NEG_EXAMPLES;
  }

  bool HoughTree::allow_split_false() const
  {
    return    depth < MAX_DEPTH &&
	      cv::determinant(predictor->get_cov_false()) > MIN_CoVar_DET &&
	      predictor->get_N_false() > MIN_EXAMPLES &&
	      predictor->get_N_false_pos() > MIN_POS_EXAMPLES &&
	      predictor->get_N_false_neg() > MIN_NEG_EXAMPLES;
  }
  
  StructuralHoughTree::StructuralHoughTree(SRFModel&model,
    vector< StructuredExample >& training_examples, int depth) :
      HoughTree(model)
  {
    // choose a split for this level
    // generate some random features and compute their info gains
    this->depth = depth;
    TaskBlock compute_info_gains("SRF_Model::Train compute info gains");
    double max_info_gain = -inf;
    vector<StructuredExample> exs_true, exs_false;
    for(int iter = 0; iter < FEATURES_PER_SPLIT; ++iter)
    {
      compute_info_gains.add_callee([&,iter]()
      {
	vector<StructuredExample> bootstrap_sample;
	if(depth == 0)
	{
	  // generate a bootstrap sample
	  bootstrap_sample = 
	    random_sample_w_replacement(training_examples,training_examples.size()/10);
	}
	else
	  bootstrap_sample = training_examples;
	  
	// generate a random feature
	unique_ptr<RandomHoughFeature> rf(new RandomHoughFeature(training_examples));
	
	// evaluate the quality
	vector<StructuredExample> exs_true_here, exs_false_here;
	InformationGain info_gain = rf->info_gain(
	  bootstrap_sample,exs_true_here,exs_false_here,model.getPCApose());
	log_file << printfpp("rf.info_gain = %f",info_gain) << endl;	
	{
	  static mutex m; lock_guard<mutex> l(m);
	  double h_here = ((depth%2)?info_gain.differential_info_gain:info_gain.shannon_info_gain);
	  if(h_here > max_info_gain)
	  {
	    max_info_gain = h_here;
	    predictor = std::move(rf);
	    exs_true = exs_true_here;
	    exs_false = exs_false_here;
	  }
	}
      });
    }
    compute_info_gains.execute();
    log_file << "SELECTED FEATURE" << endl;
    log_file << *predictor << endl;
    predictor->log_kernels();    
    
    if(depth < MAX_DEPTH)
    {
      // split recursively
      TaskBlock tree_recursive_split("tree_recursive_split");
      if(allow_split_true())
	tree_recursive_split.add_callee([&](){
	    true_branch.reset(new StructuralHoughTree(model,exs_true,depth+1));
	});
      if(allow_split_false())
	tree_recursive_split.add_callee([&](){
	  false_branch.reset(new StructuralHoughTree(model,exs_false,depth+1));
	});      
      tree_recursive_split.execute();
    }
  }

  VoteResult HoughTree::vote(HoughOutputSpace& output, 
			 const StructuredWindow& swin,
			 RandomHoughFeature*predictor) const
  {
    // default value for this argument.
    if(predictor == nullptr)
      predictor = &*this->predictor;
    
    bool prediction = predictor->predict(swin);
    if(prediction && true_branch != nullptr)
    {
      // if we predict true and true branch can be taken
      return true_branch->vote(output,swin);
    }
    else if(!prediction && false_branch != nullptr)
    {
      // if we predict false and the false branch can be taken
      return false_branch->vote(output,swin);
    }
    else
    {
      // predict here otherwise
      return predictor->vote(output,swin,model.getPCApose());
    }
  }
  
  void write(cv::FileStorage&fs, std::string&, const deformable_depth::HoughTree&srt)
  {
    fs << "{";
    fs << "depth" << srt.depth;
    fs << "predictor" << srt.predictor;
    if(srt.true_branch != nullptr)
      fs << "true_branch" << *srt.true_branch;
    if(srt.false_branch != nullptr)
      fs << "false_branch" << *srt.false_branch;
    fs << "}";
  }
  
  void read(const FileNode& fn, HoughTree& srt )
  {
    fn["depth"] >> srt.depth;
    log_file << "read tree node @ depth = " << srt.depth << endl;
    fn["predictor"] >> srt.predictor;
    //srt.predictor->log_kernels();
    if(!fn["true_branch"].empty())
    {
      srt.true_branch.reset(new HoughTree(srt.getModel()));
      read(fn["true_branch"],*srt.true_branch);
    }
    if(!fn["false_branch"].empty())
    {
      srt.false_branch.reset(new HoughTree(srt.getModel()));
      read(fn["false_branch"],*srt.false_branch);    
    }
  }
  
  ///
  /// SECTION: SRFModel_Model_Builder
  ///
  
  Model* SRFModel_Model_Builder::build(Size gtSize, Size imSize) const
  {
    return new SRFModel(gtSize,imSize);
  }
  
  SRFModel_Model_Builder::SRFModel_Model_Builder(double C, 
						 double area, 
						 shared_ptr< IHOGComputer_Factory > fc_factory, 
						 double minArea, double world_area_variance)
  {
  }
  
  string SRFModel_Model_Builder::name() const
  {
    return "SRFModel_Model_Builder";
  }
}
