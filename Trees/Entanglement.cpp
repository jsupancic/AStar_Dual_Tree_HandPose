/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Entanglement.hpp"
#include "util_vis.hpp"
#include <list>
#include "Entanglement.hpp"
#include "Entanglement.hpp"
#include <boost/filesystem.hpp>

namespace deformable_depth
{
  using namespace std;
  
  ///
  /// Section: Entangled Random Tree
  ///
  EntangledTree::EntangledTree(const EntangledTree& copy) : 
    true_samples(copy.true_samples), false_samples(copy.false_samples),
    leaf_correct_exs(copy.leaf_correct_exs)
  {
    if(copy.predictor)
      predictor.reset(new RandomFeature(*copy.predictor));
    if(copy.true_branch)
      true_branch.reset(new EntangledTree(*copy.true_branch));
    if(copy.false_branch)
      false_branch.reset(new EntangledTree(*copy.false_branch));
  }
  
  double EntangledTree::prediction() const
  {
    return true_samples/(true_samples+false_samples);
  }
  
  EntangledTree::EntangledTree(double true_samples, double false_samples) : 
    true_samples(true_samples), false_samples(false_samples)
  {
  }
  
  template<typename T>
  // T = StructuredWindow
  DetectionSet EntangledTree::detect(const vector<T>&windows) const
  {
    DetectionSet dets; dets.resize(windows.size());
    vector<const EntangledTree*> nodes(windows.size(),this);
    
    bool done = false;
    while(!done)
    {
      done = true;
      for(int iter = 0; iter < windows.size(); ++iter)
	if(!dets[iter])
	{
	  if(!nodes[iter]->predictor)
	  {
	    // make a prediction
	    dets[iter].reset(new Detection(*(windows[iter].detection)));
	    dets[iter]->resp = nodes[iter]->prediction();
	  }
	  else
	  {
	    // move down a tree
	    if(nodes[iter]->predictor->predict(windows[iter]))
	    {
	      nodes[iter] = &*nodes[iter]->true_branch;
	    }
	    else
	    {
	      nodes[iter] = &*nodes[iter]->false_branch;
	    }
	    
	    done = false;
	  }
	}
    }
    
    return dets;
  }

  bool EntangledTree::grow_self(const vector<StructuredExample >&samples, bool do_split) 
  { 
    double true_pos, false_pos, true_neg, false_neg;
    vector<StructuredExample> exs_true, exs_false, correct_exs_true,correct_exs_false;    
    
    if(!predictor)
    {
      if(do_split && samples.size() > 100)
      {
	// use the leaf correct exs
	vector<StructuredExample> union_samples(samples.begin(),samples.end());
	union_samples.insert(union_samples.end(),
			     leaf_correct_exs.begin(),leaf_correct_exs.end());
	leaf_correct_exs.clear();
	
	// generate a new predictor
	double bestIG = -inf;
	for(int trySplitIter = 0; trySplitIter < 25; ++trySplitIter)
	{
	  // try out a new predictor
	  unique_ptr<RandomFeature> newPredictor(new RandomFeature(union_samples));
	  // comptue its effect
	  newPredictor->split_examples(union_samples, exs_true, exs_false,
		        true_pos, false_pos, true_neg, false_neg,
		        correct_exs_true, correct_exs_false);  
	  double true_ratio = exs_true.size() / (double)union_samples.size();
	  double false_ratio = exs_false.size() / (double)union_samples.size();
	  double ig = 	
	    - true_ratio * shannon_entropy({true_pos/exs_true.size(),true_neg/exs_true.size()}) 
	    - false_ratio* shannon_entropy({false_pos/exs_false.size(),false_neg/exs_false.size()});
	  
	  // generate decision nodes for it
	  if(ig > bestIG)
	  {
	    bestIG = ig;
	    predictor = std::move(newPredictor);
	    true_branch.reset(new EntangledTree(true_pos+1,true_neg+1));
	    true_branch->leaf_correct_exs = correct_exs_true;
	    false_branch.reset(new EntangledTree(false_pos+1,false_neg+1));
	    false_branch->leaf_correct_exs = correct_exs_false;
	  }
	}
	return true;
      }
      else
      {
	// just update counts
	for(auto & ex : samples)
	  if(ex.is_white())
	  {
	    leaf_correct_exs.push_back(ex);
	    true_samples++;
	  }
	  else if(ex.is_black())
	    false_samples++;
	return false;
      }
    }
    else
    {
      predictor->split_examples(samples, exs_true, exs_false,
		true_pos, false_pos, true_neg, false_neg,
		correct_exs_true, correct_exs_false);         
      bool true_splitable = true_pos > 5 && true_neg > 100;
      bool false_splitable = false_pos > 5 && false_neg > 100;
      
      // update self
      true_samples += true_pos + false_pos;
      false_samples += true_neg + false_neg;
      
      // compute "test" entropies for each branch
      double true_test_h = exs_true.size()*shannon_entropy(true_pos/(true_pos+true_neg));
      double false_test_h = exs_false.size()*shannon_entropy(false_pos/(false_pos+false_neg));
      
      // update children
      vector<double> thetas;
      if(false && !std::isnan(true_test_h) && !std::isnan(false_test_h) && false_test_h + true_test_h > 0)
      {
	thetas = vector<double>{false_test_h,true_test_h};
	thetas = thetas/deformable_depth::sum(thetas);
      }
      else
      {
	thetas = vector<double>{.5,.5};
      }
      log_file << "branch dist = " << thetas << endl;
      log_file << "true_test_h = " << true_test_h << endl;
      log_file << "false_test_h = " << false_test_h << endl;
      int dir = rnd_multinom(thetas);
      assert(dir == 0 || dir == 1);
      if(dir)
      {
	bool grew = true_branch->grow_self(exs_true,true_splitable && do_split);
	grew |= false_branch->grow_self(exs_false,false_splitable && do_split && !grew);
	return grew;
      }
      else
      {
	bool grew = false_branch->grow_self(exs_false,false_splitable && do_split);
	grew |= true_branch->grow_self(exs_true,true_splitable && do_split && !grew);
	return grew;
      }      
    }
  }
  
  shared_ptr< EntangledTree > EntangledTree::grow(const vector<StructuredExample >&samples) const
  {
    shared_ptr<EntangledTree> copy(new EntangledTree(*this));
    for(int growIter = 0; growIter < 1; ++growIter)
      copy->grow_self(samples);
    return copy;
  }
  
  void write(cv::FileStorage&fs, std::string&, const deformable_depth::EntangledTree&tree)
  {
    fs << "{";
    if(tree.predictor)
      fs << "predictor" << *tree.predictor;
    if(tree.true_branch)
      fs << "true_branch" << *tree.true_branch;
    if(tree.false_branch)
      fs << "false_branch" << *tree.false_branch;
    fs << "true_samples" << tree.true_samples;
    fs << "false_samples" << tree.false_samples;
    fs << "prediction" << tree.prediction();
    fs << "}";
  }
  
  ///
  /// Section: Entangled Random Forest
  ///
  ERFModel::ERFModel(Size gtSize, Size ISize) :     gtSize(gtSize), ISize(ISize),
      feature_extractor(new FeatureExtractionModel(gtSize,ISize,
	OneFeatureModel::DEFAULT_C, 
	5*5,
	shared_ptr<IHOGComputer_Factory>(new ZHOG_FACT()))
      )
  {
  }
  
  ERFModel::ERFModel()
  {
  }
  
  DetectionSet ERFModel::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    shared_ptr<ImRGBZ> im_copy(new ImRGBZ(im,true));
    vector<StructuredWindow> wins_here = extract_windows(*feature_extractor,im_copy);    
    return tree->detect(wins_here);
  }

  static double score_candidate(
    shared_ptr<EntangledTree> candidate,
    vector<vector<StructuredExample> >&all_feats,
    vector<shared_ptr<MetaData> >&bootstrap_samples,
    vector<Mat>&visualizations)
  {
    // score each frame.
    vector<double> responces;
    vector<double> correctnesses;
    for(int frameIter = 0; frameIter < all_feats.size(); ++frameIter)
    {
      // run the detector
      auto & frame_feats = all_feats[frameIter];
      DetectionSet dets = candidate->detect(frame_feats);
      assert(frame_feats.size() == dets.size());
      
      // compute the partial score
      for(int iter = 0; iter < dets.size(); ++iter)
      {
	if(frame_feats[iter].is_black())
	{
	  responces.push_back(dets[iter]->resp);
	  correctnesses.push_back(0);
	}
	else if(frame_feats[iter].is_white())
	{
	  responces.push_back(dets[iter]->resp);
	  correctnesses.push_back(1);	  
	}
      }
      
      // update the visualization bucket
      visualizations.push_back(drawDets(*bootstrap_samples[frameIter],dets,1));
    }
    
    return std::abs(spearman(responces,correctnesses));
  }
  
  static void eval_candidate(
    shared_ptr<EntangledTree>&candidate,
    shared_ptr<EntangledTree>&tree,
    vector<vector<StructuredExample> >&all_feats,
    vector<shared_ptr<MetaData> >&bootstrap_samples,
    double&best_score)
  {
    vector<Mat> visualizations;
    double current_score = score_candidate(candidate,all_feats,bootstrap_samples,visualizations);

    log_once(printfpp("Candidate w/ SpearCor = %f",current_score));
    
    // we selected the best one?
    static mutex m; lock_guard<mutex> l(m);
    if(current_score > best_score)
    {
      best_score = current_score;
      tree = candidate;
      log_once(printfpp("Selected Candidate w/ SpearCor = %f",current_score));
      static atomic<int> counter(0);
      log_im(printfpp("UpdatedTestError%d",counter++),tileCat(visualizations));
    }    
  }
  
  void ERFModel::train(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
    // prime the feature extractor
    assert(feature_extractor);
    log_file << printfpp("SRFModel::train Got training set of size %d",training_set.size()) << endl;    
    feature_extractor->prime(training_set,train_params);       
    
    // train a tree
    // greedily construct a tree
    tree.reset(new EntangledTree);
    for(int iter = 0; iter < 50; ++iter)
    {
      // select a test set
      vector<shared_ptr<MetaData> > bootstrap_samples = random_sample(training_set,12);
      // extract the features
      vector<vector<StructuredExample> > all_feats(bootstrap_samples.size());
      vector<StructuredExample> flat_feats;
      TaskBlock windows("windows");
      for(int sampleIter = 0; sampleIter < bootstrap_samples.size(); ++sampleIter)
      {
	windows.add_callee([&,sampleIter]()
	{
	  auto & sample = bootstrap_samples[sampleIter];
	  vector<shared_ptr<MetaData> > samplev{sample};
	  vector<StructuredExample> exs_here = 
	    extract_features(*feature_extractor,samplev);
	  
	  static mutex m; lock_guard<mutex> l(m);
	  flat_feats.insert(flat_feats.end(),exs_here.begin(),exs_here.end()) ;
	  all_feats[sampleIter] = exs_here;	  
	});
      }      
      windows.execute();
      
      // generate candidate growths
      vector<shared_ptr<EntangledTree> > candidates;
      TaskBlock grow("grow");
      for(int jter = 0; jter < params::cpu_count()-1; jter++)
	grow.add_callee([&]()
	{
	  shared_ptr<EntangledTree> canddiate = tree->grow(flat_feats);
	  static mutex m; lock_guard<mutex> l(m);
	  candidates.push_back(canddiate);
	});
      grow.execute();      
      
      // choose the best detection
      double best_score = -inf;
      TaskBlock score("score");
      if(tree)
      {
	shared_ptr<EntangledTree> oldTree = tree;
	score.add_callee([&,oldTree/*copy the original tree ptr by value*/]()
	{
	  // compute a "test" error for the ungrown tree
	  vector<Mat> test_vis;
	  double test_cor = score_candidate(tree,all_feats,bootstrap_samples,test_vis);
	  log_once(printfpp("test_cor = %f",test_cor));
	  log_im("test_results",tileCat(test_vis));	
	});
      }
      for(int candIter = 0; candIter < candidates.size(); candIter++)
      {
	score.add_callee([&,candIter]()
	{
	  eval_candidate(candidates[candIter],tree,all_feats,bootstrap_samples,best_score);
	});
      }
      score.execute();
      
      // save the selected tree
      string filename = params::out_dir() + printfpp("/tree_iter=%d.yml",iter);
      FileStorage tree_storage(filename,FileStorage::WRITE);
      tree_storage << "Tree" << *tree;
      tree_storage.release();
      
      // check for manual stop
      if(boost::filesystem::exists(params::out_dir() + "/manual_stop"))
      {
	log_file << "Manual Stop Command Recived From User" << endl;
	log_file << "Halting Training" << endl;
	break;
      }
    }
  }

  Mat ERFModel::show(const string& title)
  {
    return Mat();
  }

  bool ERFModel::write(FileStorage& fs)
  {
    fs << "{";
    fs << "}";
    return true;
  }
  
  ///
  /// SECTION: Entangled Random Forest Builder
  ///
  ERFModel_Model_Builder::ERFModel_Model_Builder(
    double C, double area, 
    shared_ptr< IHOGComputer_Factory > fc_factory, 
    double minArea, double world_area_variance)
  {
  }

  Model* ERFModel_Model_Builder::build(Size gtSize, Size imSize) const
  {
    return new ERFModel(gtSize,imSize);
  }

  string ERFModel_Model_Builder::name() const
  {
    return "ERF";
  }
}
