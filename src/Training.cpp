/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include <boost/filesystem.hpp>

#include "Training.hpp"
#include "Log.hpp"
#include "Detection.hpp"
#include "OneFeatureModel.hpp"
#include "OcclusionReasoning.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: ExposedLDAModel
  /// 
  void ExposedLDAModel::debug_incorrect_resp(SparseVector& feat, Detection&det)
  {
    // NOP
    assert(false);
  }
  
  void ExposedLDAModel::collect_negatives(
    MetaData& metadata, Model::TrainParams train_params, 
    function< void (DetectorResult) > writeNeg, 
    DetectionFilter filter,
    map<string,AnnotationBoundingBox>&pos_bbs,
    map<string,AnnotationBoundingBox>&train_bbs,
    shared_ptr<ImRGBZ>&im)
  {
    // collect any negative support vectors
    if(train_params.negative_iterations >= 1)
    {
      filter.thresh = -inf;
      DetectionSet dets_for_frame = detect(*im,filter);
      Mat det_vis;
      if(dets_for_frame.size() < 100)
	det_vis = im->RGB.clone();
      // draw the positives and train bbs
      for(auto& pos_bb : pos_bbs)
	rectangle(det_vis,pos_bb.second.tl(),pos_bb.second.br(),Scalar(255,0,0));
      for(auto& train_bb : train_bbs)
	rectangle(det_vis,train_bb.second.tl(),train_bb.second.br(),Scalar(0,255,0));
      
      // draw and record the detections
      for(int detIter = 0; detIter < dets_for_frame.size(); detIter++)
      {
	shared_ptr<Detection> curDet = dets_for_frame[detIter];
	if(dets_for_frame.size() < 100)
	  rectangle(det_vis,curDet->BB.tl(),curDet->BB.br(),Scalar(0,0,255));
	bool use_det = true;
	if(!curDet->feature || curDet->resp < -1)
	  use_det = false;
	// cull from the negative set any detection which is 
	// white (true positive) or gray (similar to a positive).
	// check the root
	if(!curDet->is_black_against(pos_bbs))
	  use_det = false;
	
	// only train on totally contained BB
	//if(clamp(RGB,curDet.BB) == curDet.BB)
	if(!use_det)
	    curDet->supressed = true;
	writeNeg(curDet);
      }
      
      // draw the support vectors
      if(dets_for_frame.size() < 250)
	log_im("detections: ",det_vis);
      else
	log_file << dets_for_frame.size() << 
	  " > 250 detections is to many to log image" << endl;
      // draw the missing positives
      logMissingPositives(metadata,dets_for_frame);
    }
  }

  void ExposedLDAModel::collect_positives(
    MetaData& metadata, 
    Model::TrainParams train_params, 
    function< void (DetectorResult) > writePos, 
    DetectionFilter filter,
    map<string,AnnotationBoundingBox>&train_bbs,
    shared_ptr<ImRGBZ>&im)
  {
    // add positive SV
    // the BB must be within the im.
    for(std::pair<string,AnnotationBoundingBox> pos_bb : train_bbs)
    {      
      if(pos_bb.second.area() <= Model_RigidTemplate::default_minArea)
	continue; // skip undefined BBs
      if(pos_bb.second.visible < 1)
	continue; // skip occluded BBs
	
      Rect_<double> bb = pos_bb.second;
      vector<float> depths = manifoldFn_default(*im,bb);
	
      DetectorResult posDet;
      for(float depth : depths)
      {
	// extract a feature at the candidate depth
	pos_bb.second.depth = depth;
	SparseVector newPos(extractPos(metadata,pos_bb.second));
	if(newPos.size() == 0)
	{
	  string filename = metadata.get_filename();
	  string message = printfpp("warning: unusable positive in %s",filename.c_str());
	  log_once(message);
	}
	else if(train_params.positive_iterations >= 1)
	{
	  if(!posDet || posDet->resp < getLearner().predict(newPos))
	  {
	    posDet.reset(new Detection());
	    // these should all be the same per bb so only one is selected during latent
	    // update phase
	    posDet->depth = qnan; 	  
	    posDet->feature = [newPos](){return newPos;};
	    posDet->BB = pos_bb.second;
	    posDet->pose = metadata.get_pose_name();
	    posDet->resp = getLearner().predict(newPos);
	    posDet->src_filename = metadata.get_filename();
	  }
	}
      }
      // write the best detection
      if(posDet)
	writePos(posDet);
    }
  }
  
  void ExposedLDAModel::collect_training_examples(
    MetaData&metadata,TrainParams train_params,
    function<void (DetectorResult)> writePos, function<void (DetectorResult)> writeNeg,
    DetectionFilter filter)
  {
    filter.testing_mode = false;
    
    //printf("extracting negative examples from frame: %d of %d\n",
    //      iter,(int)train_files.size());
    // load an example
    map<string,AnnotationBoundingBox> pos_bbs = metadata.get_positives();
    map<string,AnnotationBoundingBox> train_bbs = train_params.part_subset?
      train_params.part_subset(pos_bbs):pos_bbs;
    
    shared_ptr<ImRGBZ> im = metadata.load_im();    
    
    if(metadata.use_positives() && train_params.positive_iterations > 0)
    {
      //log_file << printfpp("collecting positives from %s",metadata.get_filename().c_str()) << endl;
      collect_positives(metadata, train_params, writePos, filter,train_bbs,im);    
    }
    if(metadata.use_negatives() && train_params.negative_iterations > 0)
    {
      log_file << printfpp("collecting negatives from %s",metadata.get_filename().c_str()) << endl;
      collect_negatives(metadata, train_params, writeNeg, filter,pos_bbs,train_bbs,im);
    }
  }
  
  ///
  /// SECTION: General training algoritmhs
  ///
  
  void train_parallel_do_write_positive_for_frame(
    ExposedLDAModel&model,
    std::vector< std::shared_ptr< deformable_depth::MetaData > >& training_set,
    vector<DetectionSet >&neg_svs,
    vector<DetectionSet>&pos_svs,
    TrainingStatistics&t_stats, size_t frameIter,
    function<void (bool) > on_new_sv)
  {
    double posMult = 5;
    //double posMult = (pos_ct>0)?neg_ct/static_cast<double>(pos_ct):0;    
    
    for(int posIter = 0; posIter < pos_svs[frameIter].size(); posIter++)
    {
      Detection&curPos = *pos_svs[frameIter][posIter];
      SparseVector pos_ex_add = curPos.feature()*posMult;
      assert(curPos.src_filename != string());
      #pragma omp critical(QP_WRITE)
      if(model.getLearner().write(pos_ex_add,+1,curPos.toString()))
	on_new_sv(true);
    }    
  }
  
  void train_parallel_do_write_negative_for_frame(
    ExposedLDAModel&model,
    std::vector< std::shared_ptr< deformable_depth::MetaData > >& training_set,
    vector<DetectionSet >&neg_svs,
    vector<DetectionSet>&pos_svs,
    TrainingStatistics&t_stats, size_t frameIter,
    function<void (bool) > on_new_sv)
  {
    TaskBlock write_each_neg("train_parallel_do_write");
    for(int negIter = 0; negIter < neg_svs[frameIter].size(); negIter++)
    {
      write_each_neg.add_callee([&,negIter]()
      {
	Detection&curNeg = *neg_svs[frameIter][negIter];
	assert(!std::isnan(curNeg.depth));
	assert(curNeg.src_filename != string());
	SparseVector sparse_feat = curNeg.feature();
	
	// Deva says this is a good assertion to have:
	double model_score = model.getLearner().predict(sparse_feat);
	bool good_scores = (goodNumber(model_score) && goodNumber(curNeg.resp));
	if(!good_scores || ::abs(model_score-curNeg.resp)>1e-4) 
	{
	  log_file << 
	  printfpp("warning model_score(%f) != detector_score(%f)",model_score,curNeg.resp) 
	  << endl;
	  model.debug_incorrect_resp(sparse_feat,curNeg);
	}
	assert(good_scores);
	// because detector is float not double, we can run into numerical issues
	#pragma omp critical(QP_WRITE)
	if(!(model_score <= -1 || curNeg.resp <= -1))
	{
	  if(model.getLearner().write(sparse_feat,-1,curNeg.toString()))
	    on_new_sv(false);
	  t_stats.hard_negatives++;
	}	  
      });
    }
    write_each_neg.execute();    
  }
  
  void train_parallel_do_write(
    ExposedLDAModel&model,
    std::vector< std::shared_ptr< deformable_depth::MetaData > >& training_set,
    vector<DetectionSet >&neg_svs,
    vector<DetectionSet>&pos_svs,
    TrainingStatistics&t_stats
  )
  {
    // write to the QP in serial
    for(int frameIter = 0; frameIter < training_set.size(); frameIter++)
    {
      int new_svs_per_frame = 0;
      auto on_new_sv = [&](bool is_pos)
      {
	t_stats.new_svs++;
	new_svs_per_frame++;
	t_stats.max_new_svs_per_frame = std::max<int>
	  (t_stats.max_new_svs_per_frame,new_svs_per_frame);	
	if(is_pos)
	{
	  string pose = training_set[frameIter]->get_pose_name();
	  t_stats.new_svs_pos++;
	  t_stats.pos_examples_by_pose[pose]++;
	}
	else
	{
	  t_stats.new_svs_neg++;
	}
      };
      
      train_parallel_do_write_positive_for_frame(
	model, training_set, neg_svs, pos_svs, t_stats, frameIter, on_new_sv);

      train_parallel_do_write_negative_for_frame(
	model, training_set, neg_svs, pos_svs, t_stats, frameIter, on_new_sv);
    }
  }
  
  DetectionSet train_parallel_do_collect(
    ExposedLDAModel&model,
    shared_ptr<MetaData> metadata,
    DetectionSet&pos_svs,
    DetectionSet&neg_svs,
    deformable_depth::Model::TrainParams train_params)
  {
    DetectionSet all_svs;
    model.collect_training_examples(
      *metadata, train_params, 
      [&](DetectorResult pos)
      {    
	pos_svs.push_back(pos);
      },
      [&](DetectorResult neg)
      {
	const DetectorResult&cneg = neg;
	all_svs.push_back(cneg);
	if(!cneg->supressed)
	  neg_svs.push_back(cneg);
      }
    );
    
    // sort the negatives for this frame from wrost to best.
    sort(neg_svs.begin(),neg_svs.end(),[](const DetectorResult & lhs, const DetectorResult & rhs)
	  {
	    return lhs->resp < rhs->resp;
	  });    
    
    return all_svs;
  }
  
  TrainingStatistics train_parallel_do(
    ExposedLDAModel&model,
    std::vector< std::shared_ptr< deformable_depth::MetaData > >& training_set, 
    deformable_depth::Model::TrainParams train_params)
  {
    TrainingStatistics t_stats;
    int newSVs = 0;
    
    while(train_params.negative_iterations > 0 || train_params.positive_iterations > 0)
    {
      // collect the support vectors
      TaskBlock collect_svs("train_parallel_do");
      vector<DetectionSet > neg_svs(training_set.size());
      vector<DetectionSet> pos_svs(training_set.size()); 
      vector<Mat> dets_visualiztaions(std::min<int>(training_set.size(),25));
      for(int iter = 0; iter < training_set.size(); iter++)
      {
	collect_svs.add_callee([&,iter]()
	{
	  shared_ptr<MetaData> metadata = training_set[iter];
	  DetectionSet all_svs = train_parallel_do_collect(
	    model,metadata,pos_svs[iter],neg_svs[iter],train_params);
	  // visualize
	  if(train_params.negative_iterations > 0 && iter < dets_visualiztaions.size())
	    dets_visualiztaions[iter] = vertCat(drawDets(*metadata,all_svs,0,0),
						drawDets(*metadata,all_svs,0,1));
	  else if(train_params.positive_iterations > 0)
	    log_once("extracted positives from %s",metadata->get_filename().c_str());
	});
      }
      collect_svs.execute();
      if(train_params.negative_iterations > 0)
      {
	log_im("DetectedSVS",tileCat(dets_visualiztaions));
      }
      
      log_file << "train_parallel_do: Done Collecting Examples, Writting Them" << endl;
      train_parallel_do_write(model,training_set,neg_svs,pos_svs,t_stats);
        
      // re-optimize the QP
      QP*qp = dynamic_cast<QP*>(&model.getLearner());
      if(qp != nullptr)
      {
	if(qp->is_overcapacity())
	  t_stats.cache_overflowed |= true;
	t_stats.total_svs = qp->num_SVs();
      }
      else
	log_once(printfpp("warning: failed to aquire QP"));
      model.update_model();
      
      // show the model
      // WARNING: ASSUMES THREAD SAFETY OF MODEL.SHOW
      thread([&model](){
	log_im("updated model",model.show("model_updated"));
	FileStorage fs(params::out_dir() + "model" + uuid() + ".yml",FileStorage::WRITE);
	fs << "model";model.write(fs);
	fs.release();
      }).detach();
      
      // the loop is complete
      train_params.negative_iterations--;
      train_params.positive_iterations--;
    }
    
    return t_stats;
  }  
  
  void train_parallel(
    ExposedLDAModel&model,
    vector<shared_ptr<MetaData>>&training_set_in,
    Model::TrainParams train_params)
  { 
    vector<shared_ptr<MetaData>> training_set = pseudorandom_shuffle(training_set_in);
    
    // make increasingly large passes over subsamples
    for(int size = 1; size < training_set.size(); size *= 2)
    {
      int iter_count = std::max<int>(1,::sqrt(training_set.size()/size));
      log_file << printfpp("Doing %d iterations of size %d",iter_count,size) << endl;
      for(int iter = 0; iter < iter_count; iter++)
      {
	training_set = pseudorandom_shuffle(training_set);
	printf("Partial Set: %d examples\n",size);
	vector<shared_ptr<MetaData>> training_samples(training_set.begin(),training_set.begin()+size);
	TrainingStatistics stats = train_parallel_do(model,training_samples,train_params);
	printf("Partial Set: Got %d new Support Vectors!!!\n",stats.new_svs);
	log_file << "Partial Set Size " << size << ", Got " 
	  << stats.new_svs << " new support vectors" 
	  << " and " << stats.hard_negatives << " hard negative examples" << endl;
      }
    }
    
    // make two final passes over all the data
    constexpr int IT = 8;
    for(int iter = 0; iter < IT; iter++)
    {
      training_set = pseudorandom_shuffle(training_set);
      printf("Full Set %d of %d\n",iter,IT);
      TrainingStatistics stats = train_parallel_do(model,training_set,train_params);
      printf("Full Set: Got %d new Support Vectors!!!\n",stats.new_svs);
      log_file << "Full Set " << iter << " of " << IT << " : Got " 
	<< stats.new_svs << " new support vectors"
	<< " and " << stats.hard_negatives << " hard negative examples" << endl;
      if(stats.new_svs == 0)
	break;
    }
  }

  static void stats_log_by_pose(TrainingStatistics stats)
  {
    log_file << "stats_log_by_pose" << endl;
    for(auto & pose_positives : stats.pos_examples_by_pose)
      log_file << printfpp("\t%s: %d",
			   pose_positives.first.c_str(),
			   (int)pose_positives.second) << endl;    
  }  
  
  void train_seed_positives(
    ExposedLDAModel& model, 
    vector< shared_ptr< MetaData > >& training_set, 
    Model::TrainParams train_params)
  {
    train_params.positive_iterations = 1;
    train_params.negative_iterations = 0;
    stats_log_by_pose(train_parallel_do(model,training_set,train_params));
  }
  
  static void train_smart_log_before(int batch_size,int proc_frames,int num_frames)
  {
    log_file << printfpp("+train_smart(%d): %d of %d @ %s",
			  batch_size,proc_frames,num_frames,current_time_string().c_str()) << endl;    
  }
  
  static void train_smart_log_after(
    int batch_size,int proc_frames,int num_frames,
    TrainingStatistics&stats,
    vector<shared_ptr<MetaData>>&training_samples)
  {
    log_file << printfpp("-train_smart(%d): %d of %d @ %s",
			  batch_size,proc_frames,num_frames,current_time_string().c_str()) << endl;
    log_file << 
      printfpp("\tnew_svs = %d and hard_negs = %d and max_new_svs_per_frame = %d and total_svs = %d",
	      stats.new_svs,stats.hard_negatives,stats.max_new_svs_per_frame,
	      stats.total_svs) << endl;
    log_file << printfpp("\tframes = %d new_svs_pos = %d new_svs_neg = %d",
      (int)training_samples.size(),(int)stats.new_svs_pos,(int)stats.new_svs_neg) << endl;  

    stats_log_by_pose(stats);
  }
  
  // return break
  static bool train_smart_iteration(
    int&batch_size,
    int&proc_frames,
    int&num_frames,
    int&min_size,
    int&max_size,
    ExposedLDAModel& model,
    vector<shared_ptr<MetaData>>&training_set,
    Model::TrainParams&train_params
 				  )
  {
    // 
    train_smart_log_before(batch_size,proc_frames,num_frames);
    
    // shuffle and subsample to set size
    training_set = pseudorandom_shuffle(training_set);
    vector<shared_ptr<MetaData>> training_samples;
    if(batch_size < training_set.size())
      training_samples = vector<shared_ptr<MetaData>>(training_set.begin(),training_set.begin()+batch_size);
    else
      training_samples = training_set;
    TrainingStatistics stats = train_parallel_do(model,training_samples,train_params);
    
    // 
    train_smart_log_after(batch_size,proc_frames,num_frames,stats,training_samples);

    // early stop condition
    if(batch_size >= training_set.size() && stats.new_svs == 0)
      return true;      
    if(batch_size == max_size && stats.new_svs == 0)
      max_size *= std::min<int>(training_set.size(),2*max_size);
    // increase size
    if(!stats.cache_overflowed && stats.max_new_svs_per_frame < 0.10 * NMAX_HARD_NEG_MINING)
      batch_size = clamp<int>(min_size,2*batch_size,max_size);
    // reduce size
    else if(stats.cache_overflowed || stats.max_new_svs_per_frame >= .5 * NMAX_HARD_NEG_MINING)
      batch_size = clamp<int>(min_size,0.5*batch_size,max_size);      
    proc_frames += batch_size;
    
    // check for manual early stop
    if(boost::filesystem::exists(params::out_dir() + "/manual_stop"))
    {
      log_file << "Manual Stop Command Recived From User" << endl;
      log_file << "Halting Training" << endl;
      return true;
    }
    
    // log profile
    default_pool->print_accounts();    
    
    return false;
  }
  
  // this is a "smart" training algorithm with dynamically adjsuts 
  // the batch size to maximize parallelism while ensuring suffecent numbers of 
  // calls to qp_opt and not wasting CPU time (for shared machnies).
  void train_smart(ExposedLDAModel& model, vector< shared_ptr< MetaData > >& training_set_in, 
		   Model::TrainParams train_params)
  {
    // split the input into positive and negative subsets
    vector< shared_ptr< MetaData > > positive_set;
    vector< shared_ptr< MetaData > > negative_set;
    split_pos_neg(training_set_in,positive_set,negative_set);
        
    // visit the training set 10 times
    int num_frames = params::HARD_NEG_PASSES*negative_set.size();
    int batch_size = 1;
    int proc_frames = 0;
    vector<shared_ptr<MetaData>> training_set = pseudorandom_shuffle(negative_set);
    for(auto & training_example : training_set)
      log_file << "Training_example: " << training_example->get_filename() << endl;
    
    // bounds in the sizes we can switch to
    int min_size = 1;
    int max_size = std::min<int>(training_set.size(),params::cpu_count());    
    
    // have a max time threshold as well.
    // max time for one run
    std::chrono::minutes MAX_TIME = std::chrono::minutes(60*5); // 5 hours = default 
    if(g_params.has_key("MAX_TRAIN_MINS"))
    {
      int min_count = fromString<int>(g_params.get_value("MAX_TRAIN_MINS"));
      log_file << "Using MAX_TRAIN_MINS = " << min_count << endl;
      MAX_TIME = std::chrono::minutes(min_count);
    }
    
    auto start = std::chrono::system_clock::now();
    
    while(proc_frames < num_frames && (std::chrono::system_clock::now() - start < MAX_TIME))
    { 
      // latent update for the positives
      // begin with all positives in the QP cache
      log_file << "Latent Positive Update" << endl;
      train_params.negative_iterations = 0;
      train_params.positive_iterations = 1;
      train_seed_positives(model,positive_set,train_params);      
      
      // mine hard negatives
      log_file << "Hard Negative Mining" << endl;
      train_params.negative_iterations = 1;
      train_params.positive_iterations = 0;
      if(train_smart_iteration(
	batch_size,proc_frames,num_frames,min_size,max_size,model,training_set,train_params))
	break;
    }
  }  
}
