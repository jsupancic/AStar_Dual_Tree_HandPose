/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Eval.hpp"
#include "FauxLearner.hpp"
#include "omp.h"
#include "Faces.hpp"
#include "Log.hpp"

#include "util.hpp"
#include "Semaphore.hpp"
#include "MetaData.hpp"
#include "ThreadPool.hpp"
#include <memory>
#include <boost/graph/graph_traits.hpp>
#include <boost/filesystem/path.hpp>
#include "HandModel.hpp"
#include "ONI_Video.hpp"
#include "ThreadPoolCXX11.hpp"
#include "ThreadPool.hpp"
#include "Video.hpp"
#include "Poselet.hpp"
#include "TestModel.hpp"
#include "MetaDataKITTI.hpp"

namespace deformable_depth
{
  /// SECTION: Constant local parameters
  //string POSE = "ASL:B"; // only one pose
  string POSE = ""; // accept everything
    
  /// SECTION: Evaluate a Model... by training it and then testing it.
  void train_model_one(Model&model,vector<shared_ptr<MetaData> > &examples)
  {
    // parameter config
    Model::TrainParams params;
    params.subset_cache_key = params::target_category();
    params.part_subset = params::defaultSelectFn();
    
    // train with the data
    log_file << "train_model_one " << examples.size() << endl;
    model.train(examples,params);
  }
  
  void seed_data(vector<shared_ptr<MetaData> > train_examples,
		 Rect&BB,shared_ptr<MetaData>&metadata)
  {
    for(int iter = 0; iter < train_examples.size(); ++iter)
    {
      metadata = train_examples[iter];
      BB = params::defaultSelectFn()(metadata->get_positives()).begin()->second;
      if(BB.area() > 0)
	break;
    }
    assert(BB.size().area() != 0);
  }
  
  shared_ptr<Model> train_model(
    const Model_Builder&model_builder)
  {
    // setup params
    vector<shared_ptr<MetaData> > train_examples = default_train_data();    
    
    // first example, this is basically ignored by all the new stuff
    Rect BB;
    shared_ptr<MetaData> metadata;
    seed_data(train_examples,BB,metadata);
    shared_ptr<ImRGBZ> imRGBZ = metadata->load_im();    
    
    // setup the learning and detection machineary
    shared_ptr<Model> model(model_builder.build(BB.size(),imRGBZ->RGB.size()));
          
    // make a few passes over the training data.
    for(int iter = 0; iter < 1; iter++)
      train_model_one(*model,train_examples);
    
    return model;
  }
  
  void eval_pose_estimation(Model&model, string test_dir, Scores&scores)
  {
    log_file << "+eval_pose" << test_dir << endl;
    vector<string> test_stems = allStems(test_dir,".gz",POSE);
    active_worker_thread.V();
    #pragma omp parallel for
    for(int iter = 0; iter < test_stems.size(); iter++)
    {
      active_worker_thread.P();
      // load test example
      shared_ptr<MetaData> metadata = metadata_build(test_dir + test_stems[iter],true);    
      shared_ptr<ImRGBZ> im = metadata->load_im();
      Rect bb_gt = metadata->get_positives()["HandBB"];
      assert(metadata->get_positives().size() == 1);
      
      // detect with the model.
      DetectionFilter filter(-inf,numeric_limits<int>::infinity());
      filter.supress_feature = g_supress_feature;
      DetectionSet detections = model.detect(*im,filter);
      
      // find the best detection within the specified range.
      Detection best_pose_estimate;
      best_pose_estimate.resp = -inf;
      for(auto && current_detecetion : detections)
      {
	bool ol = rectIntersect(current_detecetion->BB,bb_gt) > .5;
	if(ol && current_detecetion->resp > best_pose_estimate.resp)
	  best_pose_estimate = *current_detecetion;
      }
      
      // score poses
      log_file << "gt_pose: " << metadata->get_pose_name() 
	<< " est_pose: " << best_pose_estimate.pose << endl;
      if(best_pose_estimate.pose == metadata->get_pose_name())
	scores.pose_correct++;
      else
	scores.pose_incorrect++;
      active_worker_thread.V();
    }
    active_worker_thread.P();
    log_file << "-eval_pose" << test_dir << endl;
  }
  
  vector<string> real_dirs()
  {
    return vector<string>
    {
      "data/James_Test/","data/Sam_Test/",
      "data/Golnaz_Test/","data/Yi_Test/",
      "data/Vivian_Test/","data/Xiangxin_Test/",
      "data/Bailey_Test/","data/Raul_Test/"
    };
  }
    
  vector<string> default_test_dirs()
  {
    //return vector<string>{ "data/HighResFace/"};
    
    vector<string> dirs =
      { 
	"data/James_Train/",
	"data/James_Test/","data/Sam_Test/",
	"data/Golnaz_Test/","data/Yi_Test/",
	"data/Vivian_Test/","data/Xiangxin_Test/",
	"data/Bailey_Test/","data/Raul_Test/",
	"data/clutter_test/","**TRAIN**",
	"data/HighRes"
      }; 
	
    std::reverse(dirs.begin(),dirs.end());
    return dirs;
  }
  
  string default_train_dir()
  {
    return string("data/James_Train/");
  }
 
  vector< string > default_train_dirs()
  {
    string training_set_name = 
      g_params.has_key("DATASET")?g_params.get_value("DATASET"):"synth";
    
    if(training_set_name == "real")
    {
      return vector<string>{default_train_dir()};
    }
    else if(training_set_name == "synthUB")
    {    
      // synthetic upper bound set.
      return vector<string>{
	default_train_dir(),
	"data/SynthA/",
	"data/SynthB/",
	"data/SynthD/",
	"data/SynthF/",
	"data/SynthO/",
	"data/SynthQ/",
	"data/SynthU/",
	"data/SynthV/",
	"data/SynthW/",
	"data/SynthY/"
      };      
    }
    else if(training_set_name == "synth")
    {
      // synthetic standard set
      vector<string> set{
	params::synthetic_directory()
      };
      
      if(g_params.get_value("NEG_SET") == "BIG")
      {
	log_file << "NegSet == B" << endl;
	vector<string> v_real_dirs = real_dirs();
	set.push_back(default_train_dir());
	set.insert(set.end(),v_real_dirs.begin(),v_real_dirs.end());
      }
      else
      {
	log_file << "NegSet == S" << endl;
	set.push_back(default_train_dir());
      }
      
      return set;
    }
    else
      assert(false);
  }
    
  vector< shared_ptr< MetaData > > load_dirs(vector< string > dirs, bool only_valid)
  {
    return metadata_build_all(dirs,true,only_valid);
  }
    
  void eval_log_result(string testDir, Scores score)
  {
    log_file << testDir << 
      " p = " << score.p(-inf) << 
      " r = " << score.r(-inf) << 
      " f1= " << score.f1(-inf) << 
      " pose = " << score.pose_accuracy() << endl;
  }
        
  void write_best_det(FileStorage&fs,string filename,
		      DetectorResult&det,DetectorResult&closest,bool correct)
  {
    // critical section
    static mutex m; unique_lock<mutex> l(m);
    static atomic<int> id(0);
    
    if(det == nullptr)
    {
      det.reset(new Detection());
      correct = false;
    }
    
    if(det != nullptr)
    {
      BestDetection bestDet{filename,correct,det};
      fs << printfpp("BestDet%d",(int)id) << bestDet;
    }
    
    if(closest != nullptr)
    {
      BestDetection closesetDet{filename,true,closest};
      fs << printfpp("CloseDet%d",(int)id) << closesetDet;
    }
    
    id++;
  }
    
  vector<Scores> eval_on_dirs(shared_ptr<Model>model, vector<string> testDirs)
  {
    // setup parameters
    vector<Scores> scores;
    map<string,Scores> scoresByPose;

    // default values
    scores = vector<Scores>(testDirs.size());
              
    // open the output file
    FileStorage best_detections(params::out_dir() + "best_detections.yml",FileStorage::WRITE);    
    
    TaskBlock test_each_directory("eval_model");
    for(int iter = 0; iter < testDirs.size(); iter++)
    {
      test_each_directory.add_callee([&,iter]()
      {
	test_model(*model,testDirs[iter],scores[iter],
		   scoresByPose,
	    [&](string filename,DetectorResult&det,DetectorResult&closest,bool correct)
	{
	  write_best_det(best_detections,filename,det,closest,correct);
	});
	//if(POSE == "")
	  //eval_pose_estimation(*model,testDirs[iter],scores[iter]);	
      });
    }
    test_each_directory.execute();
    
    // release the output directory
    best_detections.release();
    
    // log the results
    for(int iter = 0; iter < testDirs.size(); iter++)
      eval_log_result(testDirs[iter],scores[iter]);
    // by pose
    for(auto pose_score : scoresByPose)
      eval_log_result(pose_score.first,pose_score.second); 
    // total
    eval_log_result("TOTAL",Scores(scores));    
    
    return scores;
  }
    
  vector<Scores> eval_model(int argc, char**argv,const Model_Builder&model_builder)
  {   
    // train the model
    shared_ptr<Model> model = train_model(model_builder);
    
    // save the model.
    if(g_params.option_is_set("WRITE_MODEL"))
    {
      FileStorage save_model(params::out_dir() + "/model.yml",FileStorage::WRITE);
      save_model << "model";
      bool model_wrote = model->write(save_model);
      if(!model_wrote)
	log_file << "warning: failed to write model!" << endl;
      save_model.release();
    }

    // show the model
    log_im("final_model",model->show("final model"));
    
    //
    // testing phase
    //
    
    // test on video
    if(g_params.has_key("TEST_VIDEO"))
      test_model_oni_video(*model);       

    // compute training error
    //auto result = eval_on_dirs(model,
      //vector<string>{params::synthetic_directory()});    
    
    // test on images
    if(g_params.has_key("TEST_IMAGES"))
      test_model_images(model);
    
    // test on directories
    //auto result = eval_on_dirs(model,testDirs = default_test_dirs(););
    //return reuslt;
    return vector<Scores>();
  }
  
  Model_HyperParam_Setting::Model_HyperParam_Setting(double C, double AREA, Scores score) : 
    C(C), AREA(AREA), score(score)
  {
  }
  void Model_HyperParam_Setting::print()
  {
    printf("C = %f\tAREA = %f\tp = %f\tr = %f\n",
	    (float)C,(float)AREA,(float)score.p(),(float)score.r());
  };
  bool Model_HyperParam_Setting::operator<(const Model_HyperParam_Setting&other) const
  {
    return score.f1() < other.score.f1();
  }
  Scores& Model_HyperParam_Setting::getScore()
  {
    return score;
  }  
  
  void tune_model_one(FileStorage&file,
		      vector<Model_HyperParam_Setting>&settings,
		      double C, int AREA
 		    )
  {
    Scores score;
    // check if we can retrieve the result.
    ostringstream memo_name;
    bool score_read = false;
    #pragma omp critical
    {
      memo_name << "cache/HyperParamsC=" << C <<",AREA=" << AREA << ".yml";
      file.open(memo_name.str(),FileStorage::READ);
      if(file.isOpened())
      {
	// load the file
	file["Score"] >> score;
	file.release();
	score_read  = true;
      }
    }
    if(!score_read)
    {
      // compute the result for this setting
      //OneFeatureModelBuilder builder(C,AREA,RGB_FACT());
      OneFeatureModelBuilder builder(C);
      score = Scores(eval_model(0,NULL,builder));
    }
    
    Model_HyperParam_Setting current(C,AREA,score);
    current.print();
    #pragma omp critical
    settings.push_back(current);
    
    // save result
    #pragma omp critical
    {
      file.open(memo_name.str(),FileStorage::WRITE);
      file << "Score" << current.getScore();
      file.release();    
    }
  }
  
  void tune_model(int argc, char**argv)
  {    
    // memoization
    FileStorage file;
    
    // iteration parameters
    vector<double> Cs;
    for(double C = .001; C <= 10000; C *= 10)
      Cs.push_back(C);
    
    // search for the optimal results
    vector<Model_HyperParam_Setting> settings;
    for(int C_iter = 0; C_iter < Cs.size(); C_iter++)
    {
      double C = Cs[C_iter];
      //for(double AREA = 4*4; AREA <= 16*16; AREA *= 2)
      //{
	tune_model_one(file,settings,C,0/*AREA*/);
      //}
    }
      
    // dump the results.
    for(Model_HyperParam_Setting&setting : settings)
    {
      setting.print();
    }
    std::sort(settings.begin(),settings.end());
    std::reverse(settings.begin(),settings.end());
    printf("===== Best Setting For Active Model ====\n");
    if(settings.size() > 0)
      settings[0].print();
    else
      printf("error: no settings results available!\n");
  }
  
  ///
  /// SECTION: BestDetection
  ///
  void write(cv::FileStorage&fs, std::string&, const deformable_depth::BestDetection& bestDet)
  {
    fs << "{";
    fs << "filename" << bestDet.filename;
    fs << "correct" << bestDet.correct;
    fs << "detection" << *bestDet.detection;
    fs << "}";
  }
  
  void read(const FileNode& fs, BestDetection& bestDet, BestDetection )
  {
    fs["correct"] >> bestDet.correct;
    fs["filename"] >> bestDet.filename;
    bestDet.detection.reset(new Detection());
    fs["detection"] >> *bestDet.detection;
  }
}
