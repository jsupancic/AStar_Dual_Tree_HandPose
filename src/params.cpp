/**
 * Copyright 2012: James Steven Supancic III
 **/
#include "params.hpp"
#include "util.hpp"
#include "Log.hpp"
#include <sstream>
#include <iostream>
#include <float.h>
#include "Eval.hpp"
#include "MetaDataKITTI.hpp"
#include "QP.hpp"
#include <boost/algorithm/string/trim.hpp>
#include "TestModel.hpp"
#include "Video.hpp"
#include "ICL_MetaData.hpp"
#include "KinectPose.hpp"
#include "ApxNN.hpp"
#include "MiscDataSets.hpp"
#include "FLANN.hpp"
#include "ExternalModel.hpp"
#include "AONN.hpp"
#include "NYU_Hands.hpp"
#include "EgoCentric.hpp"
#include "VolumetricNN.hpp"
#include "ExportModel.hpp"

namespace deformable_depth
{
  using namespace std;
  
  Params g_params;
    
  /// KNOWN Options
  /// JOINT_C
  /// NUM_CPUS
  /// MAX_TRAIN_MINS
  /// DATASET
  /// OBJ_DEPTH
  /// STRIDE
  /// SKIN_FILTER
  /// MANIFOLD_FN
  /// HEAP_PROF
  /// CFG_FILE
  void Params::parse_token(string token)
  {
    // strip after the comment
    auto comment =  token.find('#');
    if(comment != string::npos)
      token = string(token.begin(),token.begin() + comment);
    boost::algorithm::trim(token);

    // validate token first
    if(token == "")
      return;
    auto eq = find(token.begin(),token.end(),'=');
    if(eq == token.end() || eq + 1 == token.end() || eq == token.begin())
      return;
    
    // add token
    std::ostringstream key, value;
    std::copy(token.begin(),eq,std::ostream_iterator<char>(key));
    std::copy(eq+1,token.end(),std::ostream_iterator<char>(value));
    string skey = key.str();
    boost::algorithm::trim(skey);
    while(has_key(skey))
      skey += uuid();
    string svalue = value.str();
    boost::algorithm::trim(svalue);
    m_params[skey] = svalue;
  }
  
  void Params::parse(int argc, char** argv)
  {
    // parse the arguments
    for(int iter = 0; iter < argc; iter++)
    {
      string cur_arg(argv[iter]);
      parse_token(cur_arg);
    }
    
    // parse any includeds
    bool included_a_file;
    do
    {
      map< string, string > cfg_files = matching_keys("(CFG_FILE.*)|(INCLUDE.*)");
      included_a_file = false;
      for(auto && cfg_file : cfg_files)
      {
	// don't allow double inclusion...
	if(included_files.find(cfg_file.second) == included_files.end())
	{
	  included_a_file = true;
	  included_files.insert(cfg_file.second);
	  log_once(safe_printf("included file %",cfg_file.second));

	  // actually include the file
	  ifstream ifs(cfg_file.second);
	  while(ifs)
	  {
	    string line; getline(ifs,line);
	    parse_token(line);
	  }
	}
      }
    } while(included_a_file);
  }
  
  map< string, string > Params::matching_keys(string RE)
  {
    map<string,string> matches;
    
	for(map<string,string>::iterator iter = m_params.begin(); 
		iter != m_params.end(); ++iter)
	{
      if(boost::regex_match(iter->first,boost::regex(RE)))
		matches.insert(*iter);
	} 
    return matches;
  }

  set<string> Params::matching_values(string RE)
  {
    set<string> matches;

    for(auto && pair : matching_keys(RE))
      matches.insert(pair.second);

    return matches;
  }
  
  void Params::log_params() 
  {
    // print the arguments
    for(map<string,string>::iterator iter = m_params.begin(); iter != m_params.end(); ++iter)
    {
      log_file << "parsed: " << iter->first << "=" << iter->second << endl;
      cout << "parsed: " << iter->first << "=" << iter->second << endl;
    }
  }
  
  string params::synthetic_directory()
  {
    //return "/home/jsupanci/workspace/deformable_depth/out/";
    return g_params.require("SYNTHETIC_DIRECTORY");
    //return "/home/jsupanci/workspace/data/hand/synth-2014.6.23-25000/";
    //return "/home/jsupanci/workspace/deformable_depth/data/SynthUp/";
  }

  string Params::get_value(const string key, string def) const
  {
    if(m_params.find(key) != m_params.end())
      return m_params.at(key);
    else
      return def;
  }

  bool Params::option_is_set(string name) const
  {
    if(has_key(name))
    {
      string value = get_value(name);
      return value.find("F") == string::npos and value.find("f") == string::npos;
    }
    else
      return false; // default options to false
  }
  
  string Params::require(const string key) const
  {
    if(!this->has_key(key))
    {
      cout << "You Failed to provide: " << key << endl;
      cout << "Aborting..." << endl;
    }
    assert(this->has_key(key));
    return this->get_value(key);
  }

  bool Params::has_key(const string key) const
  {
    return m_params.find(key) != m_params.end();
  }
  
  string params::KITTI_dir()
  {
    return "/mnt/big/jsupanci/workspace/KITTI/data/";
  }
  
  namespace params
  {
    // e.g. SupervisedMixtureModel_Builder, MultiFeatureModelBuilder 
    //      RigidModelBuilder, BlobModelBuilder, OneFeatureModelBuilder
    //      OneFeatureModelBuilder, RSABSVM_ModelBuilder,
    //     HandFingerTipModel_Builder SRFModel_Model_Builder
    //     ERFModel_Model_Builder, ExternalModel_Builder
    //     StackedModel_Builder, PZPS_Builder, ApxNN_Builder
    typedef ApxNN_Builder Default_Model_Builder;

    const deformable_depth::Model_Builder& model_builder()
    {
      string model_id = g_params.require("MODEL");
      if(model_id == "ApxNN")
      {
	static ApxNN_Builder builder;
	return builder;
      }
      else if(model_id == "KinectPose" or model_id == "Keskin")
      {
	static TrivialModelBuilder<KeskinsModel> builder;
	return builder;
      }
      else if(model_id == "Xu")
      {
	static TrivialModelBuilder<XusModel> builder;
	return builder;
      }
      else if(model_id == "FLANN")
      {
	static TrivialModelBuilder<FLANN_Model> builder;
	return builder;
      }
      else if(model_id == "HoughForest")
      {
	static SRFModel_Model_Builder builder;
	return builder;
      }
      else if(model_id == "KinectSegPose")
      {
	static TrivialModelBuilder<KinectSegmentationAndPose_Model> builder;
	return builder;
      }
      else if(model_id == "AONN")
      {
	static TrivialModelBuilder<AONN_Model> builder;
	return builder;
      }
      else if(model_id == "DeepYi")
      {
	static TrivialModelBuilder<DeepYiModel> builder;
	return builder;
      }
      else if(model_id == "NYUModel")
      {
	static TrivialModelBuilder<NYU_Model> builder;
	return builder;
      }
      else if(model_id == "Human")
      {
	static TrivialModelBuilder<HumanModel> builder;
	return builder;
      }
      else if(model_id == "NOP")
      {
	static TrivialModelBuilder<NOP_Model> builder;
	return builder;
      }
      else if(model_id == "Kitani")
      {
	static TrivialModelBuilder<KitaniModel> builder;
	return builder;
      }
      else if(model_id == "VolumetricNNModel" or model_id == "VolNN")
      {
	static TrivialModelBuilder<VolumetricNNModel> builder;
	return builder;
      }
      else if(model_id == "Export")
      {
	static TrivialModelBuilder<ExportModel> builder;
	return builder;
      }
      else
	throw std::runtime_error("bad MODEL");
    }

    int max_examples()
    {
      int me = -1;
      if(me > 0)
	return me;
      
      me = 
	fromString<int>(g_params.require("NPOS")) + 
	fromString<int>(g_params.require("NNEG"));
      return me;
    }    

    float obj_depth()
    {
      static float depth = qnan;
      if(!goodNumber(depth))
	depth = fromString<double>(g_params.require("OBJ_DEPTH"));
      return depth;
    }

    double pyramid_sharpness()
    {
      static float sharpness = qnan;
      if(!goodNumber(sharpness))
	sharpness = fromString<double>(g_params.require("NN_PYRAMID_SHARPNESS"));
      return sharpness;
    }    

    // for skin detection histograms.
    int channels[] = {0,1,2};
    float colRange[] = {0,255};
    const float* ranges[] = {colRange, colRange, colRange};
    
    // for deriviative filtering
    float dervFilter[] = {-1,0,1};
    cv::Mat dxFilter(1,3,CV_32F,dervFilter);
    cv::Mat dyFilter(3,1,CV_32F,dervFilter);
    // an alterantative
    float filter01[] = {-1,1};
    cv::Mat dxFilter01(1,2,CV_32F,filter01);
    cv::Mat dyFilter01(2,1,CV_32F,filter01);
    
    double finger_correct_distance_threshold()
    {
      static float correct_dist_threshold = qnan;
      if(!goodNumber(correct_dist_threshold))
      {
	if(g_params.has_key("FINGER_DIST_THRESH"))
	  correct_dist_threshold = fromString<double>(g_params.get_value("FINGER_DIST_THRESH"));
	else
	  correct_dist_threshold = 10;
      }
      return correct_dist_threshold;
    }
    
    int cpu_count()
    {
      return g_params.has_key("NUM_CPUS")?
	fromString<int>(g_params.get_value("NUM_CPUS")):
	28;
    }
    
    string out_dir()
    {
      if(g_params.has_key("OUT_DIR"))
      {
	string out_dir = g_params.get_value("OUT_DIR");
	if(out_dir == "automatic")
	{
	  static shared_ptr<string> dir_path;
	  if(!dir_path)
	  {
	    // calculate the directory path
	    //dir_path.reset(new string(string ("outs/") + uuid() + "/"));
	    std::time_t rawtime; std::time(&rawtime);
	    std::tm timeinfo; localtime_r(&rawtime,&timeinfo);
	    char buffer[80]; std::strftime(buffer,80,"%Y-%m-%d-%H-%M-%S",&timeinfo);
	    dir_path.reset(new string(string ("outs/") + buffer + "/"));
	  }
	  log_file << "dir_path = " << *dir_path << endl;
	  boost::filesystem::create_directory(*dir_path);
	  return *dir_path;
	}
	else
	  return out_dir;
      }
      else
	return "./out/";
    }

    string cache_dir()
    {
      return "./cache/";
    }
    
    bool use_skin_filter()
    {
      return g_params.has_key("SKIN_FILTER");
    }
    
    int video_annotation_stride()
    {
      int stride = -1;
      if(g_params.has_key("STRIDE"))
	stride = fromString<int>(g_params.get_value("STRIDE"));
      else
	stride = 20;
      log_once(safe_printf("Video stride = %",stride));
      return stride;
    }

    //DD: 10,40;
    // KITTI: 50
    constexpr static float default_MIN_Z = 10; 
    //DD: 150 is good; 100 is to small for new vids... 250 to big;
    // KITTI: 80m = 8000
    constexpr static float default_MAX_Z = 150; 
    
    float MAX_X()
    {
      return +100;
    }

    float MIN_X()
    {
      return -100;
    }

    float MAX_Y()
    {
      return +100;
    }

    float MIN_Y()
    {
      return -100;
    }

    float MAX_Z()
    {
      static float max_z = qnan;
      if(goodNumber(max_z))
	return max_z;

      if(g_params.has_key("MAX_Z"))
	max_z = fromString<double>(g_params.get_value("MAX_Z"));
      if(g_params.has_key("DISP_DEPTH"))
	max_z = 999;
      else
	max_z = default_MAX_Z;
      return max_z;
    }

    float MIN_Z()
    {
      return default_MIN_Z;
    }
    
    TrainingBBSelectFn makeSelectFn(string regex_pattern)
    {
      return [regex_pattern](const map<string,AnnotationBoundingBox>&all_pos)
      {
	map<string,AnnotationBoundingBox> subset;
	boost::regex pedestrainRE(regex_pattern,boost::regex::icase);
	
	for(auto & label : all_pos)
	  if(boost::regex_match(label.first,pedestrainRE))
	    subset[label.first] = label.second;
	  else
	  {
	    //cout << "KITTI didn't match target: " << label.first << endl;
	  }
	
	return subset;
      };
    }
    
    string target_category()
    {
      if(g_params.has_key("TARGET_CAT"))
	return g_params.get_value("TARGET_CAT");
      else
	return "Pedestrian";
    }

    float min_image_area()
    {
      return 25*25;
    }
    
    TrainingBBSelectFn defaultSelectFn()
    {
      if(g_params.get_value("DATASET") == "KITTI")
      {
	return makeSelectFn(string(".*") + target_category() + string(".*"));
      }
      else
	return [](const map<string,AnnotationBoundingBox>&all_pos)
	  {
	    map<string,AnnotationBoundingBox> result;
	    if(all_pos.find("HandBB") != all_pos.end())
	      result["HandBB"] = all_pos.at("HandBB");
	    return result;
	  };
    }
    
    double C()
    {
      return g_params.has_key("JOINT_C")?
	fromString<double>(g_params.get_value("JOINT_C")):1;
    }
  }
    
  // choose between various datasets
  static vector<shared_ptr<MetaData> > do_default_train_data()
  {
    vector<shared_ptr<MetaData> > train_examples;
    string dataset_name = g_params.require("DATASET");
    if(dataset_name == "synth")
    {
      vector<string> train_dirs = default_train_dirs();
      for(string train_dir : train_dirs)
	log_file << "default_train_data: " << train_dir << endl;
      train_examples = metadata_build_all(train_dirs,true);      
    }
    else if(dataset_name == "ICL")
    {
      train_examples = ICL_training_set();
    }
    else if(dataset_name == "test_videos")
    {
      for(string vid_filename : test_video_filenames())
      {
	shared_ptr<Video> pvideo = load_video(vid_filename);
	for(int frameIter = 0; frameIter < pvideo->getNumberOfFrames(); frameIter += params::video_annotation_stride())
	{
	  shared_ptr<MetaData> datum = pvideo->getFrame(frameIter,true);
	  if(datum)
	    train_examples.push_back(datum);
	}
      }
    }
    else if(dataset_name == "EGOCENTRIC_SYNTH")
    {
      train_examples = egocentric_dir_data();
    }
    else if(dataset_name == "NYU")
    {
      train_examples = NYU_training_set();
    }
    else if(dataset_name == "NONE")
    {}
    else if(dataset_name == "DIRECTORIES")
    {
      // get the directories from some configuration file.
      vector<string> train_dirs;
      for(auto && dir : g_params.matching_values(".*DIRECTORIES.*"))
      {
	log_file << "got training dir = " << dir << endl;
	train_dirs.push_back(dir);
      }
      // execute the loading.
      train_examples = metadata_build_all(train_dirs,true);
    }
    else if(dataset_name == "Egocentric_Synth_Poser")
    {
      return EgoCentric_Poser_training_set();
    }
    else
    {
      log_once("failure: bad dataset_name");
      assert(false);
    }

    // LR flips?
    if(g_params.get_value("TRAINING_INSERT_LR_FLIPS") == "TRUE")
      metadata_insert_lr_flips(train_examples);
        
    if(g_params.has_key("CHEAT_HAND_LR"))    
    {
      vector<shared_ptr<MetaData> > keep_examples;
      if(g_params.get_value("CHEAT_HAND_LR") == "RIGHT")
      {
	for(auto && ex : train_examples)
	  if(!ex->leftP())
	    keep_examples.push_back(ex);
      }
      else if(g_params.get_value("CHEAT_HAND_LR") == "LEFT")
      {
	for(auto && ex : train_examples)
	  if(ex->leftP())
	    keep_examples.push_back(ex);
      }
      train_examples = keep_examples;
    }    

    return train_examples;
  }

  vector<shared_ptr<MetaData> > default_train_data()
  {
    if(g_params.get_value("DATASET") == "KITTI")
      return KITTI_default_train_data();

    static mutex m; unique_lock<mutex> l(m);
    
    static vector<shared_ptr<MetaData> > train_examples;
    if(train_examples.size() == 0)
    { 
      log_file << "loading training data" << endl;
      train_examples = do_default_train_data();

      if(g_params.option_is_set("COUNT_LR"))
      {
	atomic<long> left_count(0), right_count(0);
	TaskBlock count_lr("count_lr");
	for(auto & datum : train_examples)      
	{
	  count_lr.add_callee([&,datum]()
			      {
				auto printFn = [&](){return safe_printf( "default_train_data: %",datum->get_filename());};
				log_text_decay_freq("default_train_data",printFn);
				if(datum->leftP())
				  left_count++;
				else
				  right_count++;
			      });
	}
	count_lr.execute();
	log_file << safe_printf("training set lr slipt = (%,%)",left_count++,right_count++) << endl;
      }
      
      train_examples = pseudorandom_shuffle(train_examples);
      log_file << "training data has been loaded" << endl;    
    }               
    return train_examples;
  }
  
  vector< shared_ptr< MetaData > > default_test_data()
  {
    if(g_params.get_value("DATASET") == "KITTI")
      return KITTI_default_test_data();    
    
    static mutex m; unique_lock<mutex> l(m);
    
    static vector<shared_ptr<MetaData> > test_examples;
    if(test_examples.size() == 0)
    { 
      test_examples = metadata_build_all(default_test_dirs(),true);
      metadata_insert_lr_flips(test_examples);
    }
    cout << "testing data has been loaded" << endl;    
    return test_examples;
  }
}
