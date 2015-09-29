/**
 * Copyright 2013: James Steven Supancic III
 **/
#include "MetaData.hpp"
#include "util_file.hpp"
#include "PXCSupport.hpp"
#include "Orthography.hpp"
#include "MetaData_Pool.hpp"

#include <boost/filesystem.hpp>
#include "util.hpp"
#include "util_depth.hpp"
#include "util_file.hpp"
#include <boost/regex.hpp>
#include "Semaphore.hpp"
#include "Log.hpp"
#include "Faces.hpp"
#include <leveldb/db.h>
#include "RegEx.hpp"

#ifndef WIN32
#include "LibHandSynth.hpp"
#include "HandSynth.hpp"
#include "ThreadPool.hpp"
#include "ThreadPoolCXX11.hpp"
#include "Cache.hpp"
#include "MetaDataKITTI.hpp"
#endif


namespace deformable_depth
{
  ///
  /// SECTION: Metadata Factories
  ///
  void split_pos_neg(
    const vector< shared_ptr< MetaData > >& training_set, 
    vector< shared_ptr< MetaData > >& positive_set, 
    vector< shared_ptr< MetaData > >& negtive_set)
  {
	for(int iter = 0; iter < training_set.size(); ++iter)
    {
      if(training_set[iter]->use_positives())
		positive_set.push_back(training_set[iter]);
      if(training_set[iter]->use_negatives())
		negtive_set.push_back(training_set[iter]);
    } 
  }
  
  vector< shared_ptr< MetaData > > filter_for_pose(
    const vector< shared_ptr< MetaData > >& set, const string& pose)
  {
    string re = printfpp(".*%s.*",pose.c_str());
    boost::regex pose_re(re);
    
    vector<shared_ptr<MetaData> > result;
    
    for(int iter = 0; iter < set.size(); ++iter)
    {
      if(boost::regex_match(set[iter]->get_pose_name(),pose_re))
      {
	result.push_back(set[iter]);
      }
    }
    
    return result;
  }
  
  vector<shared_ptr<MetaData> > filterLeftOnly(vector<shared_ptr<MetaData> >&training_set)
  {
    vector<shared_ptr<MetaData> > left_only;
	for(int iter = 0; iter < training_set.size(); ++iter)
	{
      if(training_set[iter]->leftP())
		left_only.push_back(training_set[iter]);
	}
    return left_only;
  }  
  
  vector< shared_ptr< MetaData > > filterRightOnly(vector< shared_ptr< MetaData > >& training_set)
  {
    vector<shared_ptr<MetaData> > right_only;
	for(int iter = 0; iter < training_set.size(); ++iter)
	{
      if(!training_set[iter]->leftP())
		right_only.push_back(training_set[iter]);
	}
    return right_only;
  }
  
  shared_ptr< MetaData > metadata_build_with_cache_build_one(string filename, bool read_only)
  {
    return std::make_shared<MetaData_Pooled>(filename,read_only);
  }
  
  shared_ptr< MetaData > metadata_build_with_cache(string filename, bool read_only)
  {
#ifdef WIN32
	  return metadata_build_with_cache_build_one(filename,read_only);
	  // WIN32 doesn't support caching
#else
    static Cache<shared_ptr<MetaData> > cache;    
        
    shared_ptr<MetaData> datum = cache.get(filename,[&](){return metadata_build_with_cache_build_one(filename,read_only);});
    //bool is_left = datum->leftP();
    log_once(safe_printf("metadata_build_with_cache: % % ",filename.c_str(),(int)cache.size()));

    return datum;
#endif
  }
  
  shared_ptr< MetaData > metadata_build(string filename, bool read_only, bool only_valid)
  {
    shared_ptr<MetaData> metadata;
    
    // load the metadata object from the file
    try
    {
#ifdef WIN32
	 metadata = metadata_build_with_cache_build_one(filename,read_only);
#else
      metadata = metadata_build_with_cache(filename,read_only);
#endif
    }
    catch(cv::Exception e)
    {
      cout << "warning: metadata_build bad file corruption: " << filename << endl;
      log_file << "warning: metadata_build bad file corruption: " << filename << endl;      
      return nullptr;
    }
    if(!metadata)
      return nullptr;
      
    // Only return examples within a valid depth range
    if(metadata->use_positives() && only_valid)
    {
      // check cache
      if(metadata->hasAnnotation("valid"))
	if(metadata->getAnnotation("valid") == "0")
	  return nullptr;
	else
	  return metadata;

      // compute validity
      auto poss = metadata->get_positives();
      shared_ptr<ImRGBZ> im = metadata->load_im();
      for(map<string,AnnotationBoundingBox>::iterator iter = poss.begin();
		  iter != poss.end(); ++iter)
      {
	Rect_<double> bb = iter->second;
	bool valid_bb = 
	  0 <= bb.x && 0 <= bb.width && bb.x + bb.width <= im->cols() && 
	  0 <= bb.y && 0 <= bb.height && bb.y + bb.height <= im->rows();
	if(!valid_bb)
	{
	  cout << "metadata_build bad bb: " << filename << bb << " " << iter->first << endl;
	  log_file << "metadata_build bad bb: " << filename << bb << " " << iter->first << endl;
	  metadata->putAnnotation("valid","0");
	  return nullptr;
	}
	
	double z = extrema(im->Z(bb)).min;
	log_file << "z_min = " << z << endl;
	if(z >= params::MAX_Z())
	{
	  cout << "metadata_build bad depth: " << filename << endl;
	  log_file << "metadata_build bad depth: " << filename << endl;
	  metadata->putAnnotation("valid","0");
	  return nullptr;
	}
      }

      metadata->putAnnotation("valid","1");
    }
     
    log_text_decay_freq("metadata_build accepted", [&]() -> string
			{
			  return string("info: metadata_build accepted: ") + filename;
			});
#ifndef WIN32
    //FaceDetector::build_cache(*metadata->load_im());
#endif
    return metadata;
  }
  
#ifdef DD_CXX11
  vector<shared_ptr< MetaData > > metadata_build_training_set(
    vector< string > train_files, 
    bool read_only,
    function<bool(MetaData&)> filter)
  {
    // load all training examples
    vector<shared_ptr<MetaData> > train_data;
    TaskBlock block("metadata_build_training_set");
    for(int iter = 0; iter < train_files.size(); iter++)
    {
      block.add_callee([&,iter]()
      {
	// alloc the metadatas for this file
	string train_file = train_files[iter];
	shared_ptr<MetaData> metadata = 
	  metadata_build(train_file,read_only);
	if(!metadata)
	  return;
	shared_ptr<MetaData> metadata_flip = 
	  metadata_build_with_cache(train_file + "mirrored",read_only);
	
	// synchrnoize insert them into the list
	static mutex m; unique_lock<mutex> l(m);
	train_data.push_back(metadata);
	train_data.push_back(metadata_flip);	
      }
      );
    }
    block.execute();
    
    vector<shared_ptr<MetaData> > filtered_result;
	for(size_t idx = 0; idx < train_data.size(); idx++)
	{
		shared_ptr<MetaData>&metadata = train_data[idx];
		if(filter(*metadata))
			filtered_result.push_back(metadata);
	}

    return filtered_result;
  }
#endif
  
  void metadata_insert_lr_flips(vector< shared_ptr< MetaData > >& examples)
  {
    int orig_size = examples.size();
    TaskBlock lr_flip_insertion("lr_flip_insertion");
    int size0 = examples.size();
    for(int iter = 0; iter < orig_size; ++iter)
    {
      examples.push_back(shared_ptr<MetaData>());
      lr_flip_insertion.add_callee([&,iter,size0]()
      {
	shared_ptr<MetaData> datum = examples[iter];
	string filename = datum->get_filename()+"mirrored";
	bool use_positives = datum->use_positives();
	bool use_negatives = datum->use_negatives(); 
	MetaData_Pooled::AllocatorFn fn = [datum]() -> MetaData*
	{
	  return new MetaData_LR_Flip(datum);
	};

	examples[iter+size0] = make_shared<MetaData_Pooled>(
	  fn,use_positives,use_negatives,filename);
      });
    }
    lr_flip_insertion.execute();
  }    

  // build a single directory
  vector<shared_ptr<MetaData>> metadata_build_all(string dir, bool only_valid, bool read_only)
  {
    static atomic<int> built_examples(0);

    vector<shared_ptr<MetaData> > result;
	vector<string> filenames = allStems(dir,".gz");

#ifdef DD_CXX11
    TaskBlock build_metadata("metadata_build_all");
    for(string&filename : filenames)
    {
      build_metadata.add_callee([&,filename]()
      {
	shared_ptr<MetaData> cur_data = metadata_build(dir + "/" + filename,read_only,only_valid);
	if(cur_data)
	{
	  built_examples++;
	  if(built_examples++ <= params::max_examples())
	  {
	    static mutex m; unique_lock<mutex> l(m);
	    result.push_back(cur_data);
	  }
	}
      });
    }
    build_metadata.execute();  
#else
	for(size_t idx = 0; idx < filenames.size(); idx++)
	{
		shared_ptr<MetaData> cur_data = metadata_build(
		  dir + get_path_seperator() + filenames[idx],read_only);
		if(cur_data)
		  result.push_back(cur_data);
	}
#endif


    return result;
  }     
  
  // build a set of directires
  vector< shared_ptr< MetaData > > metadata_build_all(vector< string > dirs, bool read_only, bool only_valid)
  {
    vector< shared_ptr< MetaData > > result;
    
	for(size_t idx = 0; idx < dirs.size(); idx++)
    {
      string dir = dirs[idx];
      vector< shared_ptr< MetaData > > partial_result = metadata_build_all(dir,only_valid,read_only);
      log_file << "metadata_build_all: " << dir << " count = " << partial_result.size() << endl;
      result.insert(result.end(),partial_result.begin(),partial_result.end());
    }
    
    return result;
  }
    
  vector< string > filenames(vector< shared_ptr< MetaData > >& metadatas)
  {
    vector<string> filenames;
    for(size_t idx = 0; idx < metadatas.size(); idx++)
	{
		shared_ptr<MetaData> metadata = metadatas[idx];
	    filenames.push_back(metadata->get_filename());
	}
    return filenames;
  }  
}
