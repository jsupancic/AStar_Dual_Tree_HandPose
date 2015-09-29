/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "MetaData_Pool.hpp"
#include "LibHandMetaData.hpp"
#include "MetaDataKITTI.hpp"
#include <boost/filesystem.hpp>
#include "ICL_MetaData.hpp"
#include "RegEx.hpp" 
#include "NYU_Hands.hpp"

namespace deformable_depth
{
#ifdef DD_CXX11
  shared_ptr< MetaData > metadata_build_with_cache(string filename, bool read_only);
#endif

  MetaData_Pooled::MetaData_Pooled(AllocatorFn allocatorFn,bool use_positives,bool use_negatives,string filename) :
    allocatorFn(allocatorFn), b_use_positives(use_positives), b_use_negatives(use_negatives), filename(filename)
  {    
  }

  MetaData_Pooled::MetaData_Pooled(string filename,bool read_only) : 
    filename(filename)
  {
#ifndef WIN32
    boost::regex synth_pattern(".*synth.*",boost::regex_constants::icase);
    boost::regex mirror_pattern(".*mirrored.*");
    boost::regex KITTI_pattern(".*KITTI.*");
    boost::regex ICL_pattern(".*ICL.*");
    boost::regex nyu_pattern(".*nyu.*");

    if(boost::regex_match(filename,mirror_pattern))
    {
      string orig_file = boost::regex_replace(filename,boost::regex("mirrored"),"");
      log_once(printfpp("caching mirrored metadata: %s",orig_file.c_str()));
      shared_ptr<MetaData> orig = metadata_build_with_cache(orig_file,read_only);
      allocatorFn = [orig]() -> MetaData*{return new MetaData_LR_Flip(orig);};
      b_use_positives = orig->use_positives();
      b_use_negatives = orig->use_negatives();
    }
    else if(boost::regex_match(filename,nyu_pattern))
    {
      bool training = boost::regex_match(filename,boost::regex(".*training.*"));
      assert(training);
      std::vector<std::string> nums = deformable_depth::regex_match(filename, boost::regex("\\d+"));
      require_equal<size_t>(nums.size(),1);
      int id = fromString<int>(nums.front());
      allocatorFn = [id,read_only]() -> MetaData*{return NYU_training_datum(id);};
      b_use_positives = true;
      b_use_negatives = false;      
    }
    else if(boost::regex_match(filename,synth_pattern))
    {
      allocatorFn = [filename,read_only]() -> MetaData*{return new LibHandMetadata(filename,read_only);};
      b_use_positives = true;
      b_use_negatives = false;
    }    
    else if(boost::regex_match(filename,KITTI_pattern))
    {
      istringstream iss(filename);
      string KITTI;
      int id, training; 
      iss >> KITTI >> id >> training;
      allocatorFn = [id,training]() -> MetaData*{return new MetaDataKITTI(id,training);};
      b_use_positives = true;
      b_use_negatives = true;
    }
    else if(boost::regex_match(filename,ICL_pattern))
    {
      allocatorFn = [filename,read_only]() -> MetaData*{return load_ICL_training(filename);};
      b_use_positives = true;
      b_use_negatives = false;      
    }
    else 
#endif
    {
      allocatorFn = [filename,read_only]() -> MetaData*{return new DefaultMetaData(filename,read_only);};
      b_use_positives = false;
      b_use_negatives = true;
    }        
  }

  MetaData_Pooled::~MetaData_Pooled()
  {
  }

  map<string,AnnotationBoundingBox> MetaData_Pooled::get_positives()
  {
    require_loaded();
    map<string,AnnotationBoundingBox> positives = backend->get_positives();
    //cout << "pooled" << positives["HandBB"].up << endl;
    return positives;
  }

  std::shared_ptr<ImRGBZ> MetaData_Pooled::load_im() 
  {
    require_loaded();
    return backend->load_im();
  }

  Mat MetaData_Pooled::load_raw_RGB()
  {
    require_loaded();
    return backend->load_raw_RGB();
  }

  Mat MetaData_Pooled::load_raw_depth()
  {
    require_loaded();
    return backend->load_raw_depth();
  }

  Mat MetaData_Pooled::getSemanticSegmentation() const
  {
    require_loaded();
    return backend->getSemanticSegmentation();
  }

  void MetaData_Pooled::drawSkeletons(Mat&target,Rect Boundings) const
  {
    require_loaded();
    return backend->drawSkeletons(target,Boundings);
  }

  std::shared_ptr<const ImRGBZ> MetaData_Pooled::load_im() const
  {
    require_loaded();
    // call the const version to save on memory...
    const MetaData*c_backend = backend.get();
    return c_backend->load_im();
  }

  string MetaData_Pooled::get_pose_name()
  {
    require_loaded();
    return backend->get_pose_name();
  }

  string MetaData_Pooled::get_filename() const
  {
    // if(backend)
    // {
    //   require_equal<string>(boost::filesystem::path(backend->get_filename()).stem().string(),
    // 			    boost::filesystem::path(filename).stem().string());
    // }
    return filename;
  }

  // keypoint functions
  pair<Point3d,bool> MetaData_Pooled::keypoint(string name)
  {
    require_loaded();
    return backend->keypoint(name);
  }

  int MetaData_Pooled::keypoint()
  {
    require_loaded();
    return backend->keypoint();
  }

  bool MetaData_Pooled::hasKeypoint(string name)
  {
    require_loaded();
    return backend->hasKeypoint(name);
  }

  vector<string> MetaData_Pooled::keypoint_names()
  {
    require_loaded();
    return backend->keypoint_names();
  }

  DetectionSet MetaData_Pooled::filter(DetectionSet src)
  {
    require_loaded();
    return backend->filter(src);
  }

  bool MetaData_Pooled::leftP() const
  {
    require_loaded();
    return backend->leftP(); 
  }

  bool MetaData_Pooled::use_negatives() const
  {
    if(backend)
    {
      assert(backend->use_negatives() == b_use_negatives);
      return backend->use_negatives();
    }
    else
      return b_use_negatives;
  }

  bool MetaData_Pooled::use_positives() const
  {
    if(backend)
    {
      assert(backend->use_positives() == b_use_positives);
      return backend->use_positives();
    }
    else
      return b_use_positives;
  }

  void MetaData_Pooled::require_loaded() const
  {
    lock_guard<ExclusionMutexType> m(exclusion);
    if(!backend)
    {
      backend.reset(allocatorFn());
      log_once(safe_printf("pooled % leftP = ",backend->get_filename(),backend->leftP()));
    }
  }

  shared_ptr<MetaData> MetaData_Pooled::getBackend() const
  {
    require_loaded();
    return backend;
  }

  bool MetaData_Pooled::loaded() const 
  {
    return !!backend;
  }
}
