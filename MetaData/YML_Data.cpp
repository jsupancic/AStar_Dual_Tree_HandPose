/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "YML_Data.hpp"
#include "PXCSupport.hpp"
#include "Annotation.hpp"
#include "util.hpp"
#include "boost/filesystem.hpp"

namespace deformable_depth
{
  /// SECTION: MetaData_YML_Backed
  MetaData_YML_Backed::MetaData_YML_Backed(string filename, bool read_only) : 
    filename(filename), read_only(read_only)
  {
    bool file_exists = boost::filesystem::exists(filename) or boost::filesystem::exists(filename + ".labels.yml");
    if(read_only and (not file_exists or filename == ""))
      return; // nothing to load!

    FileStorage store;
    //#pragma omp critical
    {
      log_once(printfpp("MetaData_YML_Backed: %s",filename.c_str()));
      store.open(filename,FileStorage::READ|FileStorage::FORMAT_YAML);
      if(!store.isOpened())
	store.open(filename+".labels.yml",FileStorage::READ|FileStorage::FORMAT_YAML);
    }
    if(!store.isOpened())
    {
		printf("warning, store not opened, using defaults. File was %s\n",(filename+".labels.yml").c_str());
      string PXCFired = "Unknown";
      string pose_name = "Unkown";
      return;
    }    
    
    // read the members
    // read HandBB
    HandBB_RGB = store["HandBB"].empty()?Rect():loadRect(store,"HandBB");
    // Z handBB
    HandBB_ZCoords = store["HandBB_ZCoords"].empty()?Rect():loadRect(store,"HandBB_ZCoords");
    log_file << filename << ":" << "HandBB_ZCoords" << ":" << HandBB_ZCoords << endl;
    // read PXC Fired
    if(store["PXCFired"].empty())
    {
      PXCFired == "Unknown";
    }
    else
    {
      bool didFire;
      store["PXCFired"] >> didFire;
      PXCFired = didFire?"Yes":"No";
    }
    // read pose name
    if(store["pose_name"].empty())
      pose_name = "Unknown";
    else
    {
      store["pose_name"] >> pose_name;
      std::replace(pose_name.begin(),pose_name.end(),':','_');
    }
    // read is_left_hand
    if(!store["is_left_hand"].empty())
    {
      store["is_left_hand"] >> is_left_hand.value;
      is_left_hand.initialized = true;
    }
    
    // read keypoints
    if(!store["keypoints"].empty())
    {
      deformable_depth::read<Point2d>(store["keypoints"],keypoints );
      deformable_depth::read<bool>(store["kp_vis"],kp_vis );
    }
    
    store.release();
  }
  
  MetaData_YML_Backed::~MetaData_YML_Backed()
  {
    //printf("Destroying a MetaData\n");
    if(!read_only)
    {
      string write_to = filename + ".labels.yml";
      //cout << "storing to: " << write_to << endl;
      FileStorage store(write_to,FileStorage::WRITE);
    
      store << "HandBB" << HandBB_RGB;
      store << "HandBB_ZCoords" << HandBB_ZCoords;
      store << "PXCFired" << PXCFired;
      store << "pose_name" << pose_name;
      store << "keypoints"; deformable_depth::write(store,keypoints);
      store << "kp_vis"; deformable_depth::write(store,kp_vis);
      //cout << "is_left_hand.initialized = " << is_left_hand.initialized << endl;
      if(is_left_hand.initialized)
	store << "is_left_hand" << is_left_hand.value;
      
      store.release();
    }
  }

  map<string,MetaData_YML_Backed* > MetaData_YML_Backed::get_subdata_yml()
  {
    if(leftP())
      return map<string,MetaData_YML_Backed* >{{"left_hand",this}};
    else
      return map<string,MetaData_YML_Backed* >{{"right_hand",this}};
  }
  
  DetectionSet MetaData_YML_Backed::filter ( DetectionSet src )
  {
    return src;
  }
    
  string MetaData_YML_Backed::get_filename() const
  {
    return filename;
  }
  
  void MetaData_YML_Backed::setPose_name(string newName)
  {
    pose_name = newName;
  }
  
  void MetaData_YML_Backed::set_is_left_hand(bool is_left_hand)
  {
    this->is_left_hand.value = is_left_hand;
    this->is_left_hand.initialized = true;
  }
  
  string MetaData_YML_Backed::get_pose_name()
  {
    string lr = (is_left_hand.value)?string("left"):string("right");
    return pose_name + "_" + lr;
  }

  bool MetaData_YML_Backed::leftP() const
  {
    return is_left_hand.value;
  }
  
  bool MetaData_YML_Backed::hasKeypoint(string name)
  {
    return keypoints.find(name) != keypoints.end();
  }
  
  int MetaData_YML_Backed::keypoint()
  {
    return keypoints.size();
  }
  
  std::pair< Point3d, bool > MetaData_YML_Backed::keypoint ( string name )
  {
    Point2d kp_data = keypoints[name];
    Point3d kp(kp_data.x,kp_data.y,qnan);
    return std::pair<Point3d,bool>(kp,kp_vis[name]);
  }

  string MetaData_YML_Backed::naked_pose() const
  {
    return pose_name;
  }
  
  void MetaData_YML_Backed::keypoint(string name, Point3d value, bool vis)
  {
    keypoint(name,Point3d(value.x,value.y,qnan),vis);
  }

  void MetaData_YML_Backed::keypoint ( string name , Point2d value, bool vis )
  {
    keypoints[name] = value;
    kp_vis[name]    = vis;
  }
  
  vector< string > MetaData_YML_Backed::keypoint_names()
  {
    vector<string> result;
	for(map<string,bool>::iterator iter = kp_vis.begin();
		iter != kp_vis.end(); ++iter)
      result.push_back(iter->first);
    return result;
  }
  
  void MetaData_YML_Backed::change_filename(string filename)
  {
    this->filename = filename;
  }
  
  /// SECTION: RGB Centric
  MetaData_RGBCentric::MetaData_RGBCentric(string filename, bool read_only):
    MetaData_YML_Backed(filename, read_only)
  {
  }
  
  map<string,AnnotationBoundingBox> MetaData_RGBCentric::get_positives()
  {
    map<string,AnnotationBoundingBox> result;
    result["HandBB"] = AnnotationBoundingBox(HandBB_RGB,true);
    return result;
  }
  
  void MetaData_RGBCentric::set_HandBB(Rect newHandBB)
  {
    HandBB_RGB = newHandBB;
  }
  
  shared_ptr<ImRGBZ> MetaData_RGBCentric::load_im()
  {
    Mat RGB, Z;
    //#pragma omp critical(CS_DISK)
    {
      loadPCX_RGB(filename + ".gz",RGB);
      loadPCX_Z(filename + ".gz",Z);
    }
    CustomCamera camera(params::H_RGB_FOV,params::V_RGB_FOV, qnan,qnan);
    return shared_ptr<ImRGBZ>(new ImRGBZ(RGB,Z,filename,camera));
  }

  ///
  /// SECTION: Depth Centric Implementation
  ///
  MetaData_DepthCentric::MetaData_DepthCentric(string filename, bool read_only): 
    MetaData_YML_Backed(filename, read_only),
    RGBandDepthROI(Point(34,28),Point(276,210)),
    //RGBandDepthROI(Point(30,10),Point(290,230))
    cache_ready(false)
  {
  }
 
  void MetaData_DepthCentric::set_HandBB(Rect newHandBB)
  {
    if(read_only)
      cout << "warning: read only metadata written" << endl;
    HandBB_ZCoords = newHandBB + RGBandDepthROI.tl();
    cout << printfpp("MetaData_DepthCentric::set_HandBB: (%d %d) (%d %d)",
      HandBB_ZCoords.tl().x,HandBB_ZCoords.tl().y,
      HandBB_ZCoords.br().x,HandBB_ZCoords.br().y
    ) << endl;
  }
  
  std::pair< cv::Point3d, bool > MetaData_DepthCentric::keypoint(string name)
  {
    pair< cv::Point3d, bool > kp = deformable_depth::MetaData_YML_Backed::keypoint(name);
    kp.first.x -= RGBandDepthROI.tl().x;
    kp.first.y -= RGBandDepthROI.tl().y;
    return kp;
  }

  void MetaData_DepthCentric::keypoint(string name, cv::Point2d value, bool vis)
  {
    value.x += RGBandDepthROI.tl().x;
    value.y += RGBandDepthROI.tl().y;
    
    deformable_depth::MetaData_YML_Backed::keypoint(name, value, vis);
  }
  
  map<string,AnnotationBoundingBox> MetaData_DepthCentric::get_positives()
  {    
    unique_lock<mutex> l(label_cache_mutex);
    
    if(label_cache.size() == 0)
    {
      Rect_<double> handBB = HandBB_ZCoords;
      log_file << "MetaData_DepthCentric::get_positives hbb = " << handBB << endl;
      handBB.x -= RGBandDepthROI.tl().x;
      handBB.y -= RGBandDepthROI.tl().y;
    
      label_cache = MetaData::get_essential_positives(handBB);
      log_file << "MetaData_DepthCentric::get_positives l_hbb = " << label_cache["HandBB"] << endl;
    }
    return label_cache;
  }

  void MetaData_DepthCentric::ensure_im_cached() const
  {
    if(cache_ready)
      return;
    
    unique_lock<mutex> lock(monitor);
    
    if(im == nullptr)
    {
      Mat RGB, Z;
      //#pragma omp critical(CS_DISK)
      {
	loadRGBOnZ(filename + ".gz",RGB,Z);
      }
      assert(RGB.size().area() > 0);
      CustomCamera camera(params::H_Z_FOV,params::V_Z_FOV, qnan,qnan);
      im_raw.reset(new ImRGBZ(RGB.clone(),Z.clone(),filename,camera));
      im.reset(new ImRGBZ(RGB.clone(),Z.clone(),filename,camera));
      *im = (*im)(RGBandDepthROI);
      assert(im->RGB.size().area() > 0);
      cache_ready = true;
    }
  }
  
  shared_ptr< ImRGBZ > MetaData_DepthCentric::load_im()
  {
    ensure_im_cached();
    return shared_ptr<ImRGBZ>(new ImRGBZ(*im)); // try clone
  }
  
  std::shared_ptr< const ImRGBZ > MetaData_DepthCentric::load_im() const
  {
    ensure_im_cached();
    return im;
  }
  
  Mat MetaData_DepthCentric::load_raw_RGB()
  {
    Mat RGB;
    loadPCX_RGB(filename + ".gz",RGB);
    return RGB;
  }

  Mat MetaData_DepthCentric::load_raw_depth()
  {
    Mat RGB,Z;
    loadRGBOnZ(filename + ".gz",RGB,Z);
    return Z;
  }
  
  DetectionSet MetaData_DepthCentric::filter ( DetectionSet src )
  {
    return src;
  }
  
  ///
  /// SECTION: Metadata_Simple
  ///
  Metadata_Simple::Metadata_Simple(string filename, bool read_only, bool b_use_positives, bool b_use_negatives): 
    MetaData_YML_Backed(filename, read_only),
    b_use_positives(b_use_positives), b_use_negatives(b_use_negatives)
  {
  }
  
  Metadata_Simple::~Metadata_Simple()
  {
    if(!read_only && im_cache)
    {
      FileStorage store(filename+".im.yml.gz",FileStorage::WRITE);
      
      store << "RGB" << im_cache->RGB;
      store << "Z" << im_cache->Z;
      store << "camera" << im_cache->camera;
      store << "b_use_negatives" << b_use_negatives;
      store << "b_use_positives" << b_use_positives;
      
      store.release();      
    }
  }  

  Mat Metadata_Simple::getSemanticSegmentation() const 
  {
    if(segmentation.empty())
      return MetaData::getSemanticSegmentation();
    else
      return segmentation;
  }

  void Metadata_Simple::setSegmentation(Mat&segmentation)
  {
    this->segmentation = segmentation;
  }
  
  map< string, AnnotationBoundingBox > Metadata_Simple::get_positives()
  {
    map<string,AnnotationBoundingBox> implicit_abbs = MetaData::get_essential_positives(HandBB_RGB);
    for(auto && pair : explicit_abbs)
    {
      auto found = implicit_abbs.find(pair.first);
      if(found == implicit_abbs.end() or static_cast<Rect_<double>>(found->second) == Rect_<double>())
      {
	implicit_abbs[pair.first] = pair.second;
      }
      else
      {
	implicit_abbs[pair.first] = pair.second;
	log_once(safe_printf("Metadata_Simple::get_positives % overwrite for part %",filename,pair.first));
      }
    }
    return implicit_abbs;
  }

  void Metadata_Simple::ensure_im_cached() const
  {
    unique_lock<mutex> l(cache_mutex);
    if(!im_cache)
    {
      FileStorage store(filename+".im.yml.gz",FileStorage::READ);
      
      Mat RGB; store["RGB"] >> RGB;
      Mat Z;   store["Z"  ] >> Z;
      CustomCamera cam; store["camera"] >> cam;
      
      CustomCamera camera(cam.hFov(),cam.vFov(),qnan,qnan);
      im_cache.reset(new ImRGBZ(RGB,Z,filename,camera));
      store.release();
    }
    
    if(!im_cache || im_cache->Z.size().area() == 0)
    {
      log_file << "Metadata_Simple::load_im warning, no image data" << endl;
      cout << "Metadata_Simple::load_im warning, no image data" << endl;
    }
  }
  
  bool Metadata_Simple::use_negatives() const
  {
    return b_use_negatives;
  }

  bool Metadata_Simple::use_positives() const
  {
    return b_use_positives;
  }

  shared_ptr< ImRGBZ > Metadata_Simple::load_im()
  {
    ensure_im_cached();
    return shared_ptr<ImRGBZ>(new ImRGBZ(*im_cache,true));
  }
  
  std::shared_ptr< const ImRGBZ > Metadata_Simple::load_im() const
  {
    ensure_im_cached();
    return im_cache;
  }

  void Metadata_Simple::set_HandBB(Rect newHandBB)
  {
    HandBB_RGB = newHandBB;
  }

  void Metadata_Simple::setIm(const ImRGBZ& im)
  {
    // copy the provided image into the cache
    im_cache.reset(new ImRGBZ(im,true));
  }  

  void Metadata_Simple::setPositives(const map<string,AnnotationBoundingBox>&annotation)
  {
    cout << "set " << annotation.size() << " explicit positives" << endl;
    explicit_abbs = annotation;
  }

  Mat Metadata_Simple::load_raw_RGB() 
  {
    if(not raw_im_cache)
      if(im_cache)	
	return im_cache->RGB;
      else
	return Mat();
    else
      return raw_im_cache->RGB;
  }

  void Metadata_Simple::setRawIm(ImRGBZ&im)
  {
    raw_im_cache.reset(new ImRGBZ(im,true));
  }
  
  void MetaData_Segmentable::setSegmentation(Mat&segmentation)
  {
  }
}

