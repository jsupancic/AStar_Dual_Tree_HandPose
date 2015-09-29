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
#include "Annotation.hpp"

#ifndef WIN32
#include "LibHandSynth.hpp"
#include "HandSynth.hpp"
#include "ThreadPool.hpp"
#include "ThreadPoolCXX11.hpp"
#include "Cache.hpp"
#include "MetaDataKITTI.hpp"
#include "ExternalModel.hpp"
#include "OcclusionReasoning.hpp"
#endif

namespace deformable_depth
{
  using namespace boost::filesystem;
 
  void export_depth_exr()
  {
    string dir = g_params.require("DIR");
    vector<shared_ptr<MetaData>> data = metadata_build_all(dir, false,true);
    for(auto & datum : data)
    {
      shared_ptr<ImRGBZ> im = datum->load_im();
      vector<string> nums = regex_match(datum->get_filename(), boost::regex("\\d+"));
      int id = fromString<int>(nums.back());
      imwrite(params::out_dir() + printfpp("/depth_%d.exr",id),im->Z);
    }
  }
  
  void do_imageeq()
  {
    string filename = g_params.require("FILE");
    Mat im = imread(filename,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
    log_im("display",imageeq("",im,false,false));
  }
  
  /// 
  /// SECTION: AnnotationBoundingBox
  ///
  
  AnnotationBoundingBox::AnnotationBoundingBox(Rect_< double > bb, float visible) :
    VirtualRect(bb), visible(visible), depth(qnan), confidence(qnan), 
    up(Vec3d(qnan,qnan,qnan)), normal(Vec3d(qnan,qnan,qnan))
  {
  }
  
  AnnotationBoundingBox::AnnotationBoundingBox() : visible(true), depth(qnan), confidence(qnan)
  {
  }
  
  AnnotationBoundingBox& AnnotationBoundingBox::write(const VirtualRect& update_bb)
  {
    *dynamic_cast<Rect_<double>*>(this) = update_bb;
    return *this;
  }
  
  ostream& operator<<(ostream& out, AnnotationBoundingBox& bb)
  {
    out << bb.x << "," << bb.y << "," << bb.br().x << "," << bb.br().y;
    
    return out;
  }
  
  void read(FileNode fn, AnnotationBoundingBox& abb, AnnotationBoundingBox )
  {
    if(fn.isSeq())
    {
      deformable_depth::read(fn, abb, Rect());
    }
    else
    {
      Rect_<double> default_to;
      Rect_<double>&read_ref = abb;
      read(fn["bb"],read_ref,default_to);
      fn["visible"] >> abb.visible;
      fn["depth"] >> abb.depth;
    }
  }

  void write(FileStorage& fs, string , const AnnotationBoundingBox& abb)
  {
    fs << "{";
    fs << "bb" << (Rect_<double>)abb;
    fs << "visible" << abb.visible;
    fs << "depth" << abb.depth;
    fs << "confidence" << abb.confidence;
    fs << "}";
  }
      
  /// SECTION: MetaData
  unique_ptr<leveldb::DB> annotation_db;
  
  static void setupLevelDB()
  {
    static mutex m; lock_guard<mutex> l(m);
    if(annotation_db)
      return;
    
    // open a leveldb
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = false;
    string filename = g_params.has_key("LEVEL_DB")?g_params.get_value("LEVEL_DB"):"data/annotation-database";
    leveldb::DB*db_ptr;
    leveldb::Status status = leveldb::DB::Open(
      options, filename, &db_ptr);
    annotation_db.reset(db_ptr);
    cout << "openleveldb returned: " << status.ToString() << endl;
    assert(status.ok());    
  }
  
  string MetaData::leveldbKeyRoot()
  {
    //deformable_depth::regex_match(get_filename(),"//")
    string root = get_filename() + "//";
    if(root.substr(0,2) != "./")
      root = string("./") + root;
    return root;
  }
  
  string MetaData::getAnnotation(string key)
  {
    if(!annotation_db)
      setupLevelDB();
    
    string value;
    leveldb::ReadOptions read_opts;
    annotation_db->Get(read_opts, leveldbKeyRoot() + key, &value);
    return value;
  }

  bool MetaData::hasAnnotation(string key)
  {
    if(!annotation_db)
      setupLevelDB();
    
    string value;
    leveldb::ReadOptions read_opts;
    leveldb::Status stat = annotation_db->Get(
      read_opts, leveldbKeyRoot() + key, &value);
    if(stat.IsNotFound())
    {
      cout << "Doesnt have: " << leveldbKeyRoot() + key << endl;
      return false;
    }
    else
      return true;
  }
  
  void MetaData::putAnnotation(string key, string value)
  {
    if(!annotation_db)
      setupLevelDB();
    
    annotation_db->Put(leveldb::WriteOptions(), leveldbKeyRoot() + key, value);
  }
  
  bool MetaData::operator<(const MetaData& other) const
  {
    return this->get_filename() < other.get_filename();
  }
  
  MetaData::MetaData() : 
    training_positives_cache_ready(false), all_positives_cache_ready(false)
  {
  }
  
  MetaData::~MetaData()
  {
  }

  void MetaData::export_annotations()
  {
    string filename = get_filename();
    string filename_new = boost::regex_replace(filename,boost::regex("\\.yaml"),string(""));
    filename_new = boost::regex_replace(filename_new,boost::regex("\\.gz"),string(""));
    string filename_out = filename_new + ".annotations";
    log_file << printfpp("export_annotations: %s => %s\n",filename.c_str(),filename_out.c_str()) << endl;
    
    export_annotations(filename_out);
  }
  
  Mat MetaData::load_raw_RGB()
  {
    return load_im()->RGB.clone();
  }

  Mat MetaData::load_raw_depth()
  {
    return load_im()->Z.clone();
  }

  vector<shared_ptr<BaselineDetection> > MetaData::ground_truths() const
  {
    vector<shared_ptr<BaselineDetection> > gts;
    
    for(auto subdatum : get_subdata())
    {
      auto poss = subdatum.second->get_positives();

      if(subdatum.second != this)
      {
	auto sub_gts = subdatum.second->ground_truths();
	gts.insert(gts.end(),sub_gts.begin(),sub_gts.end());
      }
      else
      {
	shared_ptr<BaselineDetection> det = make_shared<BaselineDetection>();
	det->bb = poss.at("HandBB");
	det->notes = subdatum.first;
	for(auto && part : poss)
	  if(part.first != "HandBB")
	  {
	    BaselineDetection part_det;
	    part_det.bb = part.second;
	    det->parts[part.first] = part_det;
	  }
	gts.push_back(det);
      }
    }

    return gts;
  }

  map<string,MetaData* > MetaData::get_subdata() const
  {
    if(leftP())     
      return map<string,MetaData*>{{"left_hand",const_cast<MetaData*>(this)}};
    else
      return map<string,MetaData*>{{"right_hand",const_cast<MetaData*>(this)}};
  }

  void MetaData::export_one_line_per_hand(string filename)
  {
#ifdef DD_ENABLE_HAND_SYNTH
    // precondition that the frame has been annotated.
    if(!hasFullHandPose(*this))
      return;

    shared_ptr<MetaData> gt_datum = oracle_synth(*this);
    if(!gt_datum)
      return;
    auto poss = get_positives();
    auto gt_poss = gt_datum->get_positives();    
    Mat afT = affine_transform(gt_poss.at("HandBB"),poss.at("HandBB"));

    // get the depth image
    shared_ptr<const ImRGBZ> im = load_im();

    // output the hand BB
    ofstream ofs(filename);
    ostringstream oss_header;
    ostringstream oss_example;    
    AnnotationBoundingBox handBB = poss.at("HandBB");
    float annotationDepth = manifoldFn_default(*gt_datum->load_im(),gt_poss.at("HandBB"))[0];
    Point2d hand_center = rectCenter(handBB);
    float handDepth = manifoldFn_default(*im,handBB)[0];

    // write the hand bb
    oss_header << "HandBB_X(px),HandBB_Y(px),HandBB_Z(cm),HandBB_Conf";
    oss_example << hand_center.x << "," << hand_center.y << "," << handDepth << "," << "100";

    // write the rest of the keypoints
    auto write_keypoint = [&](string title,string kp_id)
    {
      oss_header << "," << title << "_X(px)," << title << "_Y(px)," << title << "_Z(cm)," << title << "_Conf";
      if(!gt_datum->hasKeypoint(kp_id))
      {
	oss_example << "," << qnan << "," << qnan << "," << qnan << "," << qnan;
      }
      else
      {
	//Vec3d  pt = all_keypoints.at(dd2libhand(kp_id));//
	//Point2d center(pt[0],pt[1]);
	Point3d center_xyz = gt_datum->keypoint(kp_id).first;
	Point2d center(center_xyz.x, center_xyz.y);
	center = point_affine(center, afT);
	//float z = im->Z.at<float>(center.y,center.x);
	float z = center_xyz.z + handDepth - annotationDepth;

	oss_example << "," << center.x << "," << center.y << "," << z << ",100";
      }
    };
    // pinky
    write_keypoint("pinky-tip","Z_J11");
    write_keypoint("pinky-knuckle","Z_J13");
    write_keypoint("pinky-base","Z_J14");
    // ring
    write_keypoint("ring-tip","Z_J21");
    write_keypoint("ring-knuckle","Z_J23");
    write_keypoint("ring-base","Z_J24");
    // middle
    write_keypoint("middle-tip","Z_J31");
    write_keypoint("middle-knuckle","Z_J33");
    write_keypoint("middle-base","Z_J34");
    // index
    write_keypoint("index-tip","Z_J41");
    write_keypoint("index-knuckle","Z_J43");
    write_keypoint("index-base","Z_J44");
    // thumb
    write_keypoint("thumb-tip","Z_J51");
    write_keypoint("thumb-knuckle","Z_J53");
    write_keypoint("thumb-base","Z_J54");
    // palm
    write_keypoint("wrist","Z_P0");
    write_keypoint("palm","Z_P1");    

    // commit to file
    ofs << oss_header.str() << endl;
    ofs << oss_example.str() << endl;
#else
    log_once("warning");
#endif
  }

  void MetaData::drawSkeletons(Mat&target,Rect boundings) const
  {
    string message = safe_printf("ERROR drawSkeletons not supported by",typeid(*this).name());
    throw runtime_error(message);
  }

  void MetaData::export_keypoints(string filename)    
  {
    ofstream kp_ofs(filename);
    kp_ofs << "kp_name,u,v,x,y,z,occluded" << endl;
    auto im = load_im();
    Mat Z = im->Z;
    Rect handBB = get_positives()["HandBB"];
    if(handBB == Rect())
      return;
    
    for(string & kp_name : keypoint_names())
    {
      pair<Point3d,bool> kp = keypoint(kp_name);
      float u = kp.first.x;
      if(u < 0 || u >= Z.cols)
	continue;
      float v = kp.first.y;
      if(v < 0 || v >= Z.rows)
	continue;
      Point2d uv(u,v);
      if(!handBB.contains(uv))
	continue;
      float z = kp.first.z;
      if(!goodNumber(z))
	z = Z.at<float>(v,u);      
      Point2d xy = map2ortho_cm(im->camera,uv, z);
      string libhand_name_mapping = dd2libhand(kp_name,true);
      if(libhand_name_mapping != "")
      {
	log_once(safe_printf("info mapped: %",libhand_name_mapping));
	kp_ofs << libhand_name_mapping << "," << u << "," << v << "," <<
	  xy.x << "," << xy.y << "," << z << "," << kp.second << endl;
      }
      else
	log_once(safe_printf("warning unmapped: %",kp_name));
    }
  }
  
  void MetaData::export_annotations(string filename)
  {
    // export the annotation
    ofstream annotation_file;
    annotation_file.open(filename);
    
    // export the cluster IDs.
    annotation_file << "cluster," << get_pose_name() << endl;    
    
    // export the bounding boxes for the parts.
    annotation_file << "bb_name,x1,y1,x2,y2" << endl;
    auto positives = get_positives();
    for(map<string,AnnotationBoundingBox>::iterator iter = positives.begin();
	iter != positives.end(); ++iter)   
    { 
      annotation_file << iter->first << "," << iter->second << "," << iter->second.depth << endl;
    }
    
    annotation_file.close();
  }
  
  const map< string, AnnotationBoundingBox >& MetaData::default_training_positives()
  {
    if(!training_positives_cache_ready)
    {
      lock_guard<ExclusionMutexType> l(exclusion);
      if(!training_positives_cache_ready)
      {
	training_positives_cache = params::defaultSelectFn()(get_positives());
	training_positives_cache_ready = true;
      }
    }
    
    return training_positives_cache;
  }
  
  const map< string, AnnotationBoundingBox >& MetaData::get_positives_c()
  {
    if(!all_positives_cache_ready)
    {
      lock_guard<ExclusionMutexType> l(exclusion);
      if(!all_positives_cache_ready)
      {
	all_positives_cache = params::defaultSelectFn()(get_positives());
	all_positives_cache_ready = true;
      }
    }
    
    return all_positives_cache;
  }
        
  Mat MetaData::getSemanticSegmentation() const
  {
    return Mat();
  }

  bool MetaData::use_negatives() const
  {
    // negatives default to true
    return true;
  }
  
  bool MetaData::use_positives() const
  {
    // positives default to false
    return false;
  }
  
  Rect MetaData::bbForKeypoints(Point p1, Point p2, const ImRGBZ& im, double side_len)
  {
    Point2d center = .5*(p1+p2);
    auto read_depth = [&](int y, int x)
    {
      if(x < 0 or y < 0 or x >= im.Z.cols or y >= im.Z.rows)
	return (float)qnan;
      return im.Z.at<float>(y,x);
    };
    float z  = read_depth(center.y,center.x);
    float z1 = read_depth(p1.y,p1.x);
    float z2 = read_depth(p2.y,p2.x);
    vector<float> zs;
    zs.push_back(z);
    zs.push_back(z1);
    zs.push_back(z2);
    sort(zs.begin(),zs.end());
    float zm = zs[zs.size()/2]; //cout << "zm: " << zm << endl;
    //Rect bb = rectResize(Rect(tl.first,br.first),2,2);
    Rect_<double> bb = im.camera.bbForDepth(zm,im.RGB.size(),center.y,center.x,side_len,side_len);
    return bb;
  }
  
  set<string> essential_hand_positives()
  {
    set<string> answer;
    
    for(int iter = 1; iter <= 5; ++iter)
    {
      answer.insert(safe_printf("dist_phalan_%",iter));
    }

    return answer;
  }

  map< string, AnnotationBoundingBox > MetaData::get_essential_positives(Rect_<double> handBB)
  {
    map< string, AnnotationBoundingBox > positives;
    positives["HandBB"] = AnnotationBoundingBox(handBB,true);
    cout << printfpp("handBB: %f %f %f %f",handBB.x,handBB.y,handBB.width,handBB.height) << endl;
    
    const MetaData* cthis = this;
    shared_ptr<const ImRGBZ> im = cthis->load_im();
    // add the distal phalanges
    constexpr float finger_side_len = 3.5;
    double zm = manifoldFn_default(*im,handBB).front();
    for(int iter = 1; iter <= 5; iter++)
    {
      Point p1 = take2(keypoint(printfpp("Z_J%d1",iter)).first);
      Point p2 = take2(keypoint(printfpp("Z_J%d2",iter)).first);      
      //Rect bb = bbForKeypoints(p1,p2,*im,finger_side_len);
      Rect bb = im->camera.bbForDepth(
	zm,im->RGB.size(),(p1.y+p2.y)/2,(p1.x+p2.x)/2,finger_side_len,finger_side_len);
      positives[printfpp("dist_phalan_%d",iter)] = AnnotationBoundingBox(bb,true);
    }
    // now add the intermediate phalanges.
    for(int iter = 1; iter <= 5; iter++)
    {
      Point p1 = take2(keypoint(printfpp("Z_J%d2",iter)).first);
      Point p2 = take2(keypoint(printfpp("Z_J%d3",iter)).first);
      //Rect bb = bbForKeypoints(p1,p2,*im,finger_side_len);
      Rect bb = im->camera.bbForDepth(
	zm,im->RGB.size(),(p1.y+p2.y)/2,(p1.x+p2.x)/2,finger_side_len,finger_side_len);
      positives[printfpp("inter_phalan_%d",iter)] = AnnotationBoundingBox(bb,true);
    }    
    // which are connected to the intermediate phalanges.
    for(int iter = 1; iter <= 5; iter++)
    {
      string name1 = printfpp("Z_J%d3",iter);
      string name2 = printfpp("Z_J%d4",iter);
      if(hasKeypoint(name1) && hasKeypoint(name2))
      {
	Point p1 = take2(keypoint(name1).first);
	Point p2 = take2(keypoint(name2).first);
	//Rect bb = bbForKeypoints(p1,p2,*im,finger_side_len);	
	Rect bb = im->camera.bbForDepth(
	  zm,im->RGB.size(),(p1.y+p2.y)/2,(p1.x+p2.x)/2,finger_side_len,finger_side_len);
	positives[printfpp("proxi_phalan_%d",iter)] = AnnotationBoundingBox(bb,true);
      }
    }
    // add the wrist.
    {
      if(hasKeypoint("carpals") && hasKeypoint("Z_P0"))
      {
	Point p1 = take2(keypoint("carpals").first);
	Point p2 = take2(keypoint("Z_P0").first);
	Rect bb = bbForKeypoints(p1,p2,*im,20);
	positives["wrist"] = AnnotationBoundingBox(bb,true);      
      }
    }
    
    return positives;
  }  
  
  void write(cv::FileStorage& fs, std::string, const std::shared_ptr<deformable_depth::MetaData>&md)
  {
    fs << "{";
    fs << "uri" << md->get_filename();
    fs << "}";
  }
  
  void read(FileNode fn, shared_ptr< MetaData >& md, shared_ptr< MetaData > )
  {
    string uri;
    fn["uri"] >> uri;
    md = metadata_build_with_cache(uri,true);
  }
  
  /// SECTION: MetaData_flip_LR
  void MetaData_LR_Flip::build_affine_transform()
  {
    width = backend->load_im()->Z.cols;
    height = backend->load_im()->Z.rows;
    affine = affine_lr_flip(width);
  }
  
  MetaData_LR_Flip::MetaData_LR_Flip(shared_ptr< MetaData > backend) : backend(backend)
  {        
    build_affine_transform();
  }
  
  MetaData_LR_Flip::~MetaData_LR_Flip()
  {
  }

  string MetaData_LR_Flip::get_filename() const
  {
    return backend->get_filename() + "mirrored";
  }

  string MetaData_LR_Flip::get_pose_name()
  {
    string lr = (leftP())?string("left"):string("right");
    return backend->get_pose_name() + "_" + lr;
  }
  
  string MetaData_LR_Flip::get_pose_name_naked()
  {
    return backend->get_pose_name();
  }

  std::pair< Point3d, bool > MetaData_LR_Flip::keypoint(string name)
  {
    auto kp_pair = backend->keypoint(name);
    kp_pair.first = point_affine(kp_pair.first,affine);
    return kp_pair;
  }

  bool MetaData_LR_Flip::hasKeypoint(string name)
  {
    return backend->hasKeypoint(name);
  }
  
  DetectionSet MetaData_LR_Flip::filter(DetectionSet src)
  {
    build_affine_transform();
    assert(width > 0);
    assert(affine.size().area() > 0);
    
    // apply transform
    Mat affine_inv; cv::invertAffineTransform(affine,affine_inv); 
    for(size_t idx = 0; idx < src.size(); idx++)
	{
		DetectorResult&detection = src[idx];
		detection->BB = rect_Affine(detection->BB,affine_inv);
	}

    // filter
    src = backend->filter(src);
    
    // inverse transform]
    for(size_t idx = 0; idx < src.size(); idx++)
	{
		DetectorResult&detection = src[idx];
	    detection->BB = rect_Affine(detection->BB,affine);    
	}

    return src;
  }

  void MetaData_LR_Flip::ensure_im_cached() const
  {
    unique_lock<mutex> lock(monitor);
    
    if(!im_cache)
    {
      shared_ptr<ImRGBZ> im = backend->load_im();
      assert(im->RGB.size().area() > 0);
      im_cache.reset(new ImRGBZ(im->flipLR()));
    }
  }
  
  shared_ptr< ImRGBZ > MetaData_LR_Flip::load_im()
  {
    ensure_im_cached();
    return shared_ptr<ImRGBZ>(new ImRGBZ(*im_cache,true));
  }
  
  std::shared_ptr< const ImRGBZ > MetaData_LR_Flip::load_im() const
  {
    ensure_im_cached();
    return im_cache;
  }

  bool MetaData_LR_Flip::leftP() const
  {
    return !backend->leftP();
  }
  
  bool MetaData_LR_Flip::use_negatives() const
  {
      return backend->use_negatives();
  }

  bool MetaData_LR_Flip::use_positives() const
  {
      return backend->use_positives();
  }
  
  map< string, AnnotationBoundingBox > MetaData_LR_Flip::get_positives()
  {
    typedef map< string, AnnotationBoundingBox > result_map;
    result_map result = backend->get_positives();
    for(result_map::iterator iter = result.begin(); iter != result.end(); ++iter)
    {
      iter->second.write(rect_Affine(iter->second,affine));
      Vec3d & up = iter->second.up;
      Vec3d & norm = iter->second.normal;
      up[0] = - up[0];
      norm[0] = - norm[0];
    }
    //cout << "flip" << result["HandBB"].up << endl;
    return result;
  }
  
  int MetaData_LR_Flip::keypoint()
  {
    return backend->keypoint();
  }

  vector< string > MetaData_LR_Flip::keypoint_names()
  {
    return backend->keypoint_names();
  }
  
  ///
  /// SECTION: MetadataNoKeypoints
  ///
  bool MetaDataNoKeypoints::hasKeypoint(string name)
  {
    return false;
  }

  int MetaDataNoKeypoints::keypoint()
  {
    return 0;
  }

  std::pair< Point3d, bool > MetaDataNoKeypoints::keypoint(string name)
  {
    return std::pair< Point3d, bool >(Point3d(),false);
  }

  vector< string > MetaDataNoKeypoints::keypoint_names()
  {
    return vector<string>{};
  }
}
