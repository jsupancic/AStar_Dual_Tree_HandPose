/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "LibHandMetaData.hpp"
#include <boost/filesystem.hpp>
#include "Log.hpp"
#include "OcclusionReasoning.hpp"
#include "util.hpp"
#include "RegEx.hpp"
#include "LibHandRenderer.hpp"
#include "InverseKinematics.hpp"
#include "Colors.hpp"

namespace deformable_depth 
{
  ///
  /// SECTION: LibHand_DefDepth_Keypoint_Mapping
  ///
  static string get_part_name(const string&s)
  {
    auto split_pos = s.find(':');
    if(split_pos != string::npos)
    {
      return s.substr(split_pos+1,string::npos);
    }
    else
      return s;
  }
  
  void LibHand_DefDepth_Keypoint_Mapping::insert(string lhName, string ddName)
  {
    lh_to_dd[lhName] = ddName;
    dd_to_lh[ddName] = lhName;    
  }
  
  LibHand_DefDepth_Keypoint_Mapping::LibHand_DefDepth_Keypoint_Mapping()
  {
    // map the finger tips
    for(int iter = 1; iter <= 5; iter++)
    {
      string dd_name = printfpp("Z_J%d1",iter);
      string lh_name = printfpp("finger%djoint3tip",iter);
      insert(lh_name,dd_name);
    }
    
    // map the finger joints.
    for(int fingerIter = 1; fingerIter <= 5; fingerIter++)
      for(int jointIter = 2; jointIter <= 4; jointIter++)
      {
	string dd_name = printfpp("Z_J%d%d",fingerIter,jointIter);
	string lh_name = printfpp("finger%djoint%d",fingerIter,5-jointIter);
	insert(lh_name,dd_name);
      }
    
    // map the palm
    insert("metacarpals","Z_P0");
    insert("carpals","carpals");
    insert("Bone","Z_P1");
  }

  static LibHand_DefDepth_Keypoint_Mapping a_mapping;

  string dd2libhand(string dd_name, bool can_fail)
  {
    dd_name = get_part_name(dd_name);
    string answer = a_mapping.dd_to_lh[dd_name];
    if(answer == "")
      cout << "map_kp_name fails: " << dd_name << endl;
    assert(answer != "" || can_fail);
    return answer;    
  }

  string libhand2dd(string libhand_name, bool can_fail)
  {
    libhand_name = get_part_name(libhand_name);
    string answer = a_mapping.lh_to_dd[libhand_name];
    if(answer == "")
      cout << "unmap_kp_name fails: " << libhand_name << endl;
    assert(answer != "" || can_fail);
    return answer;    
  }
  
  /// 
  /// SECTION: LibHandMetadata
  ///
  static int getId(string filename)
  {
    std::vector<std::string> numbers = 
      deformable_depth::regex_match(filename, boost::regex("\\d+"));
    return fromString<int>(numbers.back());
  }

  void LibHandMetadata::log_metric_bb() const
  {
    // draw the metric bb
    Mat metric_viz = imageeq("",Z.clone(),false,false);
    Point bl(handBB.tl().x,handBB.br().y);
    Point tr(handBB.br().x,handBB.tl().y);
    cv::line(metric_viz,handBB.tl(),bl,toScalar(RED));
    cv::line(metric_viz,handBB.br(),bl,toScalar(GREEN));

    // compute edge lengths using law of cosines
    double z  = medianApx(Z,handBB,.5);
    double t1 = camera.distance_angular(handBB.tl(),bl);
    double d1 = camera.distance_geodesic(handBB.tl(),z,bl,z);
    double t2 = camera.distance_angular(handBB.br(),bl);
    double d2 = camera.distance_geodesic(handBB.br(),z,bl,z);

    Mat text = vertCat(image_text(safe_printf("% cm % rad",d1,t1),RED),
		       image_text(safe_printf("% cm % rad",d2,t2),GREEN));
    text = vertCat(text,image_text(safe_printf("% cm",z)));
    log_im("MetricBBSides",vertCat(metric_viz,text));
  }

#ifdef DD_ENABLE_HAND_SYNTH
  LibHandMetadata::LibHandMetadata(
    string filename, 
    const Mat& RGB, const Mat& Z, 
    const Mat& segmentation, const Mat&semantics,
    Rect handBB,
    libhand::HandRenderer::JointPositionMap& joint_positions,
    CustomCamera camera, bool read_only) : 
    filename(filename), RGB(RGB), Z(Z), joint_positions(joint_positions), 
    handBB(handBB), camera(camera), read_only(read_only)
  {
    // export for Greg
    if(!read_only)
    {
      int id = getId(filename);
      string save_dir = boost::filesystem::path(filename).parent_path().string();
      log_once(printfpp("save_dir = %s",save_dir.c_str()));
      export_annotations(printfpp(
			   (save_dir + "/gen%d.annotations").c_str(),id));
      imwrite(printfpp((save_dir +"/gen%d.jpg").c_str(),id),RGB);
      imwrite(printfpp((save_dir +"/segmentation%d.jpg").c_str(),id),segmentation);
      imwrite(printfpp((save_dir +"/semantics%d.png").c_str(),id),semantics);
    }
    else
      unlink(filename.c_str());

    log_metric_bb();
  }
#endif
  
  static Mat load_semantics(string filename)
  {
    int id = getId(filename);
    Mat semantics_image;
    auto load_semantics = [&](string extension) -> bool
    {
      string sem_im_filename = 
      boost::filesystem::path(filename).parent_path().string() 
      + printfpp("/semantics%d.%s",id,extension.c_str());    
      semantics_image = imread(sem_im_filename,CV_LOAD_IMAGE_COLOR);
      if(semantics_image.empty())
      {
	log_file << "Failed to load semantics_image: " << sem_im_filename << endl;
	return false;
      }
      for(int yIter = 0; yIter < semantics_image.rows; ++yIter)
	for(int xIter = 0; xIter < semantics_image.cols; ++xIter)
	{
	  Vec3b&pixel = semantics_image.at<Vec3b>(yIter,xIter);
	  pixel = HandRegion::quantize(pixel);
	}

      log_file << safe_printf("(info) loaded semantics image of size = ",semantics_image.size()) << endl;
      return true;
    };
    assert(load_semantics("png") or load_semantics("jpg"));
    return semantics_image;
  }

  Mat LibHandMetadata::getSemanticSegmentation() const
  {
    Mat sem = load_semantics(filename);
    return sem;
  }

  template<typename T> T defaultFillValue();
  template<> float defaultFillValue()
  {
    return params::MAX_Z();
  }
  template<> Vec3b defaultFillValue()
  {
    return Vec3b(255,0,0);
  }
  
  template<typename T>
  static Mat supress_background(const Mat&Z,string filename, Rect&handBB)
  {
    // get the number
    int id = getId(filename);
    
    // load segmentation_image
    string seg_im_filename = 
      boost::filesystem::path(filename).parent_path().string() 
      + printfpp("/segmentation%d.jpg",id);
    Mat segmentation_image = imread(seg_im_filename,CV_LOAD_IMAGE_GRAYSCALE);
    if(segmentation_image.empty())
    {
      log_file << "Failed to load segmentation_image: " << seg_im_filename << endl;
      assert(false);
    }
    
    // load the semantics images
    Mat semantics_image = load_semantics(filename);
    
    // supress background pixels using it.
    assert(Z.size() == segmentation_image.size());
    bool using_arm = !g_params.has_key("DONT_USE_ARM");
    Mat result = Z.clone();
    for(int rIter = 0; rIter < Z.rows; rIter++)
      for(int cIter = 0; cIter < Z.cols; cIter++)
      {
	Vec3b&sem = semantics_image.at<Vec3b>(rIter,cIter);

	// fill conditions
        // overwrite if background
	bool is_background = segmentation_image.at<uchar>(rIter,cIter) < 10 && 
									 sem[0] < 10 && sem[1] < 10 && sem[2] < 10;
	// overwrite if wrist/armed
	bool is_wrist_or_arm = !using_arm and (segmentation_image.at<uchar>(rIter,cIter) < 10 or HandRegion(sem).is_arm());
	// overwrite is boarder
	bool is_boarder = rIter == 0 or cIter == 0 or rIter == Z.rows - 1 or cIter == Z.cols - 1;

	// execute if filled		
	if(is_background or is_wrist_or_arm)
	  result.at<T>(rIter,cIter) = defaultFillValue<T>();
      }
    
    // calculate the handBB
    // handBB = bbWhere(segmentation_image,[&](Mat&im,int y, int x)
    //  		     {
    //  		       assert(im.type() == DataType<uchar>::type);
    //  		       return im.at<uchar>(y,x) > 100;
    //  		     });    
    log_im_decay_freq("calcedBB",[&]()
		      {
			Mat vis = display("",semantics_image,false,false);
			rectangle(vis,handBB.tl(),handBB.br(),Scalar(0,255,0));
			return vis;
		      });

    return result;
  }
  
  void LibHandMetadata::load()
  {
    lock_guard<ExclusionMutexType> l(exclusion);
    if(loaded)
      return;

    FileStorage fs(filename,FileStorage::READ);
    if(!fs.isOpened())
    {
      fs.open(filename + ".gz",FileStorage::READ);
      if(!fs.isOpened())
      {      
	cout << "failed to open: " << filename << endl;
	assert(false);
      }
      else
	this->filename += ".gz";
    }
      
    fs["RGB"] >> RGB;
    RGB = supress_background<Vec3b>(RGB,filename,handBB);
    fs["Z"] >> Z;
    Z = supress_background<float>(Z,filename,handBB);
    fs["camera"] >> camera;
    handBB = loadRect(fs,"handBB");
    //fs["handBB"] >> handBB;
    read(fs["joint_positions"],joint_positions);
    // reading the pose is more complex
    auto on_bad_pose  = [&] () 
    {
      //pose_name = "UNCLUSTERED";
      // take it from directory name
      boost::filesystem::path path(filename);
      boost::filesystem::path dir = path.parent_path();
      pose_name = dir.filename().string();
      log_file << "assumed pose = " << pose_name << endl;      
    };
    if(!fs["pose_name"].empty())
    {
      fs["pose_name"] >> pose_name;
      if(pose_name == "")
	on_bad_pose();
    }
    else
    {
      on_bad_pose();
    }
        
    fs.release();
    loaded = true;
  }

  LibHandMetadata::LibHandMetadata(string filename, bool read_only) : 
    filename(filename), read_only(read_only), camera(qnan,qnan,qnan,qnan), loaded(false)
  {
    load();
  }
  
  string LibHandMetadata::naked_pose() const
  {
    return pose_name;
  }
  
  LibHandMetadata::~LibHandMetadata()
  {
    if(read_only)
      return;
    
    log_file << "writing: " << filename << endl;
    FileStorage fs(filename,FileStorage::WRITE);
    
    fs << "RGB" << RGB;
    fs << "Z" << Z;
    fs << "handBB" << handBB;
    fs << "camera" << camera;
    fs << "joint_positions"; write(fs,joint_positions);
    fs << "format" << "LibHandMetadata";
    fs << "pose_name" << pose_name;
    
    fs.release();
  }
  
  DetectionSet LibHandMetadata::filter(DetectionSet src)
  {
    return src;
  }
  
  string LibHandMetadata::get_filename() const
  {
    return filename;
  }
  
  string LibHandMetadata::get_pose_name()
  {
    return pose_name;
  }
  
  void LibHandMetadata::setPose_name(string pose_name)
  {
    this->pose_name = pose_name;
  }
  
  LibHand_DefDepth_Keypoint_Mapping keypoint_mapping;
  
  // DD => LH
  string LibHandMetadata::map_kp_name(string dd_name) const
  {
    return dd2libhand(dd_name);
  }

  // LH => DD
  string LibHandMetadata::unmap_kp_name(string libhand_name) const
  {
    return libhand2dd(libhand_name);
  }  
  
  map< string, AnnotationBoundingBox > LibHandMetadata::get_positives()
  {
    unique_lock<mutex> l(monitor);
    if(!poss.empty())
    {
      //cout << "had handBB" << poss["HandBB"].up << endl;
      return poss;
    }
      
    auto parts = MetaData::get_essential_positives(handBB);
    poss["HandBB"] = parts["HandBB"];
    parts.erase("HandBB");
    // greedly assign occlusion to conflicts.
    for(auto && part : parts)
    {
      bool is_vis = is_visible(1-part.second.visible);
      bool overlaps_visible = false;
      for(auto && existing_part : poss)
      {
	if(existing_part.first == "HandBB")
	  continue;
	
	bool other_vis = is_visible(1-existing_part.second.visible);
	if(occlusion_intersection(part.second,existing_part.second) && other_vis)
	{
	  overlaps_visible = true;
	  if(part.first == "wrist")
	    log_file << printfpp("wrist overlaps %s",existing_part.first.c_str()) << endl;
	}
      }
      bool part_vis = valid_configuration(is_vis,overlaps_visible);
      if(part.first == "wrist")
	log_file << printfpp("wrist %d %d %d",(int)is_vis,(int)overlaps_visible,(int)part_vis) << endl;
      //part.second.visible = part_vis;
      poss[part.first] = part.second;
      poss[part.first].visible = part_vis;
    }
    
    // hand BB seems a bit to tight
    // compute the desired new area
    double A  = poss["HandBB"].area();
    double Ap = 1.5 * A;
    double dA = Ap - A;
    // 
    double h = poss["HandBB"].height;
    double w = poss["HandBB"].width;
    double dl = .5 * (std::sqrt(4*dA+h*h+2*h*w+w*w)-h-w);
    cout << "dl = " << dl << endl;
    poss["HandBB"].write(rectFromCenter(rectCenter(poss["HandBB"]),Size(w+dl,h+dl)));
    poss["HandBB"].up = up();
    poss["HandBB"].normal = normal_dir();
    poss["HandBB"].depth = medianApx(Z,poss["HandBB"],.5);
    
    // show what we loaded
    log_im_decay_freq("returnedHandBB",[&]()
		      {
			Mat vis = display("",Z,false,false);
			rectangle(vis,poss.at("HandBB").tl(),poss.at("HandBB").br(),Scalar(0,255,0));
			return vis;
		      });

    //cout << "set handBB" << poss["HandBB"].up << endl;
    return poss;
  }

  bool LibHandMetadata::hasKeypoint(string name)
  {
    return joint_positions.find(map_kp_name(name)) != joint_positions.end();
  }
  
  std::pair<Point3d,bool> LibHandMetadata::keypoint(string name)
  {
    const LibHandMetadata* cthis = this;
    
    return cthis->keypoint(name);
  }
  
  std::pair<Point3d,bool> LibHandMetadata::keypoint(string name) const
  {
    Vec3d pos3 = joint_positions.at(map_kp_name(name));
    Point3d pos(pos3[0],pos3[1],pos3[2]);
    return pair<Point3d,bool>(pos,true);
  }

  int LibHandMetadata::keypoint()
  {
    return keypoint_mapping.dd_to_lh.size();
  }

  vector< string > LibHandMetadata::keypoint_names()
  {
    vector<string> names;
    
    for(auto && named_point : joint_positions)
    {
      string unmapped_name = unmap_kp_name(named_point.first);
      if(unmapped_name == "")
	continue;
      names.push_back(unmapped_name);
    }
    
    return names;
  }
  
  bool LibHandMetadata::leftP() const
  {
    //Point p_bone = keypoint(unmap_kp_name("metacarpals")).first;
    //Point p_tip  = keypoint(unmap_kp_name("finger3joint3tip")).first;
    //return p_bone.x > p_tip.x;
    
    return false; // I'm sure it's a right hand
  }

  void LibHandMetadata::ensure_im_cached() const
  {
    unique_lock<mutex> l(load_im_monitor);
    if(im_cache == nullptr)
      im_cache.reset(
	new ImRGBZ(RGB,Z,filename,camera));
  }
  
  shared_ptr< ImRGBZ > LibHandMetadata::load_im()
  {
    ensure_im_cached();
    
    // return a copy
    return shared_ptr<ImRGBZ>(new ImRGBZ(*im_cache,true));
  }
  
  std::shared_ptr< const ImRGBZ > LibHandMetadata::load_im() const
  {
    ensure_im_cached();
    
    return im_cache;
  }
  
  bool LibHandMetadata::use_negatives() const
  {
    return false;
  }
  
  bool LibHandMetadata::use_positives() const
  {
    return true;
  }
  
  void LibHandMetadata::change_filename(string filename)
  {
    this->filename = filename;
  }

  template<typename K,typename V>
  static V get_bone002(const map<K,V>&map)
  {
    if(map.find("Bone_002") != map.end())
      return map.at("Bone_002");
    else
      return map.at("Bone.002");
  }

  Vec3d LibHandMetadata::up() const
  {
    Vec3d carpal = joint_positions.at("carpals");
    Vec3d center = get_bone002(joint_positions);
    //cout << "carpal: " << carpal << endl;
    //cout << "center: " << center << endl;
    Vec3d up_dir = center-carpal;        
    return up_dir;
  }

  Vec3d LibHandMetadata::lr_dir() const
  {
    Vec3d center = get_bone002(joint_positions);
    Vec3d finger1joint1 = joint_positions.at("finger1joint1");
    Vec3d lr_dir = finger1joint1 - center;
    return lr_dir;
  }

  Vec3d LibHandMetadata::normal_dir() const
  {
    Vec3d normal = lr_dir().cross(up());
    normal = normal / std::sqrt(normal.ddot(normal)); // normalize
    return normal;
  }
}
