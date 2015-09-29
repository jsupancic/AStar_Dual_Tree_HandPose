/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "LibHandRenderer.hpp"
#include "util_real.hpp"
#include "util.hpp"
#include "Renderer.hpp"
#include "RegEx.hpp"
#include "InverseKinematics.hpp"
#include "OcclusionReasoning.hpp"

#ifdef DD_ENABLE_HAND_SYNTH
#include "OGRE/OgreRoot.h"
#endif

namespace deformable_depth
{
  using namespace cv;
  using namespace std;

#ifdef DD_ENABLE_HAND_SYNTH
  recursive_mutex libhand_exclusion; // libhand is not thread safe
  
  LibHandRenderer::LibHandRenderer(string spec_path,Size sz) : 
    scene_spec(spec_path),
    world_size_randomization_sf(1),
    flip_lr(false)
  {
    lock_guard<recursive_mutex> l(libhand_exclusion);
    
    cout << "*********************************************************" << endl;
    cout << "LibHandRenderer::LibHandRenderer Setting up " << spec_path << endl;
    cout << "*********************************************************" << endl;
    // setup the renderer object
    hand_renderer.Setup(sz.width,sz.height);    
    hand_renderer.LoadScene(scene_spec);    
  }
  
  void LibHandRenderer::commit_model()
  {
    string commit_message = printfpp("commiting hand pose with %d joints",(int)hand_pose.num_joints());
    log_file << commit_message << endl;
    hand_renderer.SetHandPose(hand_pose);
    hand_renderer.set_camera_spec(cam_spec);    
  }
  
  libhand::HandCameraSpec& LibHandRenderer::get_cam_spec()
  {
    return cam_spec;
  }

  libhand::FullHandPose& LibHandRenderer::get_hand_pose()
  {
    return hand_pose;
  }

  libhand::HandRenderer& LibHandRenderer::get_hand_renderer()
  {
    return hand_renderer;
  }
  
  libhand::HandRenderer::JointPositionMap& LibHandRenderer::get_jointPositionMap()
  {
    return jointPositionMap;
  }

  libhand::SceneSpec& LibHandRenderer::get_scene_spec()
  {
    return scene_spec;
  }

  Mat& LibHandRenderer::getDepth()
  {
    return Z;
  }

  void LibHandRenderer::read_depth_buffer(Mat& dest)
  {
    int W = hand_renderer.render_width();
    int H = hand_renderer.render_height();    
    gl_cv_read_buffers_Z(dest,W,H);
    cv::flip(dest,dest,0);    
    if(flip_lr)
      cv::flip(dest,dest,+1);

    double fovy_rad, fovx_rad, clip_min, clip_max;
    gl_get_camera_parameter_from_perspective_matrix(fovy_rad,fovx_rad,clip_min,clip_max);
    camera = CustomCamera(rad2deg(fovx_rad),rad2deg(fovy_rad),W,H);
    //dest = pointCloudToDepth(dest,camera);
    //dest = 
  }

  void LibHandRenderer::read_rgb_buffer()
  {
    RGB = hand_renderer.pixel_buffer_cv().clone();
    if(flip_lr)
      cv::flip(RGB,RGB,+1);
  }
  
  void LibHandRenderer::read_joint_positions()
  {
    lock_guard<recursive_mutex> l(libhand_exclusion);
    commit_model();
    //hand_renderer.RenderHand();
    jointPositionMap.clear();
    hand_renderer.walk_bones(jointPositionMap);
    if(flip_lr)
    {
      int W = hand_renderer.render_width();
      for(auto&pair : jointPositionMap)
	pair.second[0] = W - pair.second[0]; // flip the X coordinate
    }
    register_joint_positions();
  }  
  
  void LibHandRenderer::register_joint_positions()
  {
    // now register the keypoints in the joint position map
    for(auto && kp : get_jointPositionMap())
    {
      //double&kp_z = kp.second[2];
      kp.second[2] = -register_one_point(kp.second[2], false);
      //log_file << printfpp("registered depth %f => %f",kp_z,kp.second[2]) << endl;
      // if(!Z.empty())
      // {
      // 	double imZ = Z.at<float>(kp.second[1],kp.second[0]);

      // 	string dbg_message = 
      // 	  printfpp("%s z(x,y) = %f z = %f\n",
      // 		  kp.first.c_str(),imZ , kp_z );
      // 	//cout << dbg_message << endl;
      // }
    }
  }

  void LibHandRenderer::merge_one_pix(
    float&cm_z,float&out_z,
    Vec3b&rnd_rgb,Vec3b&out_rgb,
    double&min_z, double&max_z
				  )
  {
    // merge
    if(goodNumber(cm_z) && cm_z < out_z)
    {
      // copy the nearest image
      //assert(cm_z > 0);
      out_z = cm_z;
      out_rgb = rnd_rgb;
      
      // update statistics
      min_z = std::min<float>(min_z,cm_z);
      max_z = std::max<float>(max_z,cm_z);
    }    	    
  }  
  
  bool LibHandRenderer::synth_one_register()
  {
    // init the images
    Mat init_RGB = Mat(hand_renderer.render_height(),hand_renderer.render_width(),
	      DataType<Vec3b>::type,Scalar::all(0));
    Mat init_Z = Mat(hand_renderer.render_height(),hand_renderer.render_width(),
	        DataType<float>::type,Scalar::all(inf));
    
    // map [0,1] in rndZ to [z_near,z_far] in Z
    double z_near = hand_renderer.z_near();
    double z_far  = hand_renderer.z_far();
    cout << printfpp("z_near = %f z_far = %f",z_near,z_far) << endl;
    double min_z = inf, max_z = -inf;
    bool hand_occluded = false;
    
    // register the image buffers
    //resize(rndZ,rndZ,Z.size());
    //resize(rndRGB,rndRGB,RGB.size());
    assert(RGB.size() == RGB.size());
    assert(Z.size() == Z.size());
    segmentation = Mat(Z.rows,Z.cols,DataType<uchar>::type,Scalar::all(0));
    bool invert = false; //hand_renderer.z_inverted();
    for(int rIter = 0; rIter < Z.rows; rIter++)
      for(int cIter = 0; cIter < Z.cols; cIter++)
      {
	// read data
	float rnd_z = Z.at<float>(rIter,cIter);
	Vec3b&rnd_rgb = RGB.at<Vec3b>(rIter,cIter);
	float&out_z = init_Z.at<float>(rIter,cIter);
	Vec3b&out_rgb = init_RGB.at<Vec3b>(rIter,cIter);
	
	float cm_z = register_one_point(rnd_z, invert);
	
	// check for hand occlusion
	if(rnd_rgb != Vec3b(0,0,0) && cm_z > out_z)
	{
	  hand_occluded = true;
	}	
	
	// update segmentation mask
	if(rnd_z < inf && HandRegion::quantize(rnd_rgb) != HandRegion::quantize(HandRegion::wrist_color()))
	  segmentation.at<uchar>(rIter,cIter) = 255;
	
	//if(goodNumber(rnd_z))
	  //cout << printfpp("rnd_z = %f world_z = %f cm_z = %f",rnd_z,world_z,cm_z) << endl;	
	    
	merge_one_pix(cm_z, out_z,rnd_rgb,out_rgb,min_z, max_z);
      }  
         
    // post proc segmentation
    segmentation = imopen(segmentation);

    RGB = init_RGB;
    Z = init_Z;
    cout << printfpp("z_range(cm): %f",max_z-min_z) << endl;
    return hand_occluded;
  }  
  
  void LibHandRenderer::set_flip_lr(bool flip_lr)
  {
    this->flip_lr = flip_lr;
  }
  
  bool LibHandRenderer::get_flip_lr()
  {
    return flip_lr;
  }
  
  double LibHandRenderer::fromDepthCM(double cm_depth) const
  {
    double sf = world_size_randomization_sf*5;
    double isf = 1.0/sf;
    return isf*cm_depth;
  }

  double LibHandRenderer::toDepthCM(double libhand_depth) const
  {
    /* world coords (cm) / world coords (hand model) */
    double sf = world_size_randomization_sf*5;
    return sf*libhand_depth;
  }  
  
  float LibHandRenderer::register_one_point(
    float rnd_z, bool invert
  )
  {    
    // map from frustrum [0,1] to world coordinates
    // float world_z = (1-rnd_z)*z_near + rnd_z*z_far;
    float world_z = rnd_z; // if unprojected in unprojectZ()
    
    // correct units
    /* world coords (cm) / world coords (hand model) */
    double sf = toDepthCM(1.0); 
    double cm_z = world_z - cam_spec.r; // center at 0
    cm_z = cm_z * sf;
    if(invert)
      cm_z *= -1;
    cm_z += sf*cam_spec.r;
    
    return cm_z;
  }  
  
  Mat LibHandRenderer::render_only()
  {
    lock_guard<recursive_mutex> l(libhand_exclusion);
    commit_model();
    hand_renderer.RenderHand();
    read_rgb_buffer();
    return RGB;
  }

  Mat& LibHandRenderer::getRGB()
  {
    return RGB;
  }
  
  Mat& LibHandRenderer::getSegmentation()
  {
    return segmentation;
  }

  CustomCamera& LibHandRenderer::getCamera() 
  {
    return camera;
  }

  void LibHandRenderer::set_flip_world_size_ratio(double world_size_ratio)
  {
    this->world_size_randomization_sf = world_size_ratio;
  }

  libhand::HandRenderer::JointPositionMap& LibHandRenderer::getJointPositionMap()
  {
    commit_model();
    return jointPositionMap;
  }

  bool LibHandRenderer::render()
  {
    lock_guard<recursive_mutex> l(libhand_exclusion);
    commit_model();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    hand_renderer.RenderHand();
    glFinish();
    
    // walk the bones in the rendering...
    read_joint_positions();
    // Now try to read from the GL buffers
    //log_im("rndRGB",rndRGB);
    read_rgb_buffer();
    read_depth_buffer(Z);
    glFinish();
    image_safe("raw_RGB",RGB);
    imageeq("raw_Depth",Z);    
    synth_one_register();
    
    // DEBUG
    image_safe("RGB",RGB);
    imageeq("Depth",Z);    
    return true;
  }
#endif  

  ///
  /// SECTION: HandRegion
  ///
  HandRegion::HandRegion(Vec3b color) : color(color)
  {
  }

  int fourCode(uchar b)
  {
    if(b < 42)
      return 0;
    else if(b < 127)
      return 1;
    else if(b < 212)
      return 2;
    else
      return 3;
  }
  
  Vec3b fourCode(Vec3b color)
  {
    return Vec3b(fourCode(color[0]),fourCode(color[1]),fourCode(color[2]));
  }

  Vec3b HandRegion::quantize(Vec3b src)
  {
    return 85*fourCode(src);
  }
  
  bool HandRegion::is_finger() const
  {
    if(is_arm())
      return false; // arm
    else if(fourCode(color) == Vec3b(0,1,2))
      return false; // outer hand
    else if(fourCode(color) == Vec3b(0,2,3))
      return false; // inner hand
    else if(fourCode(color) == Vec3b(0,2,2))
      return false; // finger base
    else
      return true; // finger of some kind
  }

  bool HandRegion::is_arm() const
  {
    return fourCode(color) == Vec3b(3,3,0);
  }

  bool HandRegion::is_background(Vec3b color)
  {
    return color == background_color() or color == Vec3b(0,0,0);
  }

  Vec3b HandRegion::background_color()
  {
    return Vec3b(0x40,0x40,0x40);
  }

  Vec3b HandRegion::phalan_color(int digit, int joint)
  {
    auto color_id = [&](int index)
    {
      double weight = 1 - index/3.0;
      return cv::saturate_cast<uint8_t>(weight*255);
    };

    if(digit == 1)
    {
      return Vec3b(0,0,color_id(joint));
    }
    else if(digit == 2)
    {
      return Vec3b(color_id(joint),0,0);
    }
    else if(digit == 3)
    {
      return Vec3b(color_id(joint),color_id(joint),color_id(joint));
    }
    else if(digit == 4)
    {
      return Vec3b(0,color_id(joint),0);
    }
    else if(digit == 5)
    {
      return Vec3b(color_id(joint),0,color_id(joint));
    }
    else
      throw std::runtime_error("Bad digit");
  }

  Vec3b HandRegion::wrist_color()
  {
    return Vec3b(255,255,0);
  }

  Vec3b HandRegion::part_color_greg(const string&part_name)
  {
    if(part_name == "wrist")
      return Vec3b(0x00,0x70,0xFF);
    else if(part_name == "dist_phalan_1")
      return Vec3b(0x00,0x9F,0xFF);
    else if(part_name == "dist_phalan_2")
      return Vec3b(0x30,0xFF,0xCF);
    else if(part_name == "dist_phalan_3")
      return Vec3b(0xBF,0xFF,0x40);
    else if(part_name == "dist_phalan_4")
      return Vec3b(0xFF,0xAF,0x00);
    else if(part_name == "dist_phalan_5")
      return Vec3b(0xFF,0x20,0x00);
    throw std::logic_error("bad part name");
  }

  Vec3b HandRegion::part_color_libhand(const string&part_name)
  {
    if(part_name == "wrist")
      return wrist_color();

    static boost::regex digit_regex("\\d+");
    static boost::regex dist_regex(".*dist.*");
    static boost::regex inter_regex(".*inter.*");
    static boost::regex proxi_regex(".*proxi.*");

    std::vector<std::string> numbers = deformable_depth::regex_match(part_name, digit_regex);
    if(numbers.size() > 0)
    {
      int number = fromString<int>(numbers.at(0));
      if(boost::regex_match(part_name,dist_regex))	
	return phalan_color(number,0);
      else if(boost::regex_match(part_name,inter_regex))
	return phalan_color(number,1);
      else if(boost::regex_match(part_name,proxi_regex))
	return phalan_color(number,2);
    }

    return Vec3b(255,255,255);
  }

  Vec3b HandRegion::part_color(const string&part_name)
  {
    if(g_params.require("DATASET") == "EGOCENTRIC_SYNTH")
      return part_color_greg(part_name);
    else
      return part_color_libhand(part_name);
  }

  static Vec3b drawRegions_nn(map<double,string>&points_by_dists)
  {
    return HandRegion::part_color(points_by_dists.begin()->second);
  }

  static Vec3b drawRegions_shade(int xIter, int yIter,map<string,AnnotationBoundingBox>&poss,map<double,string>&points_by_dists)
  {
    Vec3b pixel;
    double total_dist = 0;
    double total_weight = 0;
    Vec3d total_color(0,0,0);
    for(auto & pt : points_by_dists)
    {
      string second_part  = pt.second;
      Point2d c2 = rectCenter(poss.at(second_part));
      double dist2 = std::pow((std::pow(c2.x - xIter,2) + std::pow(c2.y - yIter,2)),2.5);
      total_dist += dist2;
      if(dist2 > 1)
      {
	total_color += Vec3d(HandRegion::part_color(second_part))/dist2;
	total_weight += 1/dist2;
      }
      else
      {
	total_color = HandRegion::part_color(second_part);
	total_weight = 1;
	break;
      }
    }
    pixel = total_color/total_weight;
    return pixel;
  }

  Mat drawRegions(MetaData&md)
  {
    shared_ptr<ImRGBZ> im = md.load_im();    
    auto poss = md.get_positives();
    cout << "drawRegions(parts) = ";
    for(auto && part : poss)
    {
      cout << part.first << " ";
    }
    cout << endl;
    AnnotationBoundingBox hand_abb = poss.at("HandBB");
    if(not goodNumber(hand_abb.depth))
      hand_abb.depth = manifoldFn_default(*im,hand_abb).at(0);

    auto closest_abb = [&](int x, int y)
    {     
      map<double,string> dists_to_points;
      
      for(auto && part : poss)
	if(part.first != "HandBB")
	{
	  Point2d c1 = rectCenter(part.second);	  

	  double dist = std::pow(c1.x - x,2) + std::pow(c1.y - y,2);
	  dists_to_points[dist] = part.first;
	}

      return dists_to_points;
    };

    Mat seg = im->RGB.clone();

    for(int yIter = 0; yIter < seg.rows; yIter++)
      for(int xIter = 0; xIter < seg.cols; xIter++)
      {
	float z = im->Z.at<float>(yIter,xIter);
	Vec3b&pixel = seg.at<Vec3b>(yIter,xIter);

	if(z > hand_abb.depth + params::obj_depth())
	  pixel = HandRegion::background_color();
	else
	{
	  auto points_by_dists = closest_abb(xIter,yIter);
	  pixel = drawRegions_nn(points_by_dists);
	}
      }

    return seg;
  }

#ifdef DD_ENABLE_HAND_SYNTH
  // singleton battery
  namespace renderers
  {
    static LibHandRenderer* libhand_allocate(LibHandRenderer*& cache,string path)
    {
      static mutex m; lock_guard<mutex> l(m);
      
      static thread_local bool registered = false;
      if(cache && !registered)
      {
        // make sure the thread is registered with the Ogre render systems?
	Ogre::Root::getSingleton().getRenderSystem()->registerThread();
	registered = true;
      }

      // make sure the thing is created
      if(!cache)
      {
	cache = new LibHandRenderer(
	  path,Size(params::depth_hRes,params::depth_vRes));
	registered = true;
      }
      return cache;
    }

    LibHandRenderer* intersection()
    {
      static LibHandRenderer* cache = nullptr;
      return libhand_allocate(cache,"/home/jsupanci/workspace/libhand-0.9/intersection_model/scene_spec.yml");
    }
    LibHandRenderer* no_arm()
    {
      static LibHandRenderer* cache = nullptr;
      return libhand_allocate(cache,"/home/jsupanci/workspace/libhand-0.9/hand_model_no_arm/scene_spec.yml");
    }
    LibHandRenderer* with_arm()
    {
      static LibHandRenderer* cache = nullptr;
      return libhand_allocate(cache,"/home/jsupanci/workspace/libhand-0.9/new_hand_model/scene_spec.yml");
    }
    LibHandRenderer* segmentation()
    {
      static LibHandRenderer* cache = nullptr;
      return libhand_allocate(cache,"/home/jsupanci/workspace/libhand-0.9/seg_hand_model/scene_spec.yml");
    }
  }
#endif
}


