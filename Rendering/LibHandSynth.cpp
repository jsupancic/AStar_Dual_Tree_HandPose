/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "LibHandSynth.hpp"
#include "util.hpp"
#include "Renderer.hpp"
#include "Eval.hpp"
#include "util_file.hpp"
#include "Log.hpp"
#include "Poselet.hpp"
#include "Annotation.hpp"
#include "OcclusionReasoning.hpp"
#include <boost/filesystem.hpp>

#ifdef DD_ENABLE_HAND_SYNTH
#include <OgreVector3.h>
#include <hand_pose.h>
#include <hand_renderer.h>
#include <scene_spec.h>
#endif

#include "InverseKinematics.hpp"
#include "CSG.hpp"

namespace deformable_depth
{
#ifdef DD_ENABLE_HAND_SYNTH
  using namespace libhand;
#endif
  using params::PI;
  
  ///
  /// SECTION: LibHandSynthesizer
  ///
#ifdef DD_ENABLE_HAND_SYNTH
  
  void LibHandSynthesizer::set_model(PoseRegressionPoint& pose, 
				     Size bgSize, int component_flags)
  {
    bool commit = true;
    
    for(auto & model : models)
    {
      if(component_flags & COMP_FLAG_CAM)
	model.second->get_cam_spec() = pose.cam_spec;
      if(component_flags & COMP_FLAG_HAND_POSE)
	model.second->get_hand_pose() = pose.hand_pose;
      if(component_flags & COMP_FLAG_WORLD_SIZE)
      {
	model.second->set_flip_world_size_ratio(pose.world_size_randomization_sf);
	if(component_flags == COMP_FLAG_WORLD_SIZE)
	  commit = false;
      }
      if(component_flags & COMP_FLAG_BG)
      {
	bg_tx = bg_ty = 0;
	bg_theta_rad = 0;
      }
    }
  }
  
  void LibHandSynthesizer::set_hand_pose(FullHandPose& hand_pose)
  {
    for(auto & model : models)
      model.second->get_hand_pose() = hand_pose;
  }
  
  FullHandPose LibHandSynthesizer::get_hand_pose()
  {
    auto&hand_pose = models.begin()->second->get_hand_pose();
    auto&scene_spec = models.begin()->second->get_scene_spec();
    if(hand_pose.num_joints() == scene_spec.num_bones())
      return hand_pose;
    else
      return FullHandPose(scene_spec.num_bones());
  }
  
  libhand::FullHandPose perturbation(const libhand::FullHandPose&in,double var)
  {
    auto hand_pose = in;    
    LibHandJointRanges ranges;
    for(int iter = 0; iter < LibHandSynthesizer::IDX_MAX_VALID_FINGER_JOINTS; iter++)
    {
      // perturb
      hand_pose.bend(iter) = hand_pose.bend(iter) + sample_in_range(-var*PI,var*PI); 
      hand_pose.side(iter) = hand_pose.side(iter) + sample_in_range(-var*PI,var*PI); 
      hand_pose.twist(iter) = hand_pose.twist(iter) + sample_in_range(-var*PI,var*PI); 
      // and clamp
      hand_pose.bend(iter) = clamp<double>(ranges.bend_min(iter),hand_pose.bend(iter),ranges.bend_max(iter));
      hand_pose.side(iter) = clamp<double>(ranges.side_min(iter),hand_pose.side(iter),ranges.side_max(iter));
      hand_pose.twist(iter) = clamp<double>(ranges.twist_min(iter),hand_pose.twist(iter),ranges.twist_max(iter));
    }    
    
    return hand_pose;
  }
  
  libhand::HandCameraSpec LibHandSynthesizer::perturbation(const libhand::HandCameraSpec&in,double cam_var)
  {
    auto cam_spec = in;
    cam_spec.theta = cam_spec.theta + sample_in_range(-cam_var*PI,cam_var*PI);
    cam_spec.phi = cam_spec.phi + sample_in_range(-cam_var*PI,cam_var*PI);
    cam_spec.tilt = cam_spec.tilt + sample_in_range(-cam_var*PI,cam_var*PI);
    cam_spec.r = models.begin()->second->fromDepthCM(
      sample_in_range(get_param<double,SYNTH_PERTURB_R_MIN>(string("SYNTH_PERTURB_R_MIN")),
		      get_param<double,SYNTH_PERTURB_R_MAX>(string("SYNTH_PERTURB_R_MAX"))));   
    return cam_spec;
  }
  
  void LibHandSynthesizer::perturb_model()
  {
    // get a hand pose 
    auto&hand_pose = models.begin()->second->get_hand_pose();
    auto&cam_spec = models.begin()->second->get_cam_spec();
    
    // perturb the hand pose
    hand_pose = deformable_depth::perturbation(hand_pose);
    
    // perturb the camera position
    cam_spec = this->perturbation(cam_spec);
    
    // randomize the size
    double world_size_randomization_sf = sample_in_range(.7,1.3);    
    
    // commit the libhand
    for(auto & model : models)
    {
      model.second->get_cam_spec() = cam_spec;
      model.second->get_hand_pose() = hand_pose;
      model.second->set_flip_world_size_ratio(world_size_randomization_sf);
    }
  }
  
  void LibHandSynthesizer::randomize_background(Size bgSize)
  {
    bg_tx = sample_in_range(-bgSize.width/2,bgSize.width/2);
    bg_ty = sample_in_range(-bgSize.height/2,bgSize.height/2);
    bg_theta_rad = 0; //sample_in_range(-params::PI/4,params::PI/4);
  }
  
  HandCameraSpec LibHandSynthesizer::random_cam_spec()
  {
    libhand::HandCameraSpec cam_spec;
    do
    {
      cam_spec.theta = sample_in_range(0*PI,2*PI);  // 0.8*PI,2.2*PI
    } while( (.5*PI - .5 < cam_spec.theta && cam_spec.theta <  .5*PI + .5));
    cout << "theta = " << cam_spec.theta  << endl;
    cam_spec.phi = sample_in_range(0,2*PI);
    cout << "phi   = " << cam_spec.phi << endl;
    cam_spec.tilt = 9.3899;
    //cam_spec.tilt = sample_in_range(0,2*PI);
    cout << "tilt  = " << cam_spec.tilt << endl;
    cam_spec.r = models.begin()->second->fromDepthCM(
      sample_in_range(
	get_param<double,SYNTH_PERTURB_R_MIN>(string("SYNTH_PERTURB_R_MIN")),
	get_param<double,SYNTH_PERTURB_R_MAX>(string("SYNTH_PERTURB_R_MAX"))));   
    return cam_spec;
  }
  
  void LibHandSynthesizer::randomize_camera()
  {
    auto cam_spec = random_cam_spec(); //models.begin()->second->get_cam_spec();

    for(auto & model : models)
      model.second->get_cam_spec() = cam_spec;
  }
  

  LibHandJointRanges::LibHandJointRanges() : 
    bend_mins(NUM_JOINTS), 
    bend_maxes(NUM_JOINTS), 
    side_mins(NUM_JOINTS), 
    side_maxes(NUM_JOINTS), 
    twist_mins(NUM_JOINTS), 
    twist_maxes(NUM_JOINTS),
    elongation_mins(NUM_JOINTS),
    elongation_maxes(NUM_JOINTS)
  {
    // init all to zero
    for(int iter = 0; iter < NUM_JOINTS; ++iter)
    {
      bend_mins[iter] = 0;
      bend_maxes[iter] = 0;
      side_mins[iter] = 0;
      side_maxes[iter] = 0;
      twist_mins[iter] = 0;
      twist_maxes[iter] = 0;    
      elongation_mins[iter] = 1;
      elongation_maxes[iter] = 1;
    }
    
    for(int jointIter = 0; jointIter <= LibHandSynthesizer::IDX_MAX_VALID_FINGER_JOINTS; jointIter++)
    {
      bend_mins[jointIter] = -PI/2;
      bend_maxes[jointIter] = PI/7;
    }
    // set the finger side angles
    for(int jointIter = 0; jointIter <= 9; jointIter += 3)
    {
      side_mins[jointIter] = -PI/8;
      side_maxes[jointIter] = PI/8;
    }
    // thumb articulation is important
    bend_mins[12] = deg2rad(-60);
    bend_maxes[12] = deg2rad(30);
    side_mins[12] = deg2rad(-40);
    side_maxes[12] = deg2rad(70);
    elongation_mins[12] = .8;
    elongation_mins[12] = 1.2;
    //hand_pose.twist(12) = sample_in_range(-1,1);
    bend_mins[13] = deg2rad(-60);
    bend_maxes[13] = deg2rad(35);
    side_mins[13] = deg2rad(-10);
    side_maxes[13] = deg2rad(30);
    //hand_pose.twist(13) = sample_in_range();    
    
    // the meta-carpal joint is also quite important...
    bend_mins[15] = deg2rad(-90);
    bend_maxes[15] = deg2rad(90);
    side_mins[15] = deg2rad(-30);
    side_maxes[15] = deg2rad(45);    
  }
  
  FullHandPose random_hand_pose(int num_bones)
  {
    // now we setup the pose how we want it (randomly somehow)
    auto hand_pose = FullHandPose(num_bones);
    LibHandJointRanges ranges;
    for(int iter = 0; iter < LibHandJointRanges::NUM_JOINTS; iter++)
    {
      hand_pose.bend(iter) = sample_in_range(ranges.bend_min(iter),ranges.bend_max(iter));
      hand_pose.side(iter) = sample_in_range(ranges.side_min(iter),ranges.side_max(iter));
      hand_pose.twist(iter) = sample_in_range(ranges.twist_min(iter),ranges.twist_max(iter));
      hand_pose.elongation(iter) = sample_in_range(ranges.elongation_min(iter),ranges.elongation_max(iter));
      hand_pose.dilation(iter) = 1;
      hand_pose.swelling(iter) = 1;
      hand_pose.elongation(iter) = sample_in_range(ranges.elongation_min(iter),ranges.elongation_max(iter));
    }
      
    return hand_pose;
  }
  
  void LibHandSynthesizer::randomize_model(Size bgSize)
  {    
    // now we setup the camera (randomly?)
    randomize_camera();
    
    // randomize the world size of the hand
    double world_size_randomization_sf = sample_in_range(.7,1.3);    
    
    // randomize the background parameters
    randomize_background(bgSize);
    
    // randomize the pose
    auto random_pose = random_hand_pose(models.begin()->second->get_scene_spec().num_bones());
    
    for(auto & model : models)
    {
      model.second->get_hand_pose() = random_pose;
      model.second->set_flip_world_size_ratio(world_size_randomization_sf);
    }    
  }
                
  void LibHandSynthesizer::synth_one_debug_preregistration()
  {
  }
  
  void LibHandSynthesizer::synth_one_debug(
    Mat&RGB, Mat&Z, shared_ptr<ImRGBZ>&bg_im, Rect_<double> handBB)
  {
    // DEBUG, draw joints
    Mat displayRGB = RGB.clone();
    for(auto&&joint : getJointPositionMap())
    {
      Vec3d pos = joint.second;
      Point pos2d(pos[0],pos[1]);
      circle(displayRGB,pos2d,1,Scalar(0,0,255),-1);
    }        
    rectangle(displayRGB,handBB.tl(),handBB.br(),Scalar(255,0,0));
    
    // show result
    //imageeq("pre-merge Z",rndZ); 
    //image_safe("libHand RGB",displayRGB);
    //imageeq("libHand Z",Z); 
    vector<Mat> viss;
    viss.push_back(fingers);
    for(auto & model : models)
    {
      //viss.push_back(model.second->getRGB());
      viss.push_back(
	vertCat(model.second->getRGB(),imageeq("",model.second->getDepth(),false,false)));
    }
    log_im("rendered",tileCat(viss));
    
    // report the world area for sanity
    if(0 <= handBB.x && 0 <= handBB.width && handBB.x + 
      handBB.width <= Z.cols && 0 <= handBB.y && 0 <= handBB.height 
      && handBB.y + handBB.height <= Z.rows)
    {
      double z = extrema(Z(handBB)).min;
      double world_area = bg_im->camera.worldAreaForImageArea(z,handBB);
      cout << "world area: " << world_area << endl;
    }
    else
      cout << "bad bb: no world area" << endl;    
  }
  
  // return true (filter yes) if there aren't enough interesting areas
  // visible to call it a hand (and not, say an elbow)
  bool LibHandSynthesizer::filter_seg_area()
  {
    Mat sem = models.at("segmented")->getRGB();
    Mat seg = models.at("armless")->getSegmentation();
    fingers = Mat(sem.rows,sem.cols,DataType<Vec3b>::type,Scalar::all(0)).clone();
    
    double hand_area = 0;
    double finger_area = 0;
    for(int rIter = 0; rIter < sem.rows; rIter++)
      for(int cIter = 0; cIter < sem.cols; cIter++)
	if(seg.at<uchar>(rIter,cIter) > 100)
	{
	  Vec3b s = sem.at<Vec3b>(rIter,cIter);
	  hand_area ++;
	  fingers.at<Vec3b>(rIter,cIter) = Vec3b(255,0,0);
	  if(HandRegion(s).is_finger())
	  {
	    finger_area++;
	    fingers.at<Vec3b>(rIter,cIter) = Vec3b(0,255,0);
	  }
	}
    
    double finger_ratio = finger_area/hand_area;
    if(g_params.option_is_set("SYNTH_FINGER_AREA_FILTER") && finger_ratio < 1.0/3.0)
    {
      cout << "REJECT FINGER AREA " << finger_ratio << endl;
      log_file << "REJECT FINGER AREA " << finger_ratio << endl;
      return true;
    }
    cout << "ACCEPT FINGER AREA " << finger_ratio << endl;
    return false;
  }
  
  bool is_egocentric(const map<string,Vec3d>&hand_pose)
  {
    Vec3d carpal = hand_pose.at("carpals");
    Vec3d center = hand_pose.at("Bone.002");
    //cout << "carpal: " << carpal << endl;
    //cout << "center: " << center << endl;
    Vec3d up_dir = center-carpal;    
    
    return up_dir[2] > 0;
  }
  
  // introduce a dataset bias/prior
  bool LibHandSynthesizer::filter_post_gen()
  {
    // reject certain poses (e.g. downward facing hands)
    Vec3d carpal = getJointPositionMap()["carpals"];
    Vec3d center = getJointPositionMap()["Bone.002"];
    cout << "carpal: " << carpal << endl;
    cout << "center: " << center << endl;
    Vec3d up_dir = center-carpal;
    double v_x = (up_dir)[0];
    double v_y = -(up_dir)[1];
    double v_z = (up_dir)[2];
    double deg_v_xy = (180.0/params::PI) * (::atan2(v_y,v_x) + params::PI); // 0 to 360 degrees
    cout << printfpp("vx = %f vy = %f ori = %f",v_x,v_y,deg_v_xy) << endl;
    if(0 < deg_v_xy && deg_v_xy < 180)
    {
      if(filter & FILTER_DOWNWARD)
      {
	cout << "REJECT DOWNWARD" << endl;
	log_file << "REJECT DOWNWARD" << endl;	
	return true;
      }
    }    
    
    // reject hands which are from an ego-centric viewpoint
    double deg_v_xz = (180.0/params::PI) * (::atan2(v_z,v_x) + params::PI);
    double deg_v_yz = (180.0/params::PI) * (::atan2(v_z,v_y) + params::PI);
    //if((0 < deg_v_xz && deg_v_xz < 180) && (0 < deg_v_yz && deg_v_yz < 180))
    if(is_egocentric(getJointPositionMap()))
    {
      if(filter & FILTER_EGO_CENTRIC)
      {
	cout << "REJECT EGO-CENTRIC" << endl;
	log_file << "REJECT EGO-CENTRIC" << endl;	
	return true;
      }
    }
    else
    {
      if(filter & FILTER_NOT_EGO_CENTRIC)
      {
	cout << "REJECT NOT EGO-CENTRIC" << endl;
	log_file << "REJECT NOT EGO-CENTRIC" << endl;	
	return true;
      }
    }
    
    // reject hands which aren't facing the camera?
    assert(getJointPositionMap().find("root") != getJointPositionMap().end());
    Vec3d root = getJointPositionMap()["root"]; 
    assert(getJointPositionMap().find("finger1joint1") != getJointPositionMap().end());
    Vec3d finger1joint1 = getJointPositionMap()["finger1joint1"];
    Vec3d lr_dir = finger1joint1 - center;
    Vec3d normal = lr_dir.cross(up_dir);
    if(normal[2] < 0)
    {      
      if(filter & FILTER_BACKWARD)
      {
	cout << "REJECT BACKWARD" << endl;
	log_file << "REJECT BACKWARD" << endl;	
	return true;
      }
    }
    
    cout << "ACCEPT ORI" << endl;
    log_file << "ACCEPT ORI" << endl;
    return false;
  }
  
  bool LibHandSynthesizer::filter_self_intersecting()
  {
//     auto & jpm = getJointPositionMap();
//     CSG_Workspace csg_worksapce(40,40,40,jpm);
//     int color_id = 0;
//     // draw the fingers
//     for(int fingerId = 1; fingerId <= 4; ++fingerId)
//       for(int joint = 1; joint < 2; joint++)
//       {
// 	Vec3d p1 = getJointPositionMap()[printfpp("finger%djoint%d",fingerId,joint)];
// 	Vec3d p2 = getJointPositionMap()[printfpp("finger%djoint%d",fingerId,joint+1)];
// 	csg_worksapce.writeLine(p1,p1,color_id++);
//       }
//     csg_worksapce.paint();
    
    Mat passes = models["intersection"]->getRGB();
    self_intersections = passes.clone();
    for(int yIter = 0; yIter < passes.rows; yIter++)
      for(int xIter = 0; xIter < passes.cols; xIter++)
      {
	Vec3b&ihere = self_intersections.at<Vec3b>(yIter,xIter);
	Vec3b&phere = passes.at<Vec3b>(yIter,xIter);
	if(phere[0] / 32 == phere[1]/32)
	  ihere = Vec3b(0,0,0);
	else
	  ihere = Vec3b(0,0,255);
      }
    //log_im("self_intersections",self_intersections);
    
    return false;
  }
  
  SceneSpec& LibHandSynthesizer::get_scene_spec()
  {
    return models.begin()->second->get_scene_spec();
  }

  void LibHandSynthesizer::read_joint_positions()
  {    
    for(auto & model : models)
    {
      model.second->render();
    }
    
    for(auto & model : models)
    {
      model.second->read_joint_positions();
    }    
  }
  
  Mat LibHandSynthesizer::render_only()
  {
    for(auto & model : models)
    {
      model.second->render_only();
    }
    return models.begin()->second->getRGB();
  }
  
  void LibHandSynthesizer::set_filter(int filter)
  {
    this->filter = static_cast<ExemplarFilters>(filter);
  }
  
  void LibHandSynthesizer::set_synth_bg(bool synth_bg)
  {
    this->synthBG = synth_bg;
  }
  
  shared_ptr< LibHandMetadata > LibHandSynthesizer::synth_one(bool read_only)
  {
    shared_ptr< LibHandMetadata > result = nullptr;
    result = try_synth_one(read_only);
    while(result == nullptr)
    {
      randomize_camera();
      result = try_synth_one(read_only);
    }
    return result;
  }

  Rect compute_hand_bb(LibHandRenderer&renderer,Mat&Z,LibHandSynthesizer::ExemplarFilters filter)
  {
    Vec3b arm_color(0x6B,0x7D,0xB8); // B87D6B
    Vec3b black(0,0,0);
    Rect handBB = bbWhere(renderer.getSegmentation(),[&](Mat&im,int y, int x)
      {
	assert(im.type() == DataType<uchar>::type);
	return im.at<uchar>(y,x) > 100;
      });
    cout << "HandBB = " << handBB << endl;
    log_file << "HandBB = " << handBB << endl;
    if(handBB.tl().x <= 0 || handBB.tl().y <= 0 || 
      handBB.br().x >= Z.cols - 1 || 
      handBB.br().y >= Z.rows - 1)
    {
      cout << "REJECT BAD HANDBB" << endl;
      log_file << "REJECT BAD HANDBB" << endl;
      if(filter & LibHandSynthesizer::FILTER_BAD_BB)
	return Rect();
    }
    Rect handBB_naive = rectResize(handBB,1.2,1.2);
    double side_inc = std::min<double>(
      handBB_naive.width-handBB.width,handBB_naive.height-handBB.height);
    Size newSize(handBB.width + side_inc, handBB.height + side_inc);
    handBB = rectFromCenter(rectCenter(handBB),newSize);
    if(handBB.tl().x <= 0 || handBB.tl().y <= 0 || 
      handBB.br().x >= Z.cols - 1 || 
      handBB.br().y >= Z.rows - 1)
    {
      cout << "REJECT BAD HANDBB" << endl;
      log_file << "REJECT BAD HANDBB" << endl;
      if(filter & LibHandSynthesizer::FILTER_BAD_BB)
	return Rect();
    }

    return handBB;
  }

  
  // sliding and twisting look very painfull...
  // for now just alter bend
  shared_ptr< LibHandMetadata > LibHandSynthesizer::try_synth_one(bool read_only)
  {
    cout << "===== SYNTH_ONE =====" << endl;
    
    // randomize
    random_shuffle(bgs.begin(),bgs.end());
    MetaData&bg = *bgs[0];
    
    // get the image from the old metadata
    shared_ptr<ImRGBZ> bg_im = bg.load_im();
    Z = bg_im->Z.clone();
    RGB = bg_im->RGB.clone();
    if(synthBG)
    {
      // apply a random affine transform to avoid overfitting
      Mat AT = affine_transform(bg_tx, bg_ty, bg_theta_rad);
      warpAffine(Z,Z,AT,Z.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT101);
      warpAffine(RGB,RGB,AT,RGB.size(),cv::INTER_LINEAR,cv::BORDER_REFLECT101);
    }
    else
    {     
      Z = inf;
    }
    
    for(auto & model : models)
    {
      // construct the new Metadata
      //synth_one_debug_preregistration();
      if(!model.second->render())
      {
	cout << "REGISTRATION FAILED " << model.first << endl;
	log_file << "REGISTRATION FAILED " << model.first << endl;
	if(filter & FILTER_BAD_REGISTRATION)
	  return nullptr;
      }
    }
    // filter post-generated images.
    if(filter && filter_post_gen())
      return nullptr;    
    if(filter_seg_area())
      return nullptr;
    if(filter_self_intersecting())
      return nullptr;
    
    // fill out the backgournd
    float rndOcc, capOcc;
    if(synthBG && !merge(models["armed"]->getRGB(), models["armed"]->getDepth(), 
	     RGB, Z, RGB, Z,rndOcc, capOcc))
    {
      cout << "MERGING FAILED" << endl;
      log_file << "MERGING FAILED" << endl;      
      return nullptr;
    }
    else if(!synthBG)
    {
      RGB = models["armed"]->getRGB().clone();
      Z = models["armed"]->getDepth().clone();
    }
    
    // compute the hand bb.
    LibHandRenderer&armless = *models["armless"];
    Rect handBB = compute_hand_bb(armless,Z,filter);
    if(handBB == Rect())
      return nullptr;
    synth_one_debug(RGB, Z, bg_im, handBB);
    
    //waitKey_safe(0);
    imageeq("exporting depth",Z);
    string filename; 
    int id;
    shared_ptr<LibHandMetadata> metadata(
      new LibHandMetadata(
	alloc_filename(filename,id),RGB,Z,
	models["segmented"]->getSegmentation(),
	models["segmented"]->getRGB(),
	handBB,getJointPositionMap(),models["segmented"]->getCamera(),read_only));
        
    return metadata;
  }
    
  Mat& LibHandSynthesizer::getDepth()
  {
    return Z;
  }

  Mat& LibHandSynthesizer::getRGB()
  {
    return RGB;
  }
  
  Mat& LibHandSynthesizer::getRenderedDepth()
  {
    return models["armed"]->getDepth();
  }

  Mat& LibHandSynthesizer::getRenderedRGB()
  {
    return models["armed"]->getRGB();
  }
  
  Mat& LibHandSynthesizer::getSegmentation()
  {
    return models["armed"]->getSegmentation();
  }
  
  libhand::HandRenderer::JointPositionMap& LibHandSynthesizer::getJointPositionMap()
  {
    return models["armed"]->get_jointPositionMap();
  }
  
  const HandCameraSpec LibHandSynthesizer::get_cam_spec()
  {
    return models.begin()->second->get_cam_spec();
  }

  void LibHandSynthesizer::set_flip_lr(bool flip_lr)
  {
    for(auto & model : models)
      model.second->set_flip_lr(flip_lr);
  }

  void LibHandSynthesizer::set_cam_spec(HandCameraSpec&spec)
  {
    for(auto & model : models)
      model.second->get_cam_spec() = spec;
  }

  string LibHandSynthesizer::alloc_filename(string&filename, int&file_id_out)
  {
    static atomic<unsigned long> file_id(0);
    
    string pattern = save_dir + "/gen%lu.yaml.gz";
    filename;
    int fd = alloc_file_atomic_unique(pattern,filename,file_id);
    close(fd);
    file_id_out = file_id - 1;
    
    return filename;
  }
  
  LibHandSynthesizer::LibHandSynthesizer(string save_dir) : 
    save_dir(save_dir), filter(FILTER_DEFAULT), flip_lr(false)
  {
    init();
  }
  
  LibHandSynthesizer::LibHandSynthesizer() : 
    save_dir(params::synthetic_directory()), filter(FILTER_DEFAULT), flip_lr(false)
  {
    init();
  }
  
  void LibHandSynthesizer::init()
  {
    // default parameters
    synthBG = false;
    render_armless = true;
    
    log_file << "LibHandSynthesizer::LibHandSynthesizer begin" << endl;
    // init
    bgs = metadata_build_all("data/Backgrounds/",false);
    
    // image data needed for configuration
    MetaData&bg = *bgs[0];
    shared_ptr<ImRGBZ> bg_im = bg.load_im();
    Mat bgZ = bg_im->Z.clone();    
    
    // load the hand model
    models["intersection"] = renderers::segmentation();//renderers::intersection();
    models["armless"] = renderers::segmentation();//renderers::no_arm();
    models["armed"] = renderers::segmentation();//renderers::with_arm();
    models["segmented"] = renderers::segmentation();
    
    //std::exit(0);
  }
  
  shared_ptr< LibHandMetadata > LibHandSynthesizer::synth_random(bool read_only)
  {
    shared_ptr< LibHandMetadata >  answer;
    Size bgSize(params::depth_hRes,params::depth_vRes);
    while(randomize_model(bgSize), answer = synth_one(read_only), answer == nullptr)
      /*nop*/;
    
    return answer;
  }
#endif
  
  void gen_training_data(int argc, char**argv)
  { 
#ifdef DD_ENABLE_HAND_SYNTH
    // generate one
    LibHandSynthesizer synther(params::out_dir() + "/");	
    int n_clusters = get_param<int,SYNTH_CLUSTERS>("SYNTH_CLUSTERS");
    int examples_per_cluster = get_param<int,SYNTH_EX_PER_CLUSTER>("SYNTH_EX_PER_CLUSTER");
    log_file << safe_printf("gen_training_data: % * % examples",n_clusters,examples_per_cluster) << endl;
    //int number = n_clusters*examples_per_cluster; // n_clusters clusters and examples_per_cluster
    //TaskBlock synth_data("synth_data");
    for(int clusterIter = 0; clusterIter < n_clusters; clusterIter++)
      //synth_data.add_callee([&,clusterIter]()
      {
	for(int iter = 0; iter < examples_per_cluster; iter++)
	{
	  cout << "synthesizing: " << iter << endl;
	  synther.synth_random();
	  //pause();
	}	
      }//);
    //synth_data.execute();
#else
    throw std::runtime_error("Unsupported Method");
#endif
  }    
      
  void visualize_synthetic() 
  {
#ifdef DD_ENABLE_HAND_SYNTH    
    LibHandSynthesizer synther;
    vector<Mat> depth_renders;
    vector<Mat> RGB_renders;
    
    int rows = 4;
    int cols = 4;
    for(int iter = 0; iter < rows*cols; ++iter)
    {
      shared_ptr<LibHandMetadata> example = synther.synth_random(true);
      Mat rndZ = synther.getRenderedDepth();
      Mat depth_vis_here = imageeq("",rndZ,false,false);
      for(int yIter = 0; yIter < depth_vis_here.rows; yIter++)
	for(int xIter = 0; xIter < depth_vis_here.cols; xIter++)
	{
	  // change blue bg to white bg
	  Vec3b&pixel = depth_vis_here.at<Vec3b>(yIter,xIter);
	  if(pixel == Vec3b(255,0,0))
	    pixel = Vec3b(255,255,255);
	}
      Mat RGB_Vis = synther.getRenderedRGB().clone();
      for(int yIter = 0; yIter < RGB_Vis.rows; yIter++)
	for(int xIter = 0; xIter < RGB_Vis.cols; xIter++)
	{
	  // change black bg to white bg
	  Vec3b&pixel = RGB_Vis.at<Vec3b>(yIter,xIter);
	  if(pixel == Vec3b(0,0,0))
	    pixel = Vec3b(255,255,255);
	}	
      log_im("synth1_depth",depth_vis_here);
      log_im("synth1_Rgb",RGB_Vis);
      depth_renders.push_back(depth_vis_here);
      RGB_renders.push_back(RGB_Vis);
    }
    
    log_im(printfpp("%dx%dsynth_depth",rows,cols),tileCat(depth_renders));
    log_im(printfpp("%dx%dsynth_color",rows,cols),tileCat(RGB_renders));
#else
    assert(false);
#endif
  }
}

