/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifdef DD_ENABLE_HAND_SYNTH 
#include "IKSynther.hpp"
#include "InverseKinematics.hpp"
#include "Log.hpp"
#include "util.hpp"
#include <fstream>
#include <iostream>
#include <OgreQuaternion.h>
#include <boost/filesystem/path.hpp>
 
namespace deformable_depth
{
  ///
  /// SECTION: IK_Grasp_Pose
  ///
  double IK_Grasp_Pose::depth() const
  {
    vector<double> depths;
    
    for(auto pair : joint_positions)
     depths.push_back(pair.second[2]); // push the z/depth coordinate
     
    std::sort(depths.begin(),depths.end());
    return depths[depths.size()/2];
  }

  map< string, Vec3d > IK_Grasp_Pose::joint_positions_flip_lr() const
  {
    map<string, Vec3d> flip_map = joint_positions;
    
    for(auto&pair : flip_map)
    {
      pair.second[0] *= -1; // change the sign of the x-coordinate
      require_equal(flip_map.at(pair.first)[0],-joint_positions.at(pair.first)[0]);
    }
    
    return flip_map;
  }
  
  double IK_Grasp_Pose::mean_dist() const
  {
    if(joint_positions.size() == 0)
      return inf;
    
    double total_dist = 0;
    
    for(auto&pair : joint_positions)
    {
      total_dist += pair.second[2];
    }
    
    return total_dist / joint_positions.size();
  }
  
  ///
  /// SECTION IKGraspSynther
  ///
  
  IKGraspSynther::IKGraspSynther() :
    synther((params::out_dir()))
  {
  }
  
  void IKGraspSynther::do_IK_LR_Symmetric(
    IK_Grasp_Pose&pose,
    AbsoluteOrientation&abs_ori,
    PoseRegressionPoint&angles,
    bool&fliped,
    string&lr_mesg)
  {
    AbsoluteOrientation abs_ori_reg, abs_ori_flip;
    PoseRegressionPoint angles_regular = IK_Match(pose.joint_positions,abs_ori_reg,synther);
    PoseRegressionPoint angles_flipped = IK_Match(pose.joint_positions_flip_lr(),abs_ori_flip,synther);
    log_file << printfpp("dist(reg) = %f dist(flip) = %f",abs_ori_reg.distance,abs_ori_flip.distance) << endl;
    if(abs_ori_reg.distance > abs_ori_flip.distance)
    {
      abs_ori = abs_ori_flip;
      angles = angles_flipped;
      synther.set_flip_lr(true);
      fliped = true;
      lr_mesg  = "flipped";
      log_file << "synth_one_example: flip" << endl;
    }
    else
    {
      abs_ori = abs_ori_reg;
      angles = angles_regular;
      synther.set_flip_lr(false);
      fliped = false;
      lr_mesg = "not flipped";
      log_file << "synth_one_example: no flip" << endl;
    }    
  }
  
  shared_ptr<LibHandMetadata> IKGraspSynther::synth_one_example(SynthContext&context,int index)
  {
    Size bgSize(params::depth_hRes,params::depth_vRes);
    
    // randomly select a pose
    bool perturb = false;
    if(index < 0 || index >= grasp_Poses.size())
    {
      perturb = true;
      index = rand()%grasp_Poses.size();
    }
    IK_Grasp_Pose pose = grasp_Poses[index];
    
    // find the joint angles which correspond to these finger positions
    AbsoluteOrientation abs_ori;
    PoseRegressionPoint angles;
    bool fliped;
    string lr_mesg;
    do_IK_LR_Symmetric(pose,abs_ori,angles,fliped,lr_mesg);
    string dist_desc = printfpp("min dist = %f",abs_ori.distance);
    libhand::HandCameraSpec cam_spec = angles.cam_spec;
    string cam_spec_desc0 = printfpp("theta0 = %f phi0 = %f tilt0 = %f",
			 cam_spec.theta,
			 cam_spec.phi,
			 cam_spec.tilt
			);    
    //cam_spec.theta = (-abs_ori.quaternion.yaw());
    //cam_spec.phi = (-abs_ori.quaternion.roll()); 
    //cam_spec.tilt = (abs_ori.quaternion.pitch());// this right?
    synther.set_model(angles,bgSize,LibHandSynthesizer::COMP_FLAG_WORLD_SIZE);
    cam_spec.r = 15; //synther.fromDepthCM(pose.depth());
    string cam_spec_desc = printfpp("theta = %f phi = %f tilt = %f",
			 cam_spec.theta,
			 cam_spec.phi,
			 cam_spec.tilt
			);
    string pyr_desc = printfpp("pitch = %f yaw = %f roll = %f",
			       abs_ori.quaternion.pitch(),
			       abs_ori.quaternion.yaw(),
			       abs_ori.quaternion.roll());
    string str_q0 = Quaternion(cam_spec.GetQuaternion()).toString();
    string str_q1 = Quaternion(abs_ori.quaternion).toString();
    log_file << cam_spec_desc << endl;
    angles.cam_spec = cam_spec;
    
    // render from the regression
    shared_ptr<LibHandMetadata> match_metadata;
    do
    {
      synther.randomize_model(bgSize);
      synther.set_model(angles,bgSize,
			LibHandSynthesizer::COMP_FLAG_HAND_POSE|
		        LibHandSynthesizer::COMP_FLAG_WORLD_SIZE|
			LibHandSynthesizer::COMP_FLAG_CAM);
      if(perturb)
	synther.perturb_model();
      synther.set_filter(
	LibHandSynthesizer::FILTER_BAD_BB|
	LibHandSynthesizer::FILTER_BAD_REGISTRATION);
    } while((!(match_metadata = synther.synth_one(false))));
    
    // create a visualization of the match
    Mat vis_answer = horizCat(imVGA(pose.vis),imVGA(synther.getRGB()));
    vis_answer = vertCat(vis_answer,image_text(
      {lr_mesg,cam_spec_desc0,cam_spec_desc,pyr_desc,str_q0,str_q1,dist_desc}));
    log_im(
      printfpp("IK_Match_%s",
	pose.identifier.c_str()),vis_answer);      
    
    context.pose = &pose;
    return match_metadata;
  }
  
  ///
  /// SECTION: IKSynth w/ Greg's data
  ///
  class GregIKSynther : public IKGraspSynther
  {
  public:
    GregIKSynther(string directory);
  };
  
  IK_Grasp_Pose greg_load_grasp_pose_one(string directory,string filename, bool center)
  {
    IK_Grasp_Pose pose;
    ifstream ifs(filename);
    
    // get the frame number
    auto match_iter = boost::sregex_iterator(
      filename.begin(),filename.end(),boost::regex("frame([0-9]+)"));
    log_file << "Matching Regex to : " << filename << endl;
    string str_frame_num;
    while(match_iter != boost::sregex_iterator())
    {
      str_frame_num = match_iter->str();
      log_file << "Regex_match: " << str_frame_num << endl;
      ++match_iter;
    }
    string vis_file = directory + "/" + printfpp("%s_RGB_GT.png",str_frame_num.c_str());
    log_file << "vis_file: " << vis_file << endl;
    pose.vis = imread(vis_file);
    
    pose.identifier = boost::filesystem::path(filename).stem().string();
    
    vector<double> xs, ys;
    while(true)
    {
      string joint_name, colon;
      double x, y, z;
      ifs >> joint_name >> colon >> x >> y >> z;
      if(!ifs.good())
	break;
      xs.push_back(x);
      ys.push_back(y);
      pose.joint_positions[joint_name] = Vec3d(x,y,z);
    }
    
    // center the camera
    if(center)
    {
      double xmean = sum(xs)/xs.size();
      double ymean = sum(ys)/ys.size();
      log_file << printfpp("xmean = %f ymean = %f",xmean,ymean) << endl;
      for(auto&&pair : pose.joint_positions)
      {
	pair.second[0] += 320/2 - xmean;
	pair.second[1] += 240/2 - ymean;
      }
    }
    return pose;
  }
  
  vector<IK_Grasp_Pose> greg_load_grasp_pose(string directory)
  {
    vector<IK_Grasp_Pose > grasp_Poses;
    vector<string> gt_files = allStems(directory,boost::regex(".*RGB.txt$"));
    log_file << printfpp("GregIKSynther: %s : %d",directory.c_str(),gt_files.size()) << endl;
    
    for(string& stem : gt_files)
    {
      log_file << "trying grasp: " << stem << endl;
      string filename = directory + "/" + stem + ".txt";

      IK_Grasp_Pose pose = greg_load_grasp_pose_one(directory,filename);
	  
      grasp_Poses.push_back(pose);
      log_file << "wrote grasp: " << pose.identifier << endl;
    }
    
    return grasp_Poses;
  }
  
  GregIKSynther::GregIKSynther(string directory)
  {
    log_file << printfpp("+GregIKSynther: %s",directory.c_str()) << endl;
    grasp_Poses = greg_load_grasp_pose(directory);
    log_file << "-GregIKSynther" << endl;
  }
  
  void greg_ik_synth(string directory)
  {
    GregIKSynther synther(directory);
    vector<string> gt_files = allStems(directory,boost::regex(".*RGB.txt$"));
    
    for(int iter = 0; iter < 5*gt_files.size(); ++iter)
    {
      // generate the example
      IKGraspSynther::SynthContext context;
      shared_ptr<LibHandMetadata> match = synther.synth_one_example(context);
      
      // write the ID to another file.
      string id = context.pose->identifier;
      ofstream ofs(match->get_filename() + ".source.txt");
      assert(ofs.is_open());
      ofs << directory << id << ".txt";
    }    
  }
  
  void greg_ik_synth()
  {
    greg_ik_synth("/home/grogez/Egocentric/MargaCorrected/");
    //greg_ik_synth("/home/grogez/Egocentric/Greg-GT/");
  }
   
  void setAnnotations(MetaData_YML_Backed * metadata,IK_Grasp_Pose&pose)
  {
    static const LibHand_DefDepth_Keypoint_Mapping name_mapping;

    Rect handBB;
    for(auto & joint : pose.joint_positions)
    {
      Point2d pt(joint.second[0],joint.second[1]);
      if(name_mapping.lh_to_dd.find(joint.first) != name_mapping.lh_to_dd.end())
      {
	metadata->keypoint(name_mapping.lh_to_dd.at(joint.first),pt,true);
	//cv::circle(depth_vis,pt,5,Scalar(0,255,0));
      }
	  
      // compute the handBB from the keypoints.
      if(handBB == Rect())
	handBB = Rect(pt,Size(1,1));
      else
	handBB |= Rect(pt,Size(1,1));
    }
    //log_im("VideoDirectory: loaded keypoints",depth_vis);
    metadata->set_HandBB(handBB);
  }
}

#endif
