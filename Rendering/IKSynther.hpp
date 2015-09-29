/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifdef DD_ENABLE_HAND_SYNTH 
#ifndef DD_IK_SNYTHER
#define DD_IK_SNYTHER

#include "LibHandSynth.hpp"
#include <map>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Quaternion.hpp"
#include "HornAbsOri.hpp"
#include "InverseKinematics.hpp"

namespace deformable_depth
{
  ///
  ///
  ///
  using std::string;
  using std::map;
  using std::vector;
  using cv::Vec3d;
  
  struct IK_Grasp_Pose
  {
  public:
    // data
    map<string,Vec3d> joint_positions;
    string identifier;
    Mat vis;
    
    // methods
    double depth() const;
    map<string,Vec3d> joint_positions_flip_lr() const;
    double mean_dist() const;
  };
  
  class IKGraspSynther
  {
  protected:
    LibHandSynthesizer synther;
    vector<IK_Grasp_Pose > grasp_Poses;
    
    
    void do_IK_LR_Symmetric(
	  IK_Grasp_Pose&pose,
	  AbsoluteOrientation&abs_ori,
	  PoseRegressionPoint&angles,
	  bool&fliped,
	  string&lr_mesg);

  public:
    struct SynthContext
    {
      IK_Grasp_Pose*pose;
    };
    
    IKGraspSynther();
    shared_ptr<LibHandMetadata> synth_one_example(SynthContext&context,int index = -1);
  };  
  
  ///
  /// SECTION: IKSynth w/ Greg's data
  ///
  void greg_ik_synth();
  vector<IK_Grasp_Pose> greg_load_grasp_pose(string directory);
  IK_Grasp_Pose greg_load_grasp_pose_one(string directory,string filename, bool center = true);
  void setAnnotations(MetaData_YML_Backed * metadata,IK_Grasp_Pose&pose);
}

#endif
#endif
