/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_INVERSE_KINEMATICS
#define DD_INVERSE_KINEMATICS

#include <util.hpp>
#include <hand_renderer.h>
#include <scene_spec.h>
#include <hand_pose.h>
#include <hand_camera_spec.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <map>
#include "MetaData.hpp"
#include <string>
#include "HornAbsOri.hpp"
#include "LibHandSynth.hpp"

#ifndef DD_ENABLE_HAND_SYNTH
namespace libhand
{
  // without libhand we need to provide implementations of these
  void read(const cv::FileNode&fn, libhand::FullHandPose&hand_pose, libhand::FullHandPose);
  void write(cv::FileStorage&fs, string&, const libhand::FullHandPose&hand_pose);
}
#endif

namespace deformable_depth
{
  using std::vector;
  using cv::Point2d;
  using std::map;
  using std::string;
  
  ///
  /// SECTION 2D IK/Regression
  /// 
  class IKBinaryDatabase
  {
  protected:
    LibHandRenderer& renderer;
    ifstream ifs; 
    map<string,Vec3d> init_jpm;

  public:
    struct XYZ_TYR_Pair
    {      
      bool valid;
      // XYZ
      map<string,Vec3d> joint_position_map;
      // TYR
      libhand::HandCameraSpec cam_spec;
      libhand::FullHandPose hand_pose;
    };

    IKBinaryDatabase(LibHandRenderer& renderer,string filename);   
    XYZ_TYR_Pair next_record();
    bool hasNext();
  };

  struct PoseRegressionPoint
  {
    // transient
    libhand::SceneSpec*scene_spec;
    // write
    libhand::HandRenderer::JointPositionMap jointPositionMap;
    vector<Point2d> keypoints;
    map<string,AnnotationBoundingBox> parts; 
    libhand::HandCameraSpec cam_spec;
    libhand::FullHandPose hand_pose;
    double world_size_randomization_sf;
  };      
  void write(cv::FileStorage&fs, std::string&, const libhand::HandCameraSpec&cam_spec);
  void read(const FileNode&node, libhand::HandCameraSpec&cam_spec, libhand::HandCameraSpec);
  void write(FileStorage& fs, string&, const PoseRegressionPoint& pt);
  void read(const FileNode& node, PoseRegressionPoint& pt, PoseRegressionPoint);  
    
  // serialization for libhand
  void write(cv::FileStorage&fs, std::string&, const libhand::HandCameraSpec&cam_spec);
  
  // perform a regression
  PoseRegressionPoint libhand_regress(
    const vector<Point2d>&vec_target,
    function<vector<Point2d> (PoseRegressionPoint&rp)> extract_saved_keypoints);
  Poselet::Procrustean2D regress_binary(LibHandRenderer&renderer,
					 const map<string,Vec3d>&joint_positions);  
  PoseRegressionPoint ik_regress(BaselineDetection&det);
  vector<PoseRegressionPoint> ik_regress_thresh(BaselineDetection&det,double max_cm_thresh);
  
  // generate regression map from 2D keypoints
  // to model parameters.
  void gen_reg();
  // regress libhand model parameters from essentail keypoints
  void test_regression();    
  
  ///
  /// SECTION: Incremental 2D IK
  ///
  Poselet::Procrustean2D incremental2DIk(LibHandRenderer&renderer,
					 const map<string,Vec3d>&joint_positions);
  
  ///
  /// SECTION 3D IK/Regression
  ///   
  PoseRegressionPoint IK_Match(const map<string,Vec3d>&joint_positions,
			       AbsoluteOrientation&abs_ori,LibHandSynthesizer&refiner);
}

#endif


