/**
 * Copyright 2014: James Steven Supancic III
 **/
  
#ifndef DD_LIBHAND_RENDERER
#define DD_LIBHAND_RENDERER

class LibHandRenderer;

#include <opencv2/opencv.hpp>

#include <hand_renderer.h>
#include <scene_spec.h>
#include <hand_pose.h>
#include <hand_camera_spec.h>
#include <string>
#include "Poselet.hpp"

namespace deformable_depth
{
  struct PoseRegressionPoint;

  using cv::Mat;
  using cv::Vec3b;
  using std::string; 

  class HandRegion
  {
  protected:
    Vec3b color;
    
    static Vec3b part_color_greg(const string&part_name);      
    static Vec3b part_color_libhand(const string&part_name);
    
  public:
    static Vec3b quantize(Vec3b);
    static Vec3b wrist_color();
    static Vec3b background_color();
    static bool is_background(Vec3b color) ;
    static Vec3b phalan_color(int digit, int joint);
    // esential "backbone" mapping functions
    static Vec3b part_color(const string&part_name);
    static string part_name(Vec3b part_color);

    HandRegion(Vec3b color);
    bool is_finger() const;
    bool is_arm() const;
  };
  Mat drawRegions(MetaData&md);

  class LibHandRenderer
  {
  protected:
    libhand::HandRenderer hand_renderer;
    libhand::HandRenderer::JointPositionMap jointPositionMap;
    libhand::HandCameraSpec cam_spec;
    libhand::FullHandPose hand_pose;
    libhand::SceneSpec scene_spec;
    bool flip_lr;
    double world_size_randomization_sf;
    CustomCamera camera;

    Mat RGB, Z, segmentation;
    
    // rendering functions/interact with OpenGL
    bool synth_one_register();
    void read_rgb_buffer();
    void read_depth_buffer(Mat&dest);    
    void register_joint_positions();
    float register_one_point(
      float rnd_z, bool invert);
    void merge_one_pix(
	float&cm_z,float&out_z,
	Vec3b&rnd_rgb,Vec3b&out_rgb,
	double&min_z, double&max_z);        
    void commit_model();  
        
  public:
    // coordinate conversion
    double fromDepthCM(double cm_depth) const;
    double toDepthCM(double libhand_depth) const;    

    LibHandRenderer(string spec_path,cv::Size sz);
    Mat& getRGB();
    Mat& getDepth();
    Mat& getSegmentation();
    CustomCamera& getCamera() ;
    libhand::HandRenderer::JointPositionMap& getJointPositionMap();
    
    // configuration functions
    libhand::HandRenderer&get_hand_renderer();
    libhand::HandRenderer::JointPositionMap & get_jointPositionMap();
    libhand::HandCameraSpec &get_cam_spec();
    libhand::FullHandPose &get_hand_pose();
    libhand::SceneSpec & get_scene_spec();
    void set_flip_lr(bool flip_lr);
    bool get_flip_lr();
    void set_flip_world_size_ratio(double world_size_ratio);
    
    // rendering functions
    Mat render_only();
    void read_joint_positions();
    bool render();
    
    friend void gen_reg();
  };

  // hand renderer singletons
  namespace renderers
  {
    LibHandRenderer* intersection();
    LibHandRenderer* no_arm();
    LibHandRenderer* with_arm();
    LibHandRenderer* segmentation();
  }

}

#endif
