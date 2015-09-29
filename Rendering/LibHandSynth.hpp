/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_LIBHANDSYNTH
#define DD_LIBHANDSYNTH

#include "MetaData.hpp"

#include <hand_renderer.h>
#include <scene_spec.h>
#include <hand_pose.h>
#include <hand_camera_spec.h>

#include "LibHandMetaData.hpp"
#include "LibHandRenderer.hpp"

namespace deformable_depth
{  
  struct PoseRegressionPoint;

  class LibHandSynthesizer;
  
  // class which synthesizes hand training examples using libhand
  class LibHandSynthesizer
  {
  public:
    enum ComponentFlags
    {
      COMP_FLAG_CAM = 1,
      COMP_FLAG_HAND_POSE = 2,
      COMP_FLAG_WORLD_SIZE = 4,
      COMP_FLAG_BG = 8
    };
    
    enum ExemplarFilters
    {
      NONE = 0,
      FILTER_DOWNWARD = 1,
      FILTER_EGO_CENTRIC = 2,
      FILTER_NOT_EGO_CENTRIC = 4,
      FILTER_BACKWARD = 8,
      FILTER_BAD_REGISTRATION = 16,
      FILTER_BAD_BB = 32,
      FILTER_DEFAULT = 
	FILTER_BAD_BB |FILTER_BAD_REGISTRATION|FILTER_EGO_CENTRIC
    };
    
    static constexpr int IDX_MAX_VALID_FINGER_JOINTS = 14;
    
    LibHandSynthesizer(string save_dir);
    LibHandSynthesizer();
    // functions to render 
    Mat render_only();
    void read_joint_positions();
    shared_ptr< LibHandMetadata > synth_one(bool read_only = false);
    shared_ptr< LibHandMetadata > synth_random(bool read_only = false);
    
    // access rendered data
    Mat& getRGB();
    Mat& getDepth();
    Mat& getRenderedRGB();
    Mat& getRenderedDepth();      
    Mat& getSegmentation();
    libhand::HandRenderer::JointPositionMap& getJointPositionMap();
    
    // functions to configure the pose generated
    void set_model(PoseRegressionPoint&pose);
    void set_filter(int filter);
    void randomize_model(Size bgSize);
    void randomize_camera();
    void randomize_background(Size bgSize);
    void set_model(PoseRegressionPoint&pose,Size bgSize, 
		   int component_flags = 0b11111111);
    void set_flip_lr(bool flip_lr);
    void set_render_armless(bool render_armless);
    void set_synth_bg(bool synth_bg);
    void set_hand_pose(libhand::FullHandPose&hand_pose);
    libhand::FullHandPose get_hand_pose();
    const libhand::HandCameraSpec get_cam_spec();
    void set_cam_spec(libhand::HandCameraSpec&);
    void perturb_model();
    // access the configured pose
    const libhand::HandRenderer&get_hand_renderer();
    const libhand::HandRenderer::JointPositionMap & get_jointPositionMap();
    libhand::SceneSpec & get_scene_spec();
        
  protected:
    libhand::HandCameraSpec random_cam_spec();
    libhand::HandCameraSpec perturbation(const libhand::HandCameraSpec&in,double cam_var = .15);

    map<string,LibHandRenderer* > models;
    shared_ptr< LibHandMetadata > try_synth_one(bool read_only = false);
    void init();
    bool filter_post_gen();
    bool filter_self_intersecting();
    bool filter_seg_area();
    string alloc_filename(string&filename, int&file_id);
    void synth_one_debug_preregistration();
    void synth_one_debug(
      Mat&RGB, Mat&Z, shared_ptr<ImRGBZ>&bg_im, Rect_<double> handBB); 
    
    ExemplarFilters filter;
    bool flip_lr;
    string save_dir;
    vector<shared_ptr<MetaData>> bgs;
    Mat Z, RGB, self_intersections, fingers;
    // details needed for the regression
    float bg_tx, bg_ty, bg_theta_rad;
    bool render_armless;
    bool synthBG;
    
    friend void gen_reg();
  };
      
  
  class LibHandJointRanges
  {
  protected:
    vector<double> bend_mins, bend_maxes, side_mins, side_maxes, twist_mins, twist_maxes, elongation_mins, elongation_maxes;
    
  public: 
    static constexpr int NUM_JOINTS = 18;
    LibHandJointRanges();
    double bend_min(int joint) {return bend_mins[joint];}
    double bend_max(int joint) {return bend_maxes[joint];}
    double side_min(int joint) {return side_mins[joint];}
    double side_max(int joint) {return side_maxes[joint];}
    double twist_min(int joint){return twist_mins[joint];}
    double twist_max(int joint){return twist_maxes[joint];}
    double elongation_min(int joint){return elongation_mins[joint];}
    double elongation_max(int joint){return elongation_maxes[joint];}
  };  
  
  void gen_training_data(int argc, char**argv);
  // visualize a set of synthetic examples
  void visualize_synthetic();
  libhand::FullHandPose random_hand_pose(int num_bones);
  libhand::HandCameraSpec random_cam_spec();
  libhand::FullHandPose perturbation(const libhand::FullHandPose&in,double var = .075);
  Rect compute_hand_bb(LibHandRenderer&renderer,Mat&Z,LibHandSynthesizer::ExemplarFilters filter);
  bool is_egocentric(const map<string,Vec3d>&hand_pose);
}

#endif
