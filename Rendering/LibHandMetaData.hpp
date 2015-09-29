/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_LIBHAND_METADATA
#define DD_LIBHAND_METADATA

#include "MetaData.hpp"
#include <hand_renderer.h>
#include <scene_spec.h>
#include <hand_pose.h>
#include <hand_camera_spec.h>
#include "YML_Data.hpp"

#include <map>
#include <string>

namespace deformable_depth
{
  using std::string;
  using std::map;

  class LibHand_DefDepth_Keypoint_Mapping
  {
  public:
    LibHand_DefDepth_Keypoint_Mapping();
    map<string,string> lh_to_dd;
    map<string,string> dd_to_lh;
    
  protected:
    void insert(string lhName,string ddName);
  };

  string dd2libhand(string dd_name,bool can_fail = false);
  string libhand2dd(string libhand_name,bool can_fail = false);
  
  // class which represents such a syntheized example.
  // The keypoints are automatically mapped from those
  // provided by libhand to those needed by deformable_depth
  class LibHandMetadata : public MetaData_Editable
  {
  public:
    // core functions
    LibHandMetadata(string filename,bool read_only);
    virtual ~LibHandMetadata();
    virtual map<string,AnnotationBoundingBox > get_positives();
    virtual std::shared_ptr<ImRGBZ> load_im();    
    virtual std::shared_ptr<const ImRGBZ> load_im() const;
    virtual string get_pose_name();
    virtual void setPose_name(string pose_name);
    virtual string get_filename() const;
    virtual bool leftP() const;
    // keypoint functions
    virtual pair<Point3d,bool> keypoint(string name);
    virtual pair<Point3d,bool> keypoint(string name) const;
    virtual int keypoint();
    virtual bool hasKeypoint(string name);
    virtual vector<string> keypoint_names();    
    virtual DetectionSet filter(DetectionSet src);    
    
    virtual bool use_negatives() const;
    virtual bool use_positives() const;
    virtual void change_filename(string filename);
    string naked_pose() const;
    virtual Mat getSemanticSegmentation() const override;

    // protected ctor
    LibHandMetadata(string filename,const Mat&RGB, const Mat&Z,
		    const Mat& segmentation, const Mat&semantics,Rect handBB,
		    libhand::HandRenderer::JointPositionMap&joint_positions,
		    CustomCamera camera,bool read_only);
    friend class LibHandSynthesizer;
    
    Vec3d up() const;
    Vec3d normal_dir() const;
    Vec3d lr_dir() const;

  protected:
    void log_metric_bb() const;
    void load();
    string map_kp_name(string dd_name) const;
    string unmap_kp_name(string libhand_name) const;
    
    mutable mutex monitor;
    mutable mutex load_im_monitor;
    mutable shared_ptr<ImRGBZ> im_cache;
    void ensure_im_cached() const;
    
    map< string, AnnotationBoundingBox > poss;
    
    // protected data members
    bool loaded;
    typedef map<string,cv::Vec3d> JointPositionMap;
    JointPositionMap joint_positions;
    Mat RGB, Z;
    string filename;
    bool read_only;
    Rect handBB;
    CustomCamera camera;
    string pose_name;
  };  
}

#endif

