/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_ANNOTATION
#define DD_ANNOTATION

#include "util.hpp"
#include "MetaData.hpp"

namespace libhand
{
  class HandCameraSpec;
  class FullHandPose;
}

namespace deformable_depth
{
  // create annotations
  void label(int argc, char**argv);
  vector<string> essential_keypoints();

  // Accessing the annotation
  void getFullHandPose(MetaData&metadata,
		       map<string,Vec3d>&set_keypoints,map<string,Vec3d>&all_keypoints,
		       bool&fliplr,
		       libhand::HandCameraSpec&cam_spec,libhand::FullHandPose&hand_pose);
  bool hasFullHandPose(MetaData&metadata);
  void putFullHandPose(MetaData&metadata,
		       map<string,Vec3d>&set_keypoints,map<string,Vec3d>&all_keypoints,bool&fliplr,
		       libhand::HandCameraSpec&cam_spec,libhand::FullHandPose&hand_pose);

  // visualization
  Mat showAnnotations(cv::Mat im, deformable_depth::MetaData& metadata, double sf, cv::Rect cropBB);

  // tools
  struct PointAnnotation
  {
    Point2d click;
    bool visibility;
    Mat RGB;
    // values
    // 'r', 'x', 'p', 'n', '\0'
    char code;  
  };
  map<string,PointAnnotation> labelJoints(const ImRGBZ&im, Rect handBB,bool skip_labeled = false);
  void write(cv::FileStorage&, std::string&, const deformable_depth::PointAnnotation&);
  void read(const cv::FileNode&, deformable_depth::PointAnnotation&, deformable_depth::PointAnnotation);
}

#endif
