/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_HORN_ABS_ORIENTATION
#define DD_HORN_ABS_ORIENTATION

#include <opencv2/opencv.hpp>
#include <vector>
#include <Quaternion.hpp>

namespace deformable_depth
{
  using std::vector;
  using cv::Point3d;
  using cv::Vec3d;
  using cv::Mat;
  
  struct AbsoluteOrientation
  {
    // aligned = scale*R*Raw + T;
    double distance;
    Mat T;
    Mat R;
    double scale;
    Quaternion quaternion;
  };
  
  AbsoluteOrientation distanceAbsoluteOrientation(const vector<Point3d>&xs,const vector<Point3d>&ys);
  AbsoluteOrientation distHornAO(const vector<Vec3d>&xs,
		    const vector<Vec3d>&ys);
}


#endif
