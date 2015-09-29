/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_QUATERNION
#define DD_QUATERNION

#include <opencv2/opencv.hpp>
#include <ostream>
#include <iostream>
#include <string>
#ifdef DD_ENABLE_HAND_SYNTH
#include <OgreQuaternion.h>
#endif

namespace deformable_depth
{
  using cv::Mat;
  
  class Quaternion
  {
  protected:
    double q0, qx, qy, qz;
    
  public:
#ifdef DD_ENABLE_HAND_SYNTH
    Quaternion(const Ogre::Quaternion&oq);
#endif
    // radians
    Quaternion(cv::Vec3d axis,double theta);
    Quaternion(double q0, double qx, double qy, double qz);
    Quaternion();
    Mat rotation_matrix();
    void normalize();    
    cv::Vec4d get() const;
    double roll() const;
    double pitch() const;
    double yaw() const;
    operator bool() const;
    std::string toString();
  };
  
  std::ostream& operator<< (std::ostream&oss,Quaternion&q);
}

#endif
