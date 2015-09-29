/**
 * Copyright 2014: James Steven Supancic III
 **/

#include <Quaternion.hpp>
#include "params.hpp"
#include "util.hpp"

namespace deformable_depth
{
  using namespace cv;

  Quaternion::Quaternion(Vec3d axis,double theta)
  {
    q0 = std::cos(theta/2);
    qx = std::sin(theta/2)*std::cos(axis[0]);
    qy = std::sin(theta/2)*std::cos(axis[1]);
    qx = std::sin(theta/2)*std::cos(axis[2]);
  }
  
  Quaternion::Quaternion(double q0, double qx, double qy, double qz) : 
    q0(q0), qx(qx), qy(qy), qz(qz)
  {
  }

#ifdef DD_ENABLE_HAND_SYNTH  
  Quaternion::Quaternion(const Ogre::Quaternion& oq) : 
   q0(oq.w), qx(oq.x), qy(oq.y), qz(oq.z)
  {
  }
#endif
  
  Quaternion::Quaternion() : q0(0), qx(0), qy(0), qz(0)
  {
  }
  
  Quaternion::operator bool () const
  {
    return q0 == 0 && qx == 0 && qy == 0 && qz == 0;
  }
  
  Mat Quaternion::rotation_matrix()
  {
    normalize();
    
    Mat R(3,3,DataType<double>::type,Scalar::all(0));
    // row 0
    R.at<double>(0,0) = 1 /*q0*q0+qx*qx*/ - 2*qy*qy - 2*qz*qz;
    R.at<double>(0,1) = 2*(qx*qy - q0*qz);
    R.at<double>(0,2) = 2*(qx*qz + q0*qy);
    // row 1
    R.at<double>(1,0) = 2*(qy*qx + q0*qz);
    R.at<double>(1,1) = 1 - 2*qx*qx /*q0*q0 +qy*qy*/ - 2*qz*qz;
    R.at<double>(1,2) = 2*(qy*qz - q0*qx);
    // row 2
    R.at<double>(2,0) = 2*(qz*qx - q0*qy);
    R.at<double>(2,1) = 2*(qz*qy + q0*qx);
    R.at<double>(2,2) = 1 - 2*qx*qx - 2*qy*qy /*q0*q0+qz*qz*/;
    
    return R;
  }
  
  void Quaternion::normalize()
  {
    double q_norm = std::sqrt(q0*q0 + qx*qx + qy*qy + qz*qz);
    q0 /= q_norm; qx /= q_norm; qy /= q_norm; qz /= q_norm; 
  }
  
  Vec4d Quaternion::get() const
  {
    return Vec4d(q0,qx,qy,qz);
  }
  
  static double standardize_radians(double radians)
  {
    while(radians < 0)
      radians += 2*params::PI;
    
    while(radians > 2*params::PI)
      radians -= 2*params::PI;
    
    return radians;
  }
  
  double Quaternion::roll() const
  {
    return standardize_radians(
      std::atan2(2*(q0*qx + qy*qz),1-2*(qx*qx + qy*qy)));
  }  
  
  double Quaternion::pitch() const
  {
    return standardize_radians(
      std::asin(2*(q0*qy - qz*qx)));
  }

  double Quaternion::yaw() const
  {
    return standardize_radians(
      std::atan2(2*(q0*qz+qx*qy),1-2*(qy*qy - qz*qz)));
  }
  
  string Quaternion::toString()
  {
    Vec4d v = get();
    return printfpp("quaternion(%f %f %f %f)",v[0],v[1],v[2],v[3]);
  }
  
  std::ostream& operator<<(std::ostream& oss, Quaternion& q)
  {
    Vec4d v = q.get();
    oss << printfpp("quaternion(%f %f %f %f)",v[0],v[1],v[2],v[3]);
    return oss;
  }
}


