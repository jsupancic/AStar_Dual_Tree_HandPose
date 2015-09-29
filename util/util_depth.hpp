/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_UTIL_DEPTH
#define DD_UTIL_DEPTH

#define use_speed_ 0
#include <opencv2/opencv.hpp>

#include <iostream>
#include <set>
#include "util_real.hpp"

namespace deformable_depth
{
  using namespace cv;
  
#if CV_MAJOR_VERSION >= 3 || (CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4)   
  Mat_<float> fillDepthHoles(const Mat_<float> Zin,double max_fill_dist = inf);
  Mat_<float> fillDepthHoles(const Mat_<float> Zin, Mat_<uchar> invalid,double max_fill_dist = inf);
  Mat_<float> fillDepthHoles(const Mat_<float> Zin, Mat_<uchar> invalid, Mat_<uchar> conf);  
  
  template<typename T>
  Mat fillHoles(const Mat Zin, Mat_<uchar> invalid,double max_fill_dist = inf);
#ifdef DD_CXX1
  extern template
  Mat fillHoles<float>(const Mat Zin, Mat_<uchar> invalid,double max_fill_dist = inf);
  extern template
  Mat fillHoles<Vec3b>(const Mat Zin, Mat_<uchar> invalid,double max_fill_dist = inf);
#endif
#endif

  Mat depth_float2uint16(const Mat&inDepth);

  // area of the plane depth cm from the camera in cm^2
  class Camera
  {
  public:
    // for rectlinear parameterization
    float worldAreaForImageArea(float depth, Rect_<float> bb)const;
    float areaAtDepth(float depth) const;
    float pixAreaAtDepth(float depth)const;
    // return metric (cm) sizes?
    float widthAtDepth(float depth)const;
    float heightAtDepth(float depth)const;
    Size sizeAtDepth(float depth) const;
    float depthForBB_Ratio(Rect_<float> im_bb, Rect_<float> world_bb);
    double DistPixtoWorldCM(double depth,double pix_dist) const;
    // how big is the 
    Rect_<double> bbForDepth(float z, Size im_size, 
			     int rIter, int cIter, float width, float height,
			     bool clamp = true
			    ) const;
    Rect_<double> bbForDepth(float z, int y_img_pos, int x_img_pos, float width, float height) const;
    Rect_<double> bbForDepth(Mat Z, int rIter, int cIter, float width, float height) const;
    // for orthograhpic parammeterization
    bool is_orhographic() const;
    // new stuff, based on spherical understanding.
    double depth_Cartesian_to_Spherical(double z, Point2d uv) const;   
    double depth_Spherical_to_Cartesian(double z, Point2d uv) const;
    double distance_angular(Point2d image_xy1, Point2d image_xy2) const;
    double distance_geodesic(Point2d image_xy1, double z1, Point2d image_xy2, double z2) const;
    Size imageSizeForMetricSize(float depth,Size metricSize) const;
    
  public:
    virtual float hFov() const = 0;
    virtual float vFov() const = 0;
    virtual float hRes() const = 0;
    virtual float vRes() const = 0;
    virtual float focalX() const;
    virtual float focalY() const;
    virtual float metric_correction() const;
  protected:
  };
  
  class DepthCamera : public Camera
  {
  public:
    virtual float hFov() const;
    virtual float vFov() const;
    virtual float hRes() const;
    virtual float vRes() const;
  };
  
  class RGBCamera : public Camera
  {
  public:
    virtual float hFov() const;
    virtual float vFov() const;    
    virtual float hRes() const;
    virtual float vRes() const;
  };
  
  class CustomCamera : public Camera
  {
  public:
    CustomCamera(float hFov, float vFov, float hRes, float vRes, float metric_correction = 1, float fx = qnan, float fy = qnan);
    CustomCamera();
    void setFov(float hFov, float vFov);
    void scaleFov(double hScale, double vScale);
    void setRes(float hRes, float vRes);
    CustomCamera crop(Rect_<double> roi) const;
    void setMetricCorrection(float factor);
  private:
    float m_hFov, m_vFov, m_hRes, m_vRes, m_fx, m_fy;
    // this shouldn't be needed unless there is ambiguity
    // in extrinsic camera parameters to be fixed.
    float m_metric_correction; 
  public:
    virtual float hFov() const;
    virtual float vFov() const;    
    virtual float hRes() const;
    virtual float vRes() const;
    virtual float focalX() const;
    virtual float focalY() const;
    
    friend void read(const cv::FileNode&, deformable_depth::CustomCamera&, deformable_depth::CustomCamera);
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::CustomCamera&);
    virtual float metric_correction() const;
    
  protected:
  };
  std::string to_string( const Camera & camera );
  void read(const cv::FileNode&, deformable_depth::CustomCamera&, deformable_depth::CustomCamera);
  void write(cv::FileStorage&, std::string&, const deformable_depth::CustomCamera&);
  
  float medianApx(const Mat&Z, Rect bb, float order = 0.5);
  // spherical to Cartesian depth conversions
  Vec3d vecOf(double azimuth, double altitude);
  double radius(const Vec3d&vec);
  double azimuth(const Vec3d&vec);
  double altitude(const Vec3d&vec);
}

#endif


