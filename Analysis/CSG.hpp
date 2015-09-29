/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_CSG
#define DD_CSG

#include <opencv2/opencv.hpp>
#include "boost/multi_array.hpp"
#include <string>
#include <map>

namespace deformable_depth
{
  using std::string;
  using cv::Vec3b;
  using cv::Vec3d;
  using std::map;
  
  class CSG_Workspace
  {
  protected:
    typedef boost::multi_array<Vec3b, 3> Array3D;
    Array3D volume;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    int xres, yres, zres;
    
    void put(Vec3d raw_pos, Vec3b value);
    
  public:
    CSG_Workspace(int xres, int yres, int zres, const map<string,Vec3d>&contains);
    void writeLine(Vec3d p1, Vec3d p2,int id);
    void dilate(int id);
    void paint();
  };
}

#endif
