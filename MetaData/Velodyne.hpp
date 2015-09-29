/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_VELODYNE
#define DD_VELODYNE

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

namespace deformable_depth
{
  class VelodyneData
  {
  protected:
    std::vector<cv::Vec3d> points;
    std::vector<double> reflectances;
    
  public:
    VelodyneData(std::string filename);
    std::vector<cv::Vec3d>&getPoints();
    std::vector<double>&getReflectances();    
  };
}

#endif
