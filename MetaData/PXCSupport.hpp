/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_PCXSupport
#define DD_PCXSupport
#define use_speed_ 0

#include <string>
#include <opencv2/opencv.hpp>
#include "util_mat.hpp"
#include <memory>
#include "BaselineDetection.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  struct PCX_Props
  {
    float lowConf, saturated;
  };

  typedef BaselineDetection PXCDetection;
    
  Rect loadBB_PCX(const string&filename,string bb_name = "HandBB");
  Rect loadRandomExamplePCX(string&directory,Mat&RGB,Mat&Z);
  void loadPCX_RGB(const string&filename,Mat&RGB);
  void loadPCX_Z(const string&filename,Mat&Z);
  PCX_Props loadPCX_RawZ(const string&filename, Mat&ZMap, Mat&ZConf, Mat&ZUV);
  bool PXCFile_PXCFired(const string&filename);
  void loadRGBOnZ(std::string filename, cv::Mat& RGB, cv::Mat& ZMap);
  Rect rectZtoRGB(Rect zpos, Mat&UV, Size&RGB);
  Rect rectRGBtoZ(Rect rgbPos, Mat&UV);
  
  // registration functions using the UV map
  Mat registerRGB2Depth(const Mat&raw_RGB,const Mat&ZMap,const Mat&UV);  
  Mat registerDepth2RGB(const Mat&orig_Depth,const Mat&UV);
  struct PXCRegistration
  {
    Mat registration;
    Point tl_valid;
    Point br_valid;
  };
  PXCRegistration registerRGB2Depth_adv(const Mat&raw_RGB,const Mat&ZMap,const Mat&UV);
}

#ifndef WIN32
#include "Detector.hpp"

namespace deformable_depth
{
  DetectorResult convert(const PXCDetection& det);
}
#endif

#endif
