/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifdef DD_ENABLE_OPENNI
#ifndef DD_ONI_VIDEO
#define DD_ONI_VIDEO

#include <OpenNI.h>
#include <string>
#include "MetaData.hpp"
#include "Video.hpp"

namespace deformable_depth
{
  using std::string;
  
  class ONI_Video : public Video
  {
  public:
    virtual ~ONI_Video();
    ONI_Video(string filename);
    shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    int getNumberOfFrames();
    virtual string get_name();
    
  protected:
    shared_ptr<MetaData_YML_Backed> getFrame_simple(int index,bool read_only,string lr_id);
    void cvt_oni_to_cv(
      openni::VideoFrameRef&colorFrame, 
      openni::VideoFrameRef&depthFrame,
      cv::Mat&cv_depth,
      cv::Mat&cv_rgb,CustomCamera& camera);    
    
    mutex monitor;
    string filename;
    openni::Device device;
    openni::PlaybackControl*control;
    openni::VideoStream color_video, depth_video;    
  };
}

#endif
#endif


