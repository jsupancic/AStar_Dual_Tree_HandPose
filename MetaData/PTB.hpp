/**
 * Copyright 2014: James Steven Supancic III
 *
 * Metadata for the Princeton Tracking Becnhmark
 **/

#ifndef DD_PTB_DATA
#define DD_PTB_DATA

#include "Video.hpp"

namespace deformable_depth
{
  // eg. box_no_cc.PTB
  class PTB_Video : public Video
  {
  public:
    PTB_Video(string name);
    virtual ~PTB_Video();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    virtual int getNumberOfFrames();    
    virtual string get_name();
    virtual bool is_frame_annotated(int index);

  protected:
    string filename;
    string path;
    map<int,string> rgb_files;
    map<int,string> depth_files;
    float fovx, fovy, fx, fy, cx, cy;
    Rect init;
  };
}

#endif
