/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_ASTAR_HANDS
#define DD_ASTAR_HANDS

#include "MetaData.hpp"
#include "Video.hpp"

namespace deformable_depth
{
  // the NYU Video is a single contiguous sequence.
  // construct as 0.ASTAR_HANDS
  class ASTAR_Video : public Video
  {
  public:
    // TITLE: 0.ASTAR_HANDS
    ASTAR_Video(int id = 0);
    virtual ~ASTAR_Video();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    virtual int getNumberOfFrames();    
    virtual string get_name();
    virtual bool is_frame_annotated(int index);    

  protected:
    vector<string> examples;
  };
}

#endif
