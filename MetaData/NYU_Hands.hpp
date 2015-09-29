/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_NYU_HANDS
#define DD_NYU_HANDS

#include "MetaData.hpp"
#include "Video.hpp"

namespace deformable_depth
{
  typedef function<Point2d (int index)> KeypointFn;
  shared_ptr<MetaData_YML_Backed> NYU_Video_make_datum(KeypointFn keypointFn,string metadata_filename,const ImRGBZ&im);
  shared_ptr<ImRGBZ> NYU_Make_Image(string direc, int index,string metadata_filename);
  MetaData* NYU_training_datum(int recordIter);

  // the NYU Video is a single contiguous sequence.
  // construct as 0.NYU_HANDS
  class NYU_Video : public Video
  {
  public:
    NYU_Video();
    virtual ~NYU_Video();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    virtual int getNumberOfFrames();    
    virtual string get_name();
    virtual bool is_frame_annotated(int index);    

  protected:
    vector<vector<Vec3d> > keypoints;
    int frame_count;
  };

  string nyu_prefix();
  string nyu_base(); 
  vector<shared_ptr<MetaData> > NYU_training_set();
}

#endif
