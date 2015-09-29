/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_ICL_METADATA
#define DD_ICL_METADATA

#include "MetaData.hpp"
#include "Video.hpp"

namespace deformable_depth
{
  // adapts the Imperial College London Hand Dataset for my system.
  class ICL_Video : public Video
  {
  public:
    // 1.ICL or 2.ICL
    ICL_Video(string filename);
    virtual ~ICL_Video();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    virtual int getNumberOfFrames();    
    virtual string get_name();
    virtual bool is_frame_annotated(int index);

  protected:
    string filename;
    map<int,string > frame_labels;
  };

  MetaData_YML_Backed* load_ICL(string base_path,string annotation,bool training);  
  string icl_base();
  MetaData* load_ICL_training(string descriptor);
  vector<shared_ptr<MetaData> > ICL_training_set();
  void show_ICL_training_set();  
}

#endif
