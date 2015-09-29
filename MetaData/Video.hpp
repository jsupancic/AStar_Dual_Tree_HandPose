/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_VIDEO
#define DD_VIDEO

#include <memory>
#include <map>
#include <string>

#include "MetaData.hpp"
#include "LibHandMetaData.hpp"
#include "Aggregate.hpp"
#include "util_mat.hpp"

namespace deformable_depth
{
  using std::shared_ptr;
  using std::string;
  using std::map;
  
  class Video
  {
  public:
    virtual ~Video();    
    virtual int getNumberOfFrames() = 0;    
    virtual string get_name() = 0;
    virtual bool is_frame_annotated(int index);
    // get labelable metdata (for annotation and testing)
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only) = 0;
    // get un-editable/unlabelable metadata (for evaluation)
    virtual shared_ptr<MetaData> getFrame_raw(int index,bool read_only);
    virtual int annotation_stride() const;
  };
  
  //
  // Video encoded by deformable_depth with OpenCV 2.x
  //
  class Video_CV_DD : public Video
  {
  public:
    Video_CV_DD(string filename);
    virtual ~Video_CV_DD();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    void writeFrame(shared_ptr<ImRGBZ>&frame,const vector< shared_ptr< BaselineDetection> >&poss,bool leftP,int index = -1);
    void setAnnotationStride(int stride);
    virtual int getNumberOfFrames();
    virtual string get_name();
    virtual int annotation_stride() const override;
    
  protected:
    mutable recursive_mutex monitor;
    shared_ptr< MetaData_YML_Backed > HandDatumAssemblage
        (shared_ptr<BaselineDetection> annotation,shared_ptr<ImRGBZ> im);
    void ensure_store_ready(); // only for store_write
    void ensure_read_ready() const;

    mutable unique_ptr<FileStorage> store_write;
    mutable unique_ptr<FileStorage> store_read;
    string filename;
    int length;
  };
  
  //
  // Video encoded as a directory of image (.exr) files.
  //
  struct IK_Grasp_Pose;
  class VideoDirectory : public Video
  {
  public:
    VideoDirectory(string filename);
    virtual ~VideoDirectory();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);
    virtual int getNumberOfFrames();    
    virtual string get_name();
    virtual bool is_frame_annotated(int index);
    
  protected:
    // methods
    void loadAnnotations(shared_ptr<MetaDataAggregate>&metadata,int index,const Mat&D);

    // data members
    string title;
    string filename;
    LibHand_DefDepth_Keypoint_Mapping name_mapping;
    int length = -1;
  };
   
  // for Gregs videos
  struct VideoDirectoryDatum
  {
  public:
    Mat D, RGB;
    Mat UV;

  public:
    VideoDirectoryDatum(Mat D, Mat RGB, Mat UV) : D(D), RGB(RGB), UV(UV) {};
  };
  shared_ptr<VideoDirectoryDatum> load_video_directory_datum(string directory, int index);

  //
  // Video encoded as a directory of metadata files
  // supports two formats
  //    - Greg's Egocentric images.
  //    - My collection of indvidiually numbered MetaDatum files. 
  // 
  class VideoMetaDataDirectory : public Video
  {
  public:
    VideoMetaDataDirectory(string filename);
    virtual ~VideoMetaDataDirectory();
    virtual shared_ptr<MetaData_YML_Backed> getFrame(int index,bool read_only);    
    virtual shared_ptr<MetaData> getFrame_raw(int index,bool read_only) override;
    virtual int getNumberOfFrames();    
    virtual string get_name();
    virtual bool is_frame_annotated(int index);
    virtual int annotation_stride() const override;
    
  protected:
    vector<shared_ptr<MetaData> > frames;
    string title;
    string filename;
  };

  void convert_all_videos();
  void convert_video(int argc, char**argv);
  void show_frame();
  void show_video();
  // try to automatically determine the type of video and then load 
  // the correct subclass.
  shared_ptr<Video> load_video(string filename);

  string vid_public_name(string private_name);
  string vid_private_name(string public_name);
  int vid_type_id(string vidName);
}

#endif
