/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_YML_DATA
#define DD_YML_DATA

#include "util_mat.hpp"
#include "MetaData.hpp"

namespace deformable_depth
{
  class MetaData_Segmentable
  {
  public:
    virtual void setSegmentation(Mat&segmentation);
  };

  class MetaData_YML_Backed : public MetaData_Editable, public MetaData_Segmentable
  {
  public:
    // read
    MetaData_YML_Backed(string filename, bool read_only = false);
    // write
    virtual ~MetaData_YML_Backed();
    string naked_pose() const;
  public: // data access
    // pose
    void setPose_name(string newName);
    string get_pose_name();
    // left/right hand
    void set_is_left_hand(bool is_left_hand); 
    // bounding boxes and positives
    virtual void set_HandBB(cv::Rect newHandBB) = 0;
    string get_filename() const;
    bool operator< (const MetaData& other) const;
    // use general camera knowlege to filter invalid detections.
    virtual DetectionSet filter(DetectionSet src);
    // keypoint functions
    int keypoint();
    bool hasKeypoint(string name);
    pair<Point3d,bool> keypoint(string name);
    virtual void keypoint(string, Point3d value, bool vis);
    virtual void keypoint(string, Point2d value, bool vis);
    vector<string> keypoint_names();
    bool leftP() const;
    virtual void change_filename(string filename);
    virtual map<string,MetaData_YML_Backed* > get_subdata_yml(); 
  protected:
    Maybe<bool> is_left_hand;
    bool read_only;
    string filename;
    Rect HandBB_RGB;
    Rect HandBB_ZCoords;
    string PXCFired;
    string pose_name;
    map<string,Point2d> keypoints;
    map<string,bool> kp_vis;
  };
  
  ///
  /// Represents metadata which is returned as provided, no
  ///   transformations are applied. It is saved to disk, loaded
  ///   and returned without any changes being made. This is useful
  ///   when provideing test data from other formats.
  class Metadata_Simple : public MetaData_YML_Backed
  {
  public:
    // implementing virtual methods
    Metadata_Simple(string filename, bool read_only = false, bool b_use_positives = false, bool b_use_negatives = false);
    virtual ~Metadata_Simple();
    virtual map<string,AnnotationBoundingBox> get_positives();
    virtual void set_HandBB(cv::Rect newHandBB);
    virtual std::shared_ptr<ImRGBZ> load_im();    
    virtual std::shared_ptr<const ImRGBZ> load_im() const;
    virtual bool use_negatives() const override;
    virtual bool use_positives() const override;
    virtual Mat load_raw_RGB() override;
    virtual Mat getSemanticSegmentation() const override;

    // special functions
    void setIm(const ImRGBZ&im);
    void setRawIm(ImRGBZ&im);
    void setPositives(const map<string,AnnotationBoundingBox>&annotation);
    virtual void setSegmentation(Mat&segmentation) override;

  protected:
    Mat segmentation;
    // cache for the image in RAM
    bool b_use_positives, b_use_negatives;
    void ensure_im_cached() const;
    mutable mutex cache_mutex;
    mutable shared_ptr<ImRGBZ> im_cache;
    mutable shared_ptr<ImRGBZ> raw_im_cache;
    map<string,AnnotationBoundingBox> explicit_abbs;
  };
  
  ///
  /// Represents PXC metadata with the RGB registered to the depth
  ///
  class MetaData_DepthCentric : public MetaData_YML_Backed
  {
  public:
    virtual ~MetaData_DepthCentric(){};
    MetaData_DepthCentric(string filename, bool read_only = false);
    virtual map<string,AnnotationBoundingBox> get_positives();
    virtual void set_HandBB(cv::Rect newHandBB);
    virtual std::shared_ptr<ImRGBZ> load_im();    
    virtual std::shared_ptr<const ImRGBZ> load_im() const;
    virtual Mat load_raw_RGB() override;
    virtual Mat load_raw_depth() override;
    virtual DetectionSet filter(DetectionSet src);
    // these must be overridden due to the cropping.
    virtual pair<Point3d,bool> keypoint(string name);
    virtual void keypoint(string, Point2d value, bool vis);
    
  protected:
    // cache to avoid reloading images
    mutable mutex monitor;
    mutable shared_ptr<ImRGBZ> im_raw;
    mutable shared_ptr<ImRGBZ> im;
    mutable bool cache_ready;
    
    //Rect imBB(Point(0,0),im.RGB.size());
    // the RGB Resolution is not not the RGB valid area
    // because the depth camera has a different aspect ratio.     
    Rect RGBandDepthROI;
    void ensure_im_cached() const;
    
    // functions for caching positives and avoiding extra work
    map<string,AnnotationBoundingBox> label_cache;
    mutex label_cache_mutex;
    
    // needs to access the ROI for correct loading of external resutls.
    friend PerformanceRecord analyze_pxc(string person);
  };
  
  ///
  /// Represents PXC Metadata with the Depth registered to the RGB
  ///
  class MetaData_RGBCentric : public MetaData_YML_Backed
  {
  public:
    virtual ~MetaData_RGBCentric() {};
    MetaData_RGBCentric(string filename, bool read_only = false);
    virtual map<string,AnnotationBoundingBox> get_positives();
    virtual void set_HandBB(cv::Rect newHandBB);
    virtual std::shared_ptr<ImRGBZ> load_im();
  };
  
  class MetaData_Orthographic;
    
  // e.g. MetaData_RGBCentric, MetaData_DepthCentric, MetaData_Orthographic
  typedef MetaData_DepthCentric DefaultMetaData;    
}

#endif

