/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_METADATA_POOLED
#define DD_METADATA_POOLED

#include "MetaData.hpp"

namespace deformable_depth
{
  class MetaData_Pooled : public MetaData
  {
  public:
    typedef function<MetaData* ()> AllocatorFn;

  public:
    MetaData_Pooled(AllocatorFn allocatorFn,bool use_positives,bool use_negatives,string filename);
    MetaData_Pooled(string filename,bool read_only);
    virtual ~MetaData_Pooled();

    // Metadata functions
    virtual map<string,AnnotationBoundingBox> get_positives() override;
    virtual std::shared_ptr<ImRGBZ> load_im() override;  
    virtual Mat load_raw_RGB() override;
    virtual Mat load_raw_depth() override;
    virtual std::shared_ptr<const ImRGBZ> load_im() const override;
    virtual string get_pose_name() override;
    virtual string get_filename() const override;
    // keypoint functions
    virtual pair<Point3d,bool> keypoint(string name) override;
    virtual int keypoint() override;
    virtual bool hasKeypoint(string name)override;
    virtual vector<string> keypoint_names() override;    
    virtual DetectionSet filter(DetectionSet src)override;
    virtual bool leftP() const override;    
    virtual bool use_negatives() const override;
    virtual bool use_positives() const override;
    virtual bool loaded() const override;
    virtual Mat getSemanticSegmentation() const override;
    virtual void drawSkeletons(Mat&target,Rect boundings) const override;

    // extended functions
    virtual shared_ptr<MetaData> getBackend() const;

  protected:
    void require_loaded() const;

    // always valid
    AllocatorFn allocatorFn;
    string filename;
    bool b_use_positives, b_use_negatives;

    // sometimes null
    mutable shared_ptr<MetaData> backend;
  };
}

#endif
