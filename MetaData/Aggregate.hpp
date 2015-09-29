/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_METADATA_AGGREGATE
#define DD_METADATA_AGGREGATE

#include "MetaData.hpp"
#include "YML_Data.hpp"

namespace deformable_depth
{
  // represents a frame containing multiple objects (e.g. a left and right hand).
  // aggregates them in a semi-sensible way while requiring minimal modification to
  // existing code. 
  class MetaDataAggregate : public MetaData_YML_Backed
  {
  protected:
    map<string,shared_ptr<MetaData_YML_Backed> > sub_data;

  public:
    // construction and destruction
    MetaDataAggregate();
    MetaDataAggregate(map<string,shared_ptr<MetaData_YML_Backed> > sub_data);
    virtual ~MetaDataAggregate();

    // implementation of virtual methods
    virtual void setSegmentation(Mat&segmentation) override;
    virtual Mat getSemanticSegmentation() const override;
    virtual map<string,MetaData* > get_subdata() const override; 
    virtual map<string,MetaData_YML_Backed* > get_subdata_yml() override; 
    virtual void set_HandBB(cv::Rect newHandBB) override;
    virtual map<string,AnnotationBoundingBox > get_positives() override;
    virtual std::shared_ptr<ImRGBZ> load_im() override;    
    virtual std::shared_ptr<const ImRGBZ> load_im() const override;
    virtual bool leftP() const override;
    // keypoint functions
    int keypoint();
    bool hasKeypoint(string name);
    pair<Point3d,bool> keypoint(string name);
    virtual void keypoint(string, Point3d value, bool vis);
    virtual void keypoint(string, Point2d value, bool vis);
    vector<string> keypoint_names();
    
    //
    virtual bool use_negatives() const override;
    virtual bool use_positives() const override;
  };
}

#endif
