/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_AONN_MODEL
#define DD_AONN_MODEL

#include "Model.hpp"
#include "HeuristicTemplates.hpp"

namespace deformable_depth
{
  // absolute orientation + AND/OR Nearest Neighbour model
  class AONN_Model : public Model
  {
  protected:
    // the processed training set
    vector<shared_ptr<MetaData>> training_set;
    map<string,map<string,AnnotationBoundingBox> > parts_per_ex;
    map<string,NNTemplateType> allTemplates;
    map<string,string> sources; // map uuids for the above to filenames...

    // protected method section
    DetectionSet detect_linear(const ImRGBZ&im,DetectionFilter filter) const;
    DetectionSet detect_AStar(const ImRGBZ&im,DetectionFilter filter) const;
    
  public:
    // ctor
    AONN_Model();
    // virtual method implementations
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
  };
}

#endif
