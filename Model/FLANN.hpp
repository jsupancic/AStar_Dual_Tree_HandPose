/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_FLANN_MODEL_NN
#define DD_FLANN_MODEL_NN

#include "Model.hpp"
#include "ScanningWindow.hpp"

namespace deformable_depth
{    
  class FLANN_Model : public Model
  {
  protected:
    // trained FLANN index
    shared_ptr<cv::flann::Index> flann_index;
    // for regressing pose
    map<int,string> index2pose;
    ExtractedTemplates extracted_templates;
    // the database might need to stick around?
    Mat template_database;
    int index = 0;

    // protected methods section
    DetectorResult detect_one(const ImRGBZ&im,DetectorResult&window,double theta) const;

  public:
    // pure virtual methods
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
  };    
}

#endif
