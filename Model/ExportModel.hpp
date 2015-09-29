/**
 * Copyright 2015: James Steven Supancic III
 **/

#ifndef DD_EXPORT_MODEL
#define DD_EXPORT_MODEL

#include "Model.hpp"

namespace deformable_depth
{
  ///
  /// Model which exports training and testing set (detections) to a uniform format
  /// which can then be analyzed by a classifier (eg BVLC Caffe).
  ///
  class ExportModel : public Model
  {
  protected:

  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;    
  };
}

#endif
