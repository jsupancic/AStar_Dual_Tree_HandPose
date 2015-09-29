/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_PEDESTRIAN_MODEL
#define DD_PEDESTRIAN_MODEL

#include "Model.hpp"
#include "OneFeatureModel.hpp"
#include "OcclusionReasoning.hpp"
#include "ExternalModel.hpp"
#include "FeatInterp.hpp"
#include "Platt.hpp"

namespace deformable_depth
{
  class StackedModel : public ExposedLDAModel
  {
  protected:
    // detectors
    unique_ptr<OccAwareLinearModel> occ_model;
    unique_ptr<BaselineModel_ExternalKITTI> dpm_model;
    unique_ptr<AreaModel> area_model;
    //shared_ptr<LogisticPlatt> dpm_platt;
    //shared_ptr<LogisticPlatt> occ_platt;
    Size ISize, TSize;
    
    // joint model
    shared_ptr<LDA> learner;
    shared_ptr<FeatureInterpretation> feat_interpreation;

    // detection merge straties
    virtual DetectionSet detect_dpm2occ(const ImRGBZ&im,DetectionFilter filter) const;
    virtual DetectionSet detect_occ2dpm(const ImRGBZ&im,DetectionFilter filter) const;
    virtual DetectionSet detect_dpmThenOcc(const ImRGBZ&im,DetectionFilter filter) const;
    
  public: 
    // Model overrides
    StackedModel(Size TSize, Size ISize);
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,
		       TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    virtual ~StackedModel(){};    
    virtual bool write(FileStorage&fs);
    
    // ExposedLDAModel overrides
    virtual SparseVector extractPos(MetaData&metadata,AnnotationBoundingBox bb) const;
    virtual LDA&getLearner();
    virtual void update_model();
  };
  
  class StackedModel_Builder : public Model_Builder
  {
  public:
    StackedModel_Builder(double C = OneFeatureModel::DEFAULT_C, 
			  double area = OneFeatureModel::DEFAULT_AREA,
			  shared_ptr<IHOGComputer_Factory> fc_factory = 
			  shared_ptr<IHOGComputer_Factory>(new COMBO_FACT),
			  double minArea = Model_RigidTemplate::default_minArea,
			  double world_area_variance = 
			      OneFeatureModel::DEFAULT_world_area_variance);
    virtual Model* build(Size gtSize,Size imSize) const;
    virtual string name() const;    
  };  
}

#endif
