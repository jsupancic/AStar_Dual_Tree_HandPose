/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_PZPS_MODEL
#define DD_PZPS_MODEL
#include "Model.hpp"
#include "OneFeatureModel.hpp"

namespace deformable_depth
{
  //
  // pseudo-zernike poly-span model
  //
  class PZPS : public Model
  {
  public:
    
  protected:
    
  public: 
    // Model overrides
    PZPS(Size gtSize, Size ISize);
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,
		       TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    virtual ~PZPS(){};    
    virtual bool write(FileStorage&fs);
  }; 
  
  class PZPS_Builder : public Model_Builder
  {
  public:
    PZPS_Builder(double C = OneFeatureModel::DEFAULT_C, 
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
