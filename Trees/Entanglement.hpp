/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_ENTANGLEMENT
#define DD_ENTANGLEMENT

#include "Model.hpp"
#include "RandomHoughFeature.hpp"

namespace deformable_depth
{
  ///
  /// Class representing a single entangled random tree
  ///
  class EntangledTree 
  {
  protected:
    unique_ptr<RandomFeature> predictor;
    unique_ptr<EntangledTree> true_branch;
    unique_ptr<EntangledTree> false_branch;
    double true_samples,false_samples; 
    
    // used only during training
    vector<StructuredExample> leaf_correct_exs;
    
  public:
    double prediction() const;
    EntangledTree(const EntangledTree&copy);
    EntangledTree(double true_samples = 1, double false_samples = 1);
    template<typename T>
    DetectionSet detect(const vector<T>&windows) const;
    virtual shared_ptr<EntangledTree> grow(const vector<StructuredExample >&) const;
    virtual bool grow_self(const vector<StructuredExample >&, bool do_split = true);
    
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::EntangledTree&);
  };
  
  //
  // Entangled Random Forest Model
  // 
  class ERFModel : public Model
  {
  protected:
    Size gtSize, ISize;
    shared_ptr<FeatureExtractionModel> feature_extractor;
    shared_ptr<EntangledTree> tree;
    
  public: // Model overrides
    ERFModel();
    ERFModel(Size gtSize, Size ISize);
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,
		       TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    virtual ~ERFModel(){};    
    virtual bool write(FileStorage&fs);
  };
  
  class ERFModel_Model_Builder : public Model_Builder
  {
  public:
    ERFModel_Model_Builder(double C = OneFeatureModel::DEFAULT_C, 
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
