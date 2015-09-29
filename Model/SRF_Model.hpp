/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_SRF_MODEL
#define DD_SRF_MODEL

#include "Model.hpp"
#include <functional>
#include "OneFeatureModel.hpp"
#include <boost/multi_array.hpp>
#include "GeneralizedHoughTransform.hpp"
#include "PCA_Pose.hpp"
#include "RandomHoughFeature.hpp"
#include "RandomForest.hpp"
#include "Detection.hpp"

namespace deformable_depth
{  
  class SRFModel;

  /// 
  /// Single Structural Random Tree
  ///
  class HoughTree
  {
  public:
    // stop splitting if the det of the covariance falls below
    static constexpr int MIN_CoVar_DET = 5;
    // stop splitting if the number of examples at a node falls below
    static constexpr int MIN_EXAMPLES = 500;
    static constexpr int MIN_POS_EXAMPLES = 25;
    static constexpr int MIN_NEG_EXAMPLES = 25;
    // stop splitting at MAX_DEPTH
    static constexpr int MAX_DEPTH = 10;
    static constexpr int FEATURES_PER_SPLIT = 128;
    
  protected:
    HoughTree(SRFModel&model);
    int depth;
    unique_ptr<RandomHoughFeature> predictor;
    unique_ptr<HoughTree> true_branch;
    unique_ptr<HoughTree> false_branch;
    SRFModel&model;
    
    bool allow_split_true() const;
    bool allow_split_false() const;
    
  public:
    VoteResult vote(HoughOutputSpace&output,
		const StructuredWindow&swin,
		RandomHoughFeature*predictor = nullptr) const;
    const SRFModel& getModel() const;
    SRFModel& getModel();
    
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::HoughTree&);
    friend void read(const cv::FileNode&, deformable_depth::HoughTree&);
    friend void read(const cv::FileNode&, deformable_depth::SRFModel&, deformable_depth::SRFModel);
  };
  
  void read(const cv::FileNode&, deformable_depth::HoughTree&);
  void write(cv::FileStorage&, std::string&, const deformable_depth::HoughTree&);
  
  /// 
  /// Single Structural Random Tree for Hough Transforms
  ///  
  class StructuralHoughTree : public HoughTree
  {
  public:
    StructuralHoughTree(SRFModel&model,vector< StructuredExample >& training_examples, int depth = 0);
  };
  
  ///
  /// single Discriminative Hough Tree
  ///
  class DiscriminativeHoughTree : public HoughTree
  {
  public:
    DiscriminativeHoughTree(
      SRFModel&model,
      vector< StructuredExample >& training_examples, int depth = 0);
    
  protected:
    vector< StructuredExample > training_examples;
    
    enum ObjType
    {
      Spearman,
      Shannon,
      Differential,
      Pose_Entropy
    };
    
    static void discriminative_split(
      list<DiscriminativeHoughTree*>&unsplit_nodes,
      map<string,HoughOutputSpace::Matrix3D>&target_output);
    static void discriminative_try_split(
      DiscriminativeHoughTree* node,
      double&best_info_gain,
      DiscriminativeHoughTree**best_node,
      vector<StructuredExample>&exs_true,
      vector<StructuredExample>&exs_false,
      unique_ptr<RandomHoughFeature>&best_feature,
      map<string,HoughOutputSpace::Matrix3D>&target_output,
      ObjType obj_type);
    static void discriminative_split_choose
      (list<DiscriminativeHoughTree*>&unsplit_nodes,
      map<string,HoughOutputSpace::Matrix3D>&target_output,
      DiscriminativeHoughTree** best_node,
      double&best_info_gain,
      unique_ptr<RandomHoughFeature>&best_feature,
      vector<StructuredExample>&exs_true, 
      vector<StructuredExample>&exs_false);
    static void discriminative_split_update_tree
      (list<DiscriminativeHoughTree*>&unsplit_nodes,
      DiscriminativeHoughTree* best_node,
      double&best_info_gain,
      unique_ptr<RandomHoughFeature>&best_feature,
      vector<StructuredExample>&exs_true, 
      vector<StructuredExample>&exs_false);
  };
  
  //
  // Structural Random Forest Model, makes 
  // 
  class SRFModel : public Model
  {
  protected:
    Size gtSize, ISize;
    shared_ptr<HoughTree> tree_root;
    PCAPose pca_pose;
    
    void augment_features();
    
  public: // Model overrides
    SRFModel();
    SRFModel(Size gtSize, Size ISize);
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,
		       TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    virtual ~SRFModel(){};    
    virtual bool write(FileStorage&fs);
    
    const PCAPose&getPCApose() const;
    PCAPose&getPCApose();
    
    friend void read(const cv::FileNode&, deformable_depth::SRFModel&, deformable_depth::SRFModel);
  };
  void read(const cv::FileNode&, deformable_depth::SRFModel&, deformable_depth::SRFModel);
  
  class SRFModel_Model_Builder : public Model_Builder
  {
  public:
    SRFModel_Model_Builder(double C = OneFeatureModel::DEFAULT_C, 
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

