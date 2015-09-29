/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_EXTERNAL_MODEL
#define DD_EXTERNAL_MODEL

#include "Model.hpp"
#include "OneFeatureModel.hpp"
#include "FauxLearner.hpp"
#include "LibHandRenderer.hpp"

namespace deformable_depth
{
  class BaselineModel_ExternalKITTI : public SettableLDAModel
  {
  public:
    static constexpr double BB_OL_THRESH = .50;
    static constexpr double BETA_EXTERN_CONF = 125;
    
  protected:
    unique_ptr<LDA> learner;
    vector<string> test_dirs, train_dirs;
    
  public: 
    // Model overrides
    BaselineModel_ExternalKITTI(vector<string>train_dirs,vector<string>test_dirs);
    BaselineModel_ExternalKITTI();
    BaselineModel_ExternalKITTI(Size gtSize, Size ISize);
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,
		       TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    virtual ~BaselineModel_ExternalKITTI(){};    
    virtual bool write(FileStorage&fs);
    
    // ExposedLDAModel overrides
    virtual SparseVector extractPos(MetaData&metadata,AnnotationBoundingBox bb) const;
    virtual LDA&getLearner();
    virtual void update_model();    
    virtual void setLearner(LDA*lda);
    
    // model specfic params
    static vector<double> minFeat();
    vector<double> getW() const;
  }; 
  
  // synthesize the libhand metadata which matches the hint best using inverse kinematics
  shared_ptr<MetaData> oracle_synth(MetaData&hint) ;

  // model uses human annotations processed with Inverse-kinnmematics (eg libhand)
  class IKAnnotatedModel : public Model
  {
  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
    //bool is_part_model() const override {return true;};

  protected:    
    Mat correct_metric_size(
      MetaData&hint,
      LibHandRenderer&armed,LibHandRenderer&no_arm,LibHandRenderer&segm,
      const ImRGBZ&im,Rect handBB) const;
  };

  // get annotations from a human user
  class HumanModel : public Model
  {
  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;    
  };

  ///
  /// [1] D. Tang and T. Kim, “Latent Regression Forest : Structured Estimation of 3D Articulated Hand Posture.”
  ///
  class ExternalLRF_Model : public Model
  {
  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
  };

  class ExternalModel_Builder : public Model_Builder
  {
  public:
    ExternalModel_Builder(double C = OneFeatureModel::DEFAULT_C, 
			  double area = OneFeatureModel::DEFAULT_AREA,
			  shared_ptr<IHOGComputer_Factory> fc_factory = 
			  shared_ptr<IHOGComputer_Factory>(new COMBO_FACT),
			  double minArea = Model_RigidTemplate::default_minArea,
			  double world_area_variance = 
			      OneFeatureModel::DEFAULT_world_area_variance);
    virtual Model* build(Size gtSize,Size imSize) const;
    virtual string name() const;    
  };  

  ///
  /// Kinect Segmentation based detection + Kinect RF Pose Estimation
  /// 
  class KinectSegmentationAndPose_Model : public Model
  {
  protected:
    virtual DetectionSet detect_with_seg_heuristic(const ImRGBZ&im,DetectionFilter filter) const;
    virtual DetectionSet detect_with_hough_forest(const ImRGBZ&im,DetectionFilter filter) const; 
    virtual void emplace_kinect_poses(const ImRGBZ&im, DetectionFilter filter, DetectionSet&dets) const;

  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
  };

  ///
  /// Yi's Deep Model
  ///
  class DeepYiModel : public Model
  {
  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;    
  };

  ///
  /// Jonathan Tompson's [NYU] model.
  ///
  class NYU_Model : public Model
  {
  protected:
    Mat us, vs;

  public:
    virtual Visualization visualize_result(const ImRGBZ&,Mat&background,DetectionSet&dets) const;
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;    
  };

  ///
  /// NOP Model. Model does nothing, just shows data.
  ///
  class NOP_Model : public Model
  {
  protected:

  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;    
  };

  ///
  /// Kitani's model
  /// Li, C., & Kitani, K. M. (2013). Pixel-Level Hand Detection in Ego-centric Videos. 
  //  2013 IEEE Conference on Computer Vision and Pattern Recognition, 3570–3577. doi:10.1109/CVPR.2013.458
  ///
  class KitaniModel : public Model
  {
  protected:

  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;    
  };
}

#endif
