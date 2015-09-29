/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_STAR_MODEL
#define DD_STAR_MODEL

#include "Detector.hpp"
#include "RespSpace.hpp"
#include "FeatInterp.hpp"

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

namespace deformable_depth
{  
  class StarModel : public ExposedLDAModel
  {
  // methods
  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    StarModel();
    virtual ~StarModel(){};
    virtual SparseVector extractPos(MetaData&metadata,AnnotationBoundingBox bb) const;
    virtual LDA&getLearner();
    virtual void update_model();
    virtual void debug_incorrect_resp(SparseVector&feat,Detection&det);
    virtual double min_area() const;
    virtual Mat vis_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;
    virtual bool is_part_model() const {return true;};
    virtual bool write(FileStorage&fs);
    
  protected:
    Mat vis_feats(const ImRGBZ&im,Mat& background, DetectionSet& dets) const;
    Mat vis_part_positions(Mat& background, DetectionSet& dets) const;
    void check_pretraining(vector< shared_ptr< MetaData > >& training_set);
    void save();
    virtual void prime_parts(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) = 0;
    virtual void prime_root(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) = 0;
    void train_init_joint_interpretation();
    void train_joint(vector<shared_ptr<MetaData>>&training_set,
		     TrainParams train_params = TrainParams());
    void train_pairwise();
    virtual DetectionSet detect_manifold(const ImRGBZ&im,DetectionFilter filter,Statistics&stats) const;
    virtual shared_ptr<DetectionManifold> detect_gbl_mixture(
      const ImRGBZ&im,string pose,shared_ptr<FeatPyr> rootPyr,
      PartMessages&part_manifolds,
      DetectionFilter&filter,
      DetectionSet&joint_detections,Statistics&stats) const;
    void extract_detections_from_manifold(
      const ImRGBZ& im, 
      DetectionFilter&filter,
      DetectionSet&joint_detections,
      map<string,shared_ptr<DetectionManifold>>&global_manifolds) const;
    virtual double getC() const = 0;
    virtual bool part_is_root_concreate(string part_name) const;
    virtual string getRootPartName() const = 0;
      
  // member variables
  protected:
    // number of times update_model has been called
    int update_count;
    boost::shared_mutex monitor;
    
    // detectors
    map<string,shared_ptr<SupervisedMixtureModel> > parts;
    shared_ptr<SupervisedMixtureModel> root_mixture;
    DetectionFilter partFilter;
    
    // joint model
    shared_ptr<LDA> learner;
    shared_ptr<FeatureInterpretation> feat_interpreation;
    
    // pairwise model
    // (root_pose=>part) -> PWModel
    // FixedWindowDeformationModel OR ExemplarPairwiseModel
    ExemplarPairwiseModel def_model_exemplar;
    
    friend void write(FileStorage& fs, const string& , const StarModel& star_model);
    friend void read(const FileNode& node, StarModel& model);
  };
  
  void read(const FileNode& node, StarModel& model);
  void write(FileStorage& fs, const string& , const StarModel& star_model);  
}

#endif
