/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#if !defined(DD_TRAINING) && !defined(WIN32)
#define DD_TRAINING

#include "Model.hpp"
#include "MetaData.hpp"

namespace deformable_depth
{
  static constexpr int NMAX_HARD_NEG_MINING = 2000;
  
  class MetaData;
  
  class ExposedTrainingSet
  {
  public:
    virtual void collect_training_examples(
      MetaData&metadata,Model::TrainParams train_params,
      function<void (DetectorResult)> writePos, function<void (DetectorResult)> writeNeg,
      DetectionFilter filter) = 0;  
  };
  
  class ExposedLDAModel : public Model, public ExposedTrainingSet
  {
  public:
    virtual SparseVector extractPos(MetaData&metadata,AnnotationBoundingBox bb) const = 0;
    virtual LDA&getLearner() = 0;
    virtual void update_model() = 0;
    virtual void collect_training_examples(
      MetaData&metadata,Model::TrainParams train_params,
      function<void (DetectorResult)> writePos, function<void (DetectorResult)> writeNeg,
      DetectionFilter filter = DetectionFilter(-1.0,NMAX_HARD_NEG_MINING));
    virtual void debug_incorrect_resp(SparseVector&feat,Detection&det);
    
  protected:
    virtual void collect_positives(
      MetaData&metadata,Model::TrainParams train_params,
      function<void (DetectorResult)> writePos, 
      DetectionFilter filter,
      map<string,AnnotationBoundingBox>&train_bbs,
      shared_ptr<ImRGBZ>&im);    
    virtual void collect_negatives(
      MetaData&metadata,Model::TrainParams train_params,
      function<void (DetectorResult)> writeNeg,
      DetectionFilter filter,
      map<string,AnnotationBoundingBox>&pos_bbs,
      map<string,AnnotationBoundingBox>&train_bbs,
      shared_ptr<ImRGBZ>&im);    
  };

  class SettableLDAModel : public ExposedLDAModel
  {
  public:
    virtual void setLearner(LDA*lda) = 0;
  };
  
  struct TrainingStatistics
  {
    int new_svs;
    int new_svs_pos, new_svs_neg;
    int hard_negatives;
    bool cache_overflowed;
    int max_new_svs_per_frame;
    int total_svs;
    map<string,size_t> pos_examples_by_pose;
    TrainingStatistics() : 
      new_svs(0), hard_negatives(0), cache_overflowed(false), 
      max_new_svs_per_frame(0), total_svs(0), 
      new_svs_pos(0), new_svs_neg(0)
    {}
  };
  
  void train_seed_positives(ExposedLDAModel&,vector<shared_ptr<MetaData>>&training_set,Model::TrainParams train_params);
  void train_parallel(ExposedLDAModel&,vector<shared_ptr<MetaData>>&training_set,Model::TrainParams train_params);
  void train_smart(ExposedLDAModel&,vector<shared_ptr<MetaData>>&training_set,Model::TrainParams train_params);
  TrainingStatistics train_parallel_do(ExposedLDAModel&,vector<shared_ptr<MetaData>>&training_set, Model::TrainParams train_params);  
}

#endif

