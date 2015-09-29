/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_HAND_MODEL
#define DD_HAND_MODEL

#include "Detector.hpp"
#include "OneFeatureModel.hpp"
#include "GlobalMixture.hpp"
#include "RespSpace.hpp"
#include "StarModel.hpp"

#include <functional>
#include <map>

namespace deformable_depth
{
  using std::function;
  using std::map;
  
  // part selectors
  typedef std::function<map<string,AnnotationBoundingBox>(const map<string,AnnotationBoundingBox>&all_pos)> PartSelector;
  extern PartSelector select_fingers;
  extern PartSelector select_hand;
  
  /**
   * To change the model we need to update
   * (1) detect
   * (2) Train (non-w stuff)
   * (2) extractPos
   * (3) update_model
   **/
  class HandFingerTipModel : public StarModel
  {
  public:
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual SparseVector extractPos(MetaData&metadata,AnnotationBoundingBox bb) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    virtual Mat vis_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;
    virtual bool write(FileStorage&fs);
    
    static void filter_lr_using_face(
      const ImRGBZ&im,DetectionSet&left_dets,DetectionSet&right_dets);
  protected:
    void train_sizes(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    virtual void prime_parts(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    virtual void prime_root(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());    
    virtual double getC() const;
    virtual string getRootPartName() const;
    bool part_is_root_concreate(string part_name) const;
    
  // constants
  public:
    static constexpr double handMinArea = 50*50;
    static constexpr double fingerMinArea = 4*4;
    static constexpr double wristMinArea = 8*8;
    
    static constexpr double fingerNAREA = 1*1;
    static constexpr double handNAREA = 8*8;
    static constexpr double wristNAREA = 2*2;
    
    static constexpr double wristAreaVariance = 1.5;
    static constexpr double fingerAreaVariance = 1.5;
    static constexpr double handAreaVariance = 1.0;
    
    static constexpr double wristC = .1;
    static constexpr double fingerC = .1;
    static constexpr double handC = .1;

  protected:
    // size information
    double finger_gtMuX, finger_gtMuY, finger_count;    
    
    friend void read(const FileNode& node, HandFingerTipModel& model, 
	    const HandFingerTipModel& default_value);
  };
  
  class HandFingerTipModel_Builder : public Model_Builder
  {
  public:
    virtual Model* build(Size gtSize,Size imSize) const;
    virtual string name() const;
    virtual ~HandFingerTipModel_Builder() {};
  };
      
  void read(const FileNode& node, HandFingerTipModel& model, 
	    const HandFingerTipModel& default_value = HandFingerTipModel());
}

#endif
