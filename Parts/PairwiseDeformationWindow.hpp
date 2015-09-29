/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_WINDOW_PW
#define DD_WINDOW_PW

#include "RespSpace.hpp"

namespace deformable_depth
{
  class FixedWindowPairwiseModel
  {
  protected:
    Vec2d offset, pers_offset;
    double min_scale;
    double max_scale;
    double min_offset_x, min_offset_y, max_offset_x, max_offset_y;
        
  public:
    FixedWindowPairwiseModel(){};
    FixedWindowPairwiseModel(
      SupervisedMixtureModel::TrainingSet&training_set,
      string part1, string part2);
    virtual ~FixedWindowPairwiseModel();
    pw_cost_fn getFunction();
    Vec2i getOffset() const;
    Vec2d getPersOffset() const;
    Vec2i getWindowSize() const;
    Vec2d getScales() const;
    friend void write(FileStorage&, std::string&, const FixedWindowPairwiseModel&);
    friend void read(FileNode, FixedWindowPairwiseModel&, FixedWindowPairwiseModel); 
    shared_ptr<DetectionManifold> merge_manifolds(
      DetectionManifold&m1, DetectionManifold&m2, pw_cost_fn pwResp,string part_name,
      DetectionFilter&filter) const;
  };
  void read(FileNode, FixedWindowPairwiseModel&, FixedWindowPairwiseModel);  
  void write(FileStorage&, std::string&, const FixedWindowPairwiseModel&);  
  
  class FixedWindowDeformationModel : public DeformationModel
  {
  protected:
    map<string,FixedWindowPairwiseModel> pw_models;  
    mutex m;
    set<string> parts;
    Vec2d get_card_position_normalized(string pose, string part) const;
    
  public:
    shared_ptr<DetectionManifold> merge(
      string root_pose,
      shared_ptr< DetectionManifold > root_manifold,
      PartMessages& part_manifolds,
      pw_cost_fn pwResp,const ImRGBZ&im,DetectionFilter&filter) const;
    virtual void train_pair(
	vector<shared_ptr<MetaData> > train_set,
	string pose_name,
	string root_name,
	string part_name);
    Mat vis_model(Mat& background, DetectorResult& dets) const;
    
    friend void read(cv::FileNode, deformable_depth::FixedWindowDeformationModel&);
    friend void write(cv::FileStorage&, std::string, const deformable_depth::FixedWindowDeformationModel&);
  };  
  void read(cv::FileNode, deformable_depth::FixedWindowDeformationModel&);
  void write(cv::FileStorage&, std::string, const deformable_depth::FixedWindowDeformationModel&);  
}

#endif
