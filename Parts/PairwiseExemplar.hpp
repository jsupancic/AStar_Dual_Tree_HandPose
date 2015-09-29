/**
 * Copyright 2013: James Steven Supancic III
 **/


#ifndef DD_EXEMPLAR_PW
#define DD_EXEMPLAR_PW

#include "RespSpace.hpp"
#include "Detection.hpp"
#include <stddef.h>

namespace deformable_depth
{
  // Implement Xianxin's pairwise model here.
  class ExemplarPairwiseModel : public DeformationModel
  {
  protected:
    // examples[pose][root][part] = [metadata1 , ...., metadataN]
    typedef 
    map<string,
      map<string,
	map<string,
	  vector<shared_ptr<MetaData> >
	>
      >
    > ExemplarSet;
    
    ExemplarSet examples;
    set<string> poses, parts;
    string root_name;
    mutable mutex monitor;
    
    DetectorResult merge_one_exemplar_position(
      shared_ptr<MetaData>&exemplar,
      DetectorResult&root_candidate,
      PartMessages& part_manifolds,
      pw_cost_fn pwResp,
      string root_pose,
      long&pairs,EPM_Statistics&stats,const ImRGBZ&im,
      DetectionFilter&filter) const;
    void compute_scale_offset(
      MetaData&exemplar,
      DetectionFilter&filter,
      string&part_name,
      Rect_<double> bb_joint_detection_root,
      Vec2d&scale_range,
      double&raw_part_min_area,double&raw_part_max_area,bool&part_occluded
      ) const;
    void compute_part_offset_ortho(
      MetaData&exemplar,
      DetectionFilter&filter,const string&part_name,
      Rect_<double> root_ortho_bb,
      double&part_x,double&part_y) const;
    void compute_part_offset_affine(
      MetaData&exemplar,
      DetectionFilter&filter,const string&part_name,
      Rect_<double> root_ortho_bb,
      double&part_x,double&part_y) const;
    shared_ptr<DetectionManifold> merge_prep_manifold
      (string root_pose,
       shared_ptr< DetectionManifold > root_manifold,
       PartMessages& part_manifolds,DetectionFilter&filter,
       EPM_Statistics&stats,const ImRGBZ&im) const;
    void merge_per_position(
      shared_ptr<DetectionManifold> joint_manifold,
      string root_pose,
      shared_ptr< DetectionManifold > root_manifold,
      PartMessages& part_manifolds,pw_cost_fn pwResp,
      DetectionFilter&filter,EPM_Statistics&stats,const ImRGBZ&im) const;  
    void merge_check_part_candidate(
      DetectorResult&exemplars_result,
      DetectorResult&best_merged,
      const ManifoldCell_Tree::iterator::value_type*part_candidate,
      pw_cost_fn pwResp,
      string&part_name,
      DetectionFilter&filter,
      EPM_Statistics&stats,long&pairs,
      double&part_candidates,
      Vec2d scale_range,bool part_occluded) const;      
    void cluster_exemplars(
	map<int,std::tuple<shared_ptr<MetaData>,string,string,string> >&metadata_enumerations,
	Mat&data, Mat&data_centers);
      
  public:
    virtual shared_ptr<DetectionManifold> merge(
      string root_pose,
      shared_ptr< DetectionManifold > root_manifold,
      PartMessages& part_manifolds,
      pw_cost_fn pwResp, const ImRGBZ&im, DetectionFilter&filter, Statistics&stats) const;
    virtual void train_pair(
	vector<shared_ptr<MetaData> > train_set,
	string pose_name,
	string root_name,
	string part_name);
    void optimize();
    virtual Mat vis_model(Mat& background, DetectorResult& dets) const ;
    virtual Mat vis_model_offline(Mat& background, DetectorResult& dets) const ;
    int size() const;
    
    friend void read(cv::FileNode, deformable_depth::ExemplarPairwiseModel&);
    friend void write(cv::FileStorage&, std::string, const deformable_depth::ExemplarPairwiseModel&);     
  };
  void read(cv::FileNode, deformable_depth::ExemplarPairwiseModel&);
  void write(cv::FileStorage&, std::string, const deformable_depth::ExemplarPairwiseModel&);    
}

#endif
