/**
 * Copyright 2013: James Steven Supancic III
 **/


#ifndef DD_RESP_SPACE
#define DD_RESP_SPACE

#include <boost/multi_array.hpp>

#include "Detector.hpp"
#include "util_mat.hpp"
#include "GlobalMixture.hpp"
#include "MetaData.hpp"

#include <functional>

namespace deformable_depth
{
  using namespace std;
  
  // we provide space for 3 types of detection at each cell:
  // occluded, partially occluded, and visible. 
  class ManifoldCell_3Occ
  {
  protected:
    enum Cell_Type
    {
      full_occ, part_occ, visible
    };
    DetectionSet contents;    
    Cell_Type cellType(const DetectorResult&result);
    
  public:
    typedef DetectionSet::iterator iterator;
    
    int conflict_count(const ManifoldCell_3Occ&other);
    void emplace(const DetectorResult&det);
    float max_score();
    void merge(const ManifoldCell_3Occ&from_other);
    iterator begin();
    iterator end();
  };
  
  // store all emplaced results in a BST
  class ManifoldCell_Tree
  {
  protected:
    map<float/*BB area*/,DetectorResult> contents;
    
  public:
    typedef map<float/*BB area*/,DetectorResult>::iterator iterator;
    
    static const DetectorResult& deref(const iterator::value_type&iter);
    static DetectorResult& deref(iterator::value_type&iter);
    float max_score();
    void emplace(const DetectorResult&det);
    int conflict_count(const ManifoldCell_Tree&other);
    void merge(const ManifoldCell_Tree&other);
    iterator begin(float min_area);
    iterator end(float max_area);
    double size() const;
  };
  
  typedef ManifoldCell_Tree ManifoldCell;
  
  class DetectionManifold : public boost::multi_array<ManifoldCell,2 >
  {
  public:
    DetectionManifold(int xres,int yres);
    size_t xSize();
    size_t ySize();
    void concat(const DetectionManifold&other);
    DetectionManifold max_pool(int amount);
  };
  
  struct EPM_Statistics;
  typedef std::function<float(const Detection&root,const Detection&part)> pw_cost_fn;
  shared_ptr<DetectionManifold> create_manifold(const ImRGBZ& im, DetectionSet& detections);
  // it is often desirable to sort detections with different poses into different
  // manifolds. E.g. if a members of a local mixutre are to be treated differently. 
  map<string, DetectionManifold > create_sort_manifolds(const ImRGBZ& im, DetectionSet& detections);
  Mat draw_manifold(DetectionManifold&manifold);
  DetectorResult merge_manifolds_one_pair(
    Vec2d scales,
    const Detection& det1,const Detection& det2,
    pw_cost_fn pwResp,
    long&pairs, string part_name, EPM_Statistics&stats,
    DetectionFilter&filter);  
  
  void part_offset(
    Rect_<double> bb_root, 
    Rect_<double> bb_part,
    const shared_ptr<const ImRGBZ>&im,
    Vec2d&current_offset,
    Vec2d&cur_perspective_offset,
    double&current_scale,
    Rect_<double>&ortho_root, 
    Rect_<double>&ortho_part);  
    
  // these are messages passed via Dynamic programming in a star model.
  typedef map<string/*part name*/,map<string/*pose*/,DetectionManifold>> PartMessages;
    
  // pairwise model represents model between two pairs of parts,
  // while deformation model captures the more general idea of how 
  // parts relate to each other.
  class DeformationModel
  {
  public:
    virtual shared_ptr<DetectionManifold> merge(
      string root_pose,
      shared_ptr< DetectionManifold > root_manifold,
      PartMessages& part_manifolds,
      pw_cost_fn pwResp,const ImRGBZ&im,DetectionFilter&filter, Statistics&stats) const = 0;
   virtual void train_pair(
	vector<shared_ptr<MetaData> > train_set,
	string pose_name,
	string root_name,
	string part_name) = 0;
    virtual Mat vis_model(Mat& background, DetectorResult& dets) const = 0;
  };
}

#include "PairwiseDeformationWindow.hpp"
#include "PairwiseExemplar.hpp"

#endif
