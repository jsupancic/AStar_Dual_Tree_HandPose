/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_OCCREASON
#define DD_OCCREASON

#include "MetaFeatures.hpp"
#include "OneFeatureModel.hpp"
#include "OcclusionFeat.hpp"

namespace deformable_depth
{  
  /**
   * This class implements special logic necessarily to efficently use the
   *  OccSliceFeature.
   * 
   **/
  class OccAwareLinearModel : public SparseOneFeatureModel
  {
  public:
    OccAwareLinearModel(
      Size gtSize, Size ISize,
      double C = OneFeatureModel::DEFAULT_C, 
      double area = OneFeatureModel::DEFAULT_AREA,
      shared_ptr<IHOGComputer_Factory> comp_factory = 
	shared_ptr<IHOGComputer_Factory>(new Default_Computer_Factory()),
      shared_ptr<LDA> learner = shared_ptr<LDA>(new QP()));   
    OccAwareLinearModel(shared_ptr<IHOGComputer_Factory> comp_factory);
    virtual Mat vis_result(Mat&background,DetectionSet&dets) const;
    virtual void debug_incorrect_resp(SparseVector&feat,Detection&det);
    struct PosInfo
    {
      SparseVector sp_vec;
      double real, occ, bg;
    };
    virtual PosInfo extract(const ImRGBZ&im, AnnotationBoundingBox bb,bool pos) const;  
    virtual float getObjDepth() const;
    
  protected:
    DetectionSet do_extern(
      vector< Intern_Det >& interns, string filename, 
      shared_ptr< FeatIm > im_feats, Size im_blocks, DetectionFilter filter,
      shared_ptr<CellDepths> cell_depths,
      const vector<float>&occ_percent, const vector<float>&real_percent,double scale) const;
    void do_dot_extract_one_feat(
      float&xValue, bool&occluded,bool&background,
      int x0, int xIter, int y0, int yIter,
      int bin,float cell_depth, float min_depth, float max_depth,
      shared_ptr< FeatIm >&im_feats,DepthFeatComputer& hog_for_scale,
      const vector<double>&depths_histogram) const;
    float do_dot(     int y0, int x0, double manifold_z,
		      int blocks_x, int blocks_y, int nbins, 
		      const ImRGBZ&im,
		      std::vector< float >& wf, 
		      shared_ptr< FeatIm > im_feats, 
		      DepthFeatComputer& hog_for_scale,const CellDepths&cellDepths,
		      const DetectionFilter&filter,
		      float&occ_percent,float&real_perc) const;
    virtual void prime_learner();
    virtual Mat show(const string&title);
    void chose_pos_depth
      (const AnnotationBoundingBox&bb, const Mat&cell_depths,
       float&min_depth,float&max_depth) const;
    vector<float> flatten_cell_depths(const Mat&cell_depths,
      int xmin, int ymin, int xsup, int ysup) const;
    virtual SparseVector extractPos(MetaData&metadata, AnnotationBoundingBox bb) const;  
    void init();
    virtual DetectionSet detect_at_scale(const ImRGBZ&im, DetectionFilter filter,FeatPyr_Key  scale) const;
    virtual void init_post_construct();  
    
    float obj_depth;
  protected:
    mutable atomic<int> times_logged;
    shared_ptr< IHOGComputer_Factory >  occ_factory;
    shared_ptr< OccSliceFeature > occ_hog, occ_hog1;
  };  
  
  bool valid_configuration(bool part_visible, bool part_overlaps_visible);
  bool occlusion_intersection(const Rect_<double>&r1, const Rect_<double>&r2);
  bool is_visible(float occ);
}

#endif
