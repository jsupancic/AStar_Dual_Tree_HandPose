/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_APX_NN
#define DD_APX_NN

#include "Model.hpp"
#include "boost/multi_array.hpp"
#include "OcclusionReasoning.hpp"
#include "HeuristicTemplates.hpp"
#include "AStarNN.hpp"

namespace deformable_depth
{    
  struct NNSearchParameters
  {
    bool stop_at_window;
    NNSearchParameters() : stop_at_window(false) {};
  };
  

  // model which implements various fast (approx. and exact) NN inferance methods.
  // using cascades. 
  class ApxNN_Model : public Model
  {
  public:
    
  protected:  
    // protected data members requring serialization
    SearchTree search_tree_root;    
    map<string,map<string,AnnotationBoundingBox> > parts_per_ex;
    map<string,NNTemplateType> allTemplates;
    map<string,string> sources; // map uuids for the above to filenames...
    // protected transient members not needing serialization
    vector<shared_ptr<MetaData>> training_set;
    mutable mutex monitor;
    mutable map<string,Mat> respImages;
    ManifoldFn manifoldFn;
    
    // protected methods
    DetectionSet detect_for_window
      (NNSearchParameters,const ImRGBZ& im,
       Rect bb_orig,
	const vector<Rect>&bbs_faces,
	Mat&respIm,Mat&invalid,	
       EPM_Statistics&stats,DetectionFilter&filter) const;
    DetectorResult detect_at_position
      (NNTemplateType&X,Rect orig_bb,float depth,
       Mat&respIm,Mat&invalid,EPM_Statistics&stats) const;
    map<string,NNTemplateType> train_extract_templates(
      vector< shared_ptr< MetaData > >& training_set, TrainParams train_params);
    virtual DetectionSet detect_Astar(const ImRGBZ&im,DetectionFilter filter) const;
    void set_parent(int pyr_level,DetectorResult&child,DetectionSet&parent_cell,DetectionSet&all_parents) const;
    bool accept_det(MetaData&hint,Detection&det,Vec3d gt_up,Vec3d gt_norm, bool fliplr) const;
    Mat vis_result_agreement(const ImRGBZ&im,Mat& background, DetectionSet& dets) const;
    Mat vis_adjacent(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;

  public: 
    // exported functions specific to this class.
    virtual DetectionSet detect_linear(const ImRGBZ&im,DetectionFilter filter,NNSearchParameters) const;  

    // Model overrides
    virtual Mat vis_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;
    virtual Visualization visualize_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;
    ApxNN_Model(Size gtSize, Size ISize);
    ApxNN_Model();
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,
		       TrainParams train_params = TrainParams());
    virtual void train_on_test(vector<shared_ptr<MetaData>>&training_set,
			       TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "ApxNN_Model");
    virtual ~ApxNN_Model(){};    
    virtual bool write(FileStorage&fs);
    // part model overrides
    bool is_part_model() const override {return true;};

    friend struct AStarSearch;
    friend class InformedSearch;
  };   
  
  class ApxNN_Builder : public Model_Builder
  {
  public:
    ApxNN_Builder();
    virtual Model* build(Size gtSize,Size imSize) const;
    virtual string name() const;    
  };      
}

#endif

