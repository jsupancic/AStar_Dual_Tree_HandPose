/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_ONE_FEATURE_MODEL
#define DD_ONE_FEATURE_MODEL

namespace deformable_depth
{
  // forward declarations
  class OneFeatureModel;
}

#include "Detector.hpp"
#include "DiskArray.hpp"
#include "MetaFeatures.hpp"

namespace deformable_depth
{ 
  struct Intern_Det;
  
  bool is_valid_pos_bb(Rect BB, const ImRGBZ&im);
    
  struct Intern_Det
  {
    float resp;
    float depth;
    int row, col;
    
    bool operator<(const Intern_Det&other) const
    {
      return resp < other.resp;
    }
  };    
  
  class OneFeatureModel : public Model_RigidTemplate
  {
  public:
    static constexpr double DEFAULT_world_area_variance = params::world_area_variance;
    
  protected:
    // RGB 15*15 > 10*10 > 6*6
    float area;
    shared_ptr<IHOGComputer_Factory> comp_factory;
    DepthFeatComputer *hog, *hog1;
    double world_area_variance;
    
  public:
    OneFeatureModel(
      Size gtSize, Size ISize,
      double C = DEFAULT_C, 
      double area = DEFAULT_AREA,
      shared_ptr<IHOGComputer_Factory> comp_factory = 
	shared_ptr<IHOGComputer_Factory>(new Default_Computer_Factory()),
      shared_ptr<LDA> learner = shared_ptr<LDA>(new QP()));
    ~OneFeatureModel();
    virtual Mat show(const string&title);
    SparseVector extractPos_w_border(
      const ImRGBZ&im, AnnotationBoundingBox bb, IHOGComputer_Factory*scale_fact = nullptr,
      function<void (DepthFeatComputer* comp)> config = [](DepthFeatComputer* comp){}) const;
    DepthFeatComputer& getHOG1();
    virtual Mat resps(const ImRGBZ&im) const;
    void setWorldAreaVariance(double world_area_variance);
    
  protected:
    OneFeatureModel(shared_ptr<IHOGComputer_Factory> comp_factory);
    virtual Size resp_impl(Mat&resp, vector<float>&imFeats,const ImRGBZ&im) const;
    // externDets: convert Intern_Det data structure to Detection data strucutre.
    // note, Detection[n] should equal Intern[n]
    virtual DetectionSet externDets(
      vector<Intern_Det>&interns,string filename,shared_ptr<FeatIm> im_feats, 
      Size im_blocks,DetectionFilter filter,double scale) const;    
    Rect_<double> externBB(int row, int col) const;
    virtual void prime_learner() = 0;
    virtual void init_post_construct();
  private:
    friend void write(FileStorage&, string&, const std::shared_ptr<OneFeatureModel>&);
    friend void read(const FileNode&, shared_ptr<OneFeatureModel>&, shared_ptr<OneFeatureModel>);
  };
  
  /**
   * Implements scanning window detector using convolution
   **/
  class DenseOneFeatureModel : public OneFeatureModel
  {
  public:
    DenseOneFeatureModel(
      Size gtSize, Size ISize,
      double C = OneFeatureModel::DEFAULT_C, 
      double area = OneFeatureModel::DEFAULT_AREA,
      shared_ptr<IHOGComputer_Factory> comp_factory = 
	shared_ptr<IHOGComputer_Factory>(new Default_Computer_Factory()),
      shared_ptr<LDA> learner = shared_ptr<LDA>(new QP()));
    virtual SparseVector extractPos(MetaData&metadata, AnnotationBoundingBox bb) const;
  protected: 
    DenseOneFeatureModel(shared_ptr<IHOGComputer_Factory> comp_factory);
    virtual void prime_learner();
    virtual DetectionSet detect_at_scale(const ImRGBZ&im, DetectionFilter filter,FeatPyr_Key  scale) const;
    friend void write(FileStorage&, string&, const std::shared_ptr<OneFeatureModel>&);
    friend void read(const FileNode&, shared_ptr<OneFeatureModel>&, shared_ptr<OneFeatureModel>);
  };
  
  /**
   * Implements sparse scanning window, skips invalid scale+location pairs.
   * 
   * In this class min_depth and max_depth are computed per scale and correspond
   *  to the depths an object of the model's size can reasonable occupy. 
   **/
  class SparseOneFeatureModel : public OneFeatureModel
  {
  public:
    SparseOneFeatureModel(
      Size gtSize, Size ISize,
      double C = OneFeatureModel::DEFAULT_C, 
      double area = OneFeatureModel::DEFAULT_AREA,
      shared_ptr<IHOGComputer_Factory> comp_factory = 
	shared_ptr<IHOGComputer_Factory>(new Default_Computer_Factory()),
      shared_ptr<LDA> learner = shared_ptr<LDA>(new QP()));
    virtual SparseVector extractPos(MetaData&metadata, AnnotationBoundingBox bb) const;
    virtual Mat show(const string&title);
    virtual Mat resps(const ImRGBZ&im) const;
  protected:  
    typedef function< float (int y0, int x0,float z_manifold,int blocks_x, int blocks_y, int nbins,const ImRGBZ&im,
		      vector<float>&wf,shared_ptr<FeatIm> im_feats,
		      DepthFeatComputer&hog_for_scale)> DotFn;
    typedef function< DetectionSet (vector< Intern_Det >& interns, 
					       string filename, shared_ptr< FeatIm > im_feats, 
					       Size im_blocks, DetectionFilter filter)> ExternFn;
    typedef function < vector<float> (int x, int y)> ManifoldFn;
    
    virtual float doDot(int y0, int x0,float z_manifold,
			int blocks_x, int blocks_y, int nbins,const ImRGBZ&im,
			vector<float>&wf,shared_ptr<FeatIm> im_feats,
			DepthFeatComputer&hog_for_scale) const ; 
    SparseOneFeatureModel(shared_ptr<IHOGComputer_Factory> comp_factory);
    virtual DetectionSet detect_at_scale(const ImRGBZ&im, DetectionFilter filter,FeatPyr_Key  scale) const;
    virtual DetectionSet detect_at_scale(const ImRGBZ&im, DetectionFilter filter,FeatPyr_Key  scale,DotFn dot,
						   ExternFn externFn,ManifoldFn manifoldFn) const;
    void check_window(
      int y0, int x0,double z_manifold,FeatPyr_Key key,const ImRGBZ&im,DetectionFilter&filter,
      int blocks_x, int blocks_y, int nbins,
      vector<float>&wf,float B,float min_world_area,float max_world_area,
      vector<Intern_Det>&idets,shared_ptr<FeatIm> im_feats,
      DepthFeatComputer&hog_for_scale,DotFn dot) const;      
    friend void write(FileStorage&, string&, const std::shared_ptr<OneFeatureModel>&);
    friend void read(const FileNode&, shared_ptr<OneFeatureModel>&, shared_ptr<OneFeatureModel>);
    void check_feats_vs_resps(vector<Intern_Det>&idets,DetectionSet&dets) const;
    virtual void prime_learner();
    void minMaxWorldAreas(const ImRGBZ&im,Rect_<double> bb0,
			  float&min_world_area,float&max_world_area,
			  DetectionFilter filter
 			) const;
    
    virtual shared_ptr<FeatIm> getFeat
	(DetectionFilter filter,FeatPyr_Key scale) const;
    virtual float getObjDepth() const ;
  };
  
  ///
  /// Section: Feature extraction by detection
  ///
  class FeatureExtractionModel : public SparseOneFeatureModel
  {
  public:
    FeatureExtractionModel(shared_ptr< IHOGComputer_Factory > comp_factory);
    FeatureExtractionModel(Size gtSize, Size ISize, 
			   double C = OneFeatureModel::DEFAULT_C, 
			   double area = OneFeatureModel::DEFAULT_AREA, 
			   shared_ptr< IHOGComputer_Factory > comp_factory = 
			      shared_ptr<IHOGComputer_Factory>(new Default_Computer_Factory()), 
			   shared_ptr< LDA > learner = shared_ptr<LDA>(new QP()));
    
  protected: // SparseOneFeatureModel overrides
    virtual float doDot(int y0, int x0,float z_manifold,
			int blocks_x, int blocks_y, int nbins,const ImRGBZ&im,
			vector<float>&wf,shared_ptr<FeatIm> im_feats,
			DepthFeatComputer&hog_for_scale) const ; 
    virtual shared_ptr<FeatIm> getFeat
      (DetectionFilter filter,FeatPyr_Key scale) const;
      
    friend void read(const cv::FileNode&, std::shared_ptr<deformable_depth::FeatureExtractionModel>&, std::shared_ptr<deformable_depth::FeatureExtractionModel>);
  };    
  void read(const cv::FileNode&, std::shared_ptr<deformable_depth::FeatureExtractionModel>&, std::shared_ptr<deformable_depth::FeatureExtractionModel>);
  
  /// SECTION: Serialization and utility 
  void read(const FileNode&, 
	    shared_ptr<OneFeatureModel>&, 
	    shared_ptr<OneFeatureModel>);
  void write(FileStorage&, string&, const std::shared_ptr<OneFeatureModel>&);
  
  class OneFeatureModelBuilder : public Model_Builder
  {
  public:
    double C;
    double AREA;
    shared_ptr<IHOGComputer_Factory> fc_factory;
    std::function<std::shared_ptr<LDA>()> lda_source;
    double minArea;
    double world_area_variance;
  public:
    OneFeatureModelBuilder(double C = OneFeatureModel::DEFAULT_C, 
		      double area = OneFeatureModel::DEFAULT_AREA,
		      shared_ptr<IHOGComputer_Factory> fc_factory = 
			shared_ptr<IHOGComputer_Factory>(new COMBO_FACT),
		      double minArea = Model_RigidTemplate::default_minArea,
		      double world_area_variance = OneFeatureModel::DEFAULT_world_area_variance);
    virtual Model* build(Size gtSize,Size imSize) const;
    virtual string name() const;    
  };  
}

#endif

