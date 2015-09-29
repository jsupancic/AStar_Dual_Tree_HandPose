/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_RANDOM_FEATURE
#define DD_RANDOM_FEATURE

#include "Detection.hpp"
#include "GeneralizedHoughTransform.hpp"
#include "OneFeatureModel.hpp"

namespace deformable_depth
{
  struct StructuredWindow
  {
    DetectorResult detection; // y
    shared_ptr<const ImRGBZ> im;
    
    Point3d       ortho_detBBCenter() const;
    Rect_<double> ortho_detBB() const;
    
    Vec2d  rgChroma(const Point2d&query) const;
    double metric_feature(const Point2d&query) const;
    double skin_likelihood(const Point2d&query) const;
    double mean_area(const Rect_<double>&query) const;
    double mean_intensity(const Rect_<double>&query) const;
    double relative_depth(const Rect_<double>&query) const;
    
    Point2d queryToImCoords(const Point2d&query) const;
  };
  
  struct StructuredExample : public StructuredWindow
  {
    shared_ptr<MetaData> metadata; // y
    
    Point3d ortho_handBBCenter() const;
    double correctness() const;
    bool is_correct() const; 
    bool is_white() const;
    bool is_black() const;
  };
  
  struct InformationGain
  {
  public:
    InformationGain();
    double shannon_info_gain;
    double differential_info_gain;
    double latent_structural_gain;
  };
  
  void write(cv::FileStorage&, std::string&, const deformable_depth::InformationGain&);
  void read(const cv::FileNode&, deformable_depth::InformationGain&, deformable_depth::InformationGain);
    
  class RandomFeature;
  class RandomHoughFeature;
  
  struct VoteResult
  {
    const RandomHoughFeature*leaf;
    double conf;
  };
  
  class RandomFeature
  {
  protected:
    enum FeatureTypes
    {
      DistanceInvarientDepth,
      SkinLikelihood,
      MeanDepth,
      MeanIntensity,
      R_Chroma,
      G_Chroma,
      Resolution,
      RelativeDepth
    };
    
    // the test
    FeatureTypes feature_type;
    Point2d p;
    Rect_<double> roi1, roi2;
    double threshold;
    
    // statistics about this node
    string uuid; // for diagnostics    
    
    double feature(const StructuredWindow&ex) const;
    RandomFeature();
    
  public:
    void split_examples(
      const vector< StructuredExample >& examples, 
      vector<StructuredExample> &exs_true, 
      vector<StructuredExample> &exs_false,
      double&true_pos, 
      double&false_pos, 
      double&true_neg, 
      double&false_neg,
      vector<StructuredExample> &correct_exs_true, 
      vector<StructuredExample> &correct_exs_false);        
    void split_props(const vector< StructuredExample >& examples,
		     double&true_pos, double&false_pos, double&true_neg, double&false_neg);
    double shannon_gain(const vector< StructuredExample >& examples);
    double structural_gain(
	double correct_true_ratio, double correct_false_ratio,
	vector<StructuredExample>&correct_exs_true, vector<StructuredExample>&correct_exs_false,
	const PCAPose&pcaPose, ostringstream&oss,
	Mat&pose_mu_true, Mat&pose_cov_true,
	Mat&pose_mu_false, Mat&pose_cov_false);	
    bool predict(const StructuredWindow& ex) const;
    RandomFeature(const vector<StructuredExample>&examples);
    
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::RandomFeature&);
  };
  
  // 
  class RandomHoughFeature : public RandomFeature
  {
  protected:
    double h0;
    InformationGain InfoGain;
    double p_correct;
    double correct_true, correct_false;
    int N_true, N_false;
    // statistics of the location (observed variables in the Hough transform)
    HoughVote vote_location;
    HoughVote vote_pose;
    
    // performance statistics (for debug)
    mutable std::atomic<int> times_voted_true, times_voted_false;
    
    RandomHoughFeature();
    
  public:
    Mat get_sigma_inv_true() const;
    Mat get_sigma_inv_false() const;
    Mat get_mu_true() const;
    Mat get_mu_false() const;
    Mat get_cov_true() const;
    Mat get_cov_false() const;
    int get_N_true() const;
    int get_N_false() const;
    double get_N_true_pos() const;
    double get_N_true_neg() const;
    double get_N_false_pos() const;
    double get_N_false_neg() const;
    string get_uuid() const;
    VoteResult vote(HoughOutputSpace&output,
		  const StructuredWindow&swin,const PCAPose&pose) const;	  
    InformationGain info_gain(
      vector< StructuredExample >& examples, 
      vector<StructuredExample> &exs_true, 
      vector<StructuredExample> &exs_false,
      const PCAPose&pcaPose);
    RandomHoughFeature(vector<StructuredExample>&examples);
    void print_voting_history() const;
    
    void log_kernels() const;
    friend ostream& operator << (ostream&, const RandomHoughFeature&rf);
    friend void write(cv::FileStorage&, std::string&, const std::unique_ptr<deformable_depth::RandomHoughFeature>&);
    friend void read(const cv::FileNode&, std::unique_ptr<deformable_depth::RandomHoughFeature>&, std::unique_ptr<deformable_depth::RandomHoughFeature>);
  };
  
  void read(const cv::FileNode&, std::unique_ptr<deformable_depth::RandomHoughFeature>&, std::unique_ptr<deformable_depth::RandomHoughFeature>);
  void write(cv::FileStorage&, std::string&, const std::unique_ptr<deformable_depth::RandomHoughFeature>&);
  ostream& operator << (ostream&, const RandomHoughFeature&rf);    

  // utility functions
  vector<StructuredExample> extract_gt_features(
    FeatureExtractionModel&feature_extractor,vector<StructuredExample>&all_feats);
  vector<StructuredExample> extract_features(FeatureExtractionModel&feature_extractor,
					     vector< shared_ptr< MetaData > >& training_set);
  vector<StructuredExample> extract_features(vector< shared_ptr< MetaData > >& training_set);
  vector<StructuredWindow> extract_windows(
    FeatureExtractionModel&feature_extractor,shared_ptr< ImRGBZ >& image);  
}

#endif
