/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_EVAL
#define DD_EVAL

#include <vector>
#include <util.hpp>
#include <memory>

#include "Detector.hpp"

#include "PXCSupport.hpp"
#include "GlobalMixture.hpp"
#include "SRF_Model.hpp"
#include "HandModel.hpp"
#include "Entanglement.hpp"
#include "ExternalModel.hpp"
#include "StackedModel.hpp"
#include "PZPSModel.hpp"
#include "ApxNN.hpp"

/**
 * Functions which help quantitatively evaluate systems and models.
 **/
namespace deformable_depth
{
  constexpr bool g_supress_feature = false;
  
  vector<string> real_dirs();
  vector<string> default_test_dirs();
  string default_train_dir();
  vector<string> default_train_dirs();
  vector<shared_ptr<MetaData> > default_train_data();
  vector<shared_ptr<MetaData> > default_test_data();
  vector<shared_ptr<MetaData> > load_dirs(vector<string> dirs, bool only_valid = true);
  
  /// SECTION: Use a validation set to tune the AREA and C parameters
  class Model_HyperParam_Setting
  {
  private:
    // key
    double C;
    double AREA;
    // value
    Scores score;
    
  public:
    Model_HyperParam_Setting(double C, double AREA, Scores score);
    void print();
    bool operator<(const Model_HyperParam_Setting&other) const;
    Scores& getScore();
  };  
  
  // represents the output for a frame given by our model
  struct BestDetection
  {
    string filename;
    bool correct;
    DetectorResult detection;
  };  
  void write(FileStorage&fs, string&, const BestDetection& bestDet);
  void read(const FileNode&, BestDetection&, BestDetection);
    
  DetectionSet detect_default(
			      Model&model,shared_ptr<MetaData>&metadata,shared_ptr<ImRGBZ> imRGBZ,
    string filename);  
  DetectionSet efficent_detection(
    Model&model,MetaData&metadata,shared_ptr<ImRGBZ> imRGBZ,
    string filename, bool allow_reject = false);
  void test_model_oni_video(Model&model);
  struct TestModelShowResult
  {
    DetectorResult closest;
  };
  TestModelShowResult test_model_show(const Model&model,MetaData&metadata,DetectionSet&dets,string prefix = "TopDetections");
  vector<Scores> eval_model(int argc, char**argv, 
			    const Model_Builder&model_builder);
  void tune_model(int argc, char**argv);
  void seed_data(vector<shared_ptr<MetaData> > train_examples,
		 Rect&BB,shared_ptr<MetaData>&metadata);
  void train_model_one(Model&model,vector<shared_ptr<MetaData> > &examples);
  
  // for pose re-construction and validation
  void pose_reconstruction(int argc, char**argv);
}

#endif
