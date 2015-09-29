/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_TEST_MODEL
#define DD_TEST_MODEL

#include "Model.hpp"

namespace deformable_depth
{
  typedef function<void (string,DetectorResult&best,DetectorResult&closest,bool)> DetectionOutFn;
  Mat test_model_show_exemplar(DetectorResult best_det);
  
  void test_kitti(MetaData&data,shared_ptr<Model>&model,bool validation);
  vector<string> test_video_filenames();
  void test_model_oni_video(Model&model);
  void test_model_images(shared_ptr<Model> model);
  void test_model_one_example
      (shared_ptr<MetaData> metadata,
      Model&model,
      Scores&scores,
      map<string,Scores>&scoresByPose,
      DetectionOutFn write_best_det
      );
  void test_model(Model&model,string test_dir, 
		  Scores&scores, 
		  map<string,Scores>&scoresByPose,
		  DetectionOutFn write_best_det);
}

#endif
