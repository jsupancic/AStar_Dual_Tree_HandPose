/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_RANDOM_FOREST
#define DD_RANDOM_FOREST

#include <opencv2/opencv.hpp>
#include <functional>
#include <memory>

namespace deformable_depth
{
  using namespace cv;
  using std::function;
  using std::shared_ptr;
  
  class RandomForest : public cv::ml::RTrees
  {
  public:
    float get_oob_error();
  };
}

#endif
