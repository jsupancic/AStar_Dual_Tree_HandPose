/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_ANALYZE
#define DD_ANALYZE

#include <opencv2/opencv.hpp>
#include <string>
#include <util_rect.hpp>
#include <iostream>
#include <fstream>
#include "Plot.hpp"

namespace deformable_depth
{
  using cv::Rect_;
  using std::string;
  using std::ofstream;
  
  class SavedResult
  {
  public:
    SavedResult(Rect_<double> bb);
    
  protected:
    Rect_<double> bb;
  };
  
  map<string,PerformanceRecord > analyze();
  
  // generate PR curves
  void analyze_video();
  // generate PR plots for the anytime experiments
  void analyze_anytime();
  void analyze_egocentric();
  void export_responces();
  
  void regress_finger_conf();
}

#endif
