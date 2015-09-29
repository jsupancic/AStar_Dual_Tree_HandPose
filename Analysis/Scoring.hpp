/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_SCORING
#define DD_SCORING

#include "Plot.hpp"

namespace deformable_depth
{
  PerformanceRecord analyze_pxc(string person = "Vivian");
  PerformanceRecord analyze_pbm(string det_filename, string person = "Vivian");
  PerformanceRecord score_video(BaseLineTest test);
}

#endif
