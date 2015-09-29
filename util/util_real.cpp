/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "util_real.hpp"
#include "params.hpp"

namespace deformable_depth
{
  void memnorm(float*ptr,int count)
  {
    float ss = 0;
    for(int iter = 0; iter < count; iter++)
      ss += ptr[iter]*ptr[iter];
    ss = ::sqrt(ss);
    
    if(ss == 0)
      return;
    
    for(int iter = 0; iter < count; iter++)
      ptr[iter] /= ss;
  }
    
  float sigmoid(float t)
  {
    return 1/(1+::exp(t));
  }
  
  int binOf(float x, float y)
  {
    float best_dot =  -numeric_limits<float>::infinity();
    int bestBin = 0;
    for(int oIter = 0; oIter < params::ORI_BINS; oIter++)
    {
      float dot = params::uu[oIter]*x + params::vv[oIter]*y;
      if(dot > best_dot)
      {
	best_dot = dot;
	bestBin = oIter;
      }
    }
    
    return bestBin;
  }
  
  double rad2deg(double rad)
  {
    return rad*180.0/params::PI;
  }
  
  double deg2rad(double deg)
  {
    return deg*params::PI/180.0;
  }

  double interpolate_linear_prop(double x, double in_1, double in_2, double out_1, double out_2)
  {  
    double in_min = std::min(in_1,in_2);
    double in_max = std::max(in_1,in_2);
    double out_min= std::min(out_1,out_2);
    double out_max= std::max(out_1,out_2);
    return interpolate_linear(x, in_min, in_max, out_min, out_max);
  }

  double interpolate_linear(
    double x, 
    double in_min, double in_max, 
    double out_min, double out_max)
  {
    double x_standardized = (x - in_min)/(in_max - in_min);
    return clamp<double>(out_min,x_standardized * (out_max - out_min) + out_min,out_max);
  }
}
