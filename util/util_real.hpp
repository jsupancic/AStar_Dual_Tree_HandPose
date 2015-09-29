/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_REAL
#define DD_REAL

#include <limits>

namespace deformable_depth
{
  using namespace std;
  
  static float inf = numeric_limits<float>::infinity();
  static double qnan = numeric_limits<double>::quiet_NaN();
  
  template<typename T>
  T clamp(T min, T value, T max)
  {
    if(value > max)
      return max;
    if(value < min)
      return min;
    return value;
  }
  
  void memnorm(float*ptr,int count);
  float sigmoid(float t);
  
  #ifdef DD_CXX11
  constexpr int MAX_INT = numeric_limits<int>::max();
  #endif
  
  int binOf(float x, float y);
  double sign(double value);
  double rad2deg(double rad);
  double deg2rad(double deg);
  
  float sample_in_range(float min, float max);
  double interpolate_linear_prop(double x, double in0, double in1, double out0, double out1);
  double interpolate_linear(double x, 
			    double in_min, double in_max, 
			    double out_min, double out_max);
}

#endif
