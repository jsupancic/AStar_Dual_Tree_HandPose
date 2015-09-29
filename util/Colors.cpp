/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Colors.hpp"
#include "util.hpp"
#include "RegEx.hpp"

#include <string>

namespace deformable_depth
{
  using namespace cv;
  using namespace std;

  Scalar toScalar(Vec3b v)
  {
    return Scalar(v[0],v[1],v[2]);
  }
  
  template<const char**name,Vec3b&def> 
  Scalar getColor()
  {
    static Scalar color = Scalar::all(-1);
    if(color[0] == -1)
    {
      if(g_params.has_key(*name))
      {
	string color_param = g_params.get_value(*name);
	std::vector<std::string> number_strings = deformable_depth::regex_matches(color_param,boost::regex("\\d+"));
	if(number_strings.size() == 3)
	{
	  int r = fromString<int>(number_strings.at(0));
	  int g = fromString<int>(number_strings.at(1));
	  int b = fromString<int>(number_strings.at(2));
	  color = Scalar(b,g,r);
	}
	else
	{
	  log_once(safe_printf("warning: bad color spec given % %",*name,color_param));
	  color = toScalar(def);
	}
      }
      else
      {
	color = toScalar(def);
      }
    }
    return color;
  }

  namespace Colors
  {
    char const*invalid_str = "COLOR_INVALID";
    cv::Scalar invalid()
    {    
      return getColor<&invalid_str,INVALID_COLOR>();
    }

    char const*inf_str = "COLOR_INF";
    cv::Scalar inf()
    {    
      return getColor<&inf_str,INVALID_COLOR>();
    }

    char const*ninf_str = "COLOR_NINF";
    cv::Scalar ninf()
    {    
      return getColor<&ninf_str,INVALID_COLOR>();
    }
  }

  cv::Vec3b  toVec3b(Scalar s)
  {
    return Vec3b(s[0],s[1],s[2]);
  }
}
