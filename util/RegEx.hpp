/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_REGEX
#define DD_REGEX

#include <vector>
#include <string>
#include <boost/regex.hpp>

namespace deformable_depth
{
  // match numbers: "\\d+"
  std::vector<std::string> regex_match(std::string, boost::regex re);
  std::vector<std::string> regex_matches(std::string, boost::regex re);
}

#endif

