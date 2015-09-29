/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "RegEx.hpp"

namespace deformable_depth
{
  using namespace std;
  
  std::vector< std::string > regex_match(std::string str, boost::regex re)
  {
    vector<string> matches;
    
    boost::sregex_token_iterator iter(str.begin(),str.end(),re,0);
    boost::sregex_token_iterator end;
    
    for(; iter != end; ++iter)
    {
      matches.push_back(*iter);
    }
    
    return matches;
  }

  std::vector<std::string> regex_matches(std::string s, boost::regex re)
  {
    return deformable_depth::regex_match(s, re);
  }
}
