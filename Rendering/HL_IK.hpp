/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_HL_IK
#define DD_HL_IK

#include "MetaData.hpp"

namespace deformable_depth
{
  // Implements high level inverse kinematics functions
  struct IKSyntherError
  {
    double pose_error;
    double template_error;
  };

  // returns how well 
  IKSyntherError synther_error(MetaData&metadata);  
  shared_ptr<MetaData> closest_exemplar_ik(MetaData&metadata);
}

#endif
