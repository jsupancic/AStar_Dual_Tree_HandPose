/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "RandomForest.hpp"
#include "util.hpp"

namespace deformable_depth
{
  float RandomForest::get_oob_error()
  {
    return qnan; //oob_error;
  }
}
