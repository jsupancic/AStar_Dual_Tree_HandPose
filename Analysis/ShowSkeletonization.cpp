/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ShowSkeletonization.hpp"
#include "InverseKinematics.hpp"

namespace deformable_depth
{
  void show_skeletonization(BaselineDetection&det,MetaData&frame)
  {
    // find all records within a threshold.
    LibHandRenderer* renderer = renderers::no_arm();
    //vector<PoseRegressionPoint> matches = ik_regress_thresh(*renderer,
//							    det,
//							    params::finger_correct_distance_threshold());

    // choose the best with NN

    // draw the skeletonization
  }
}

