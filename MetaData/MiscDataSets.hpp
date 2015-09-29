/**
 * Copyright 2014: James Steven Supancic III
 **/ 

#ifndef DD_MISC_DATA_SETS
#define DD_MISC_DATA_SETS

#include "MetaData.hpp"
#include "Aggregate.hpp"

namespace deformable_depth
{
  shared_ptr<MetaDataAggregate> form_Metadatum(Mat RGB, Mat D, Mat UV,string im_name,string md_name);
  vector<shared_ptr<MetaData> > egocentric_dir_data();
}

#endif

