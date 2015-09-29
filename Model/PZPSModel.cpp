/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "PZPSModel.hpp"
#include "ZSpanTree.hpp"
#include "SubTrees.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: PZPS Model
  ///
  PZPS::PZPS(Size gtSize, Size ISize)
  {
  }

  DetectionSet PZPS::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    TreeStats stats = treeStatsOfImage(im); 
    
    return DetectionSet{};
  }

  Mat PZPS::show(const string& title)
  {
    return Mat();
  }

  void PZPS::train(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
    return;
    for(auto && train_ex : training_set)
    {
      auto im = train_ex->load_im();
      TreeStats stats = treeStatsOfImage(*im); 
    }
  }

  bool PZPS::write(FileStorage& fs)
  {
      return deformable_depth::Model::write(fs);
  }
  
  ///
  /// SECTION: PZPS Model Builder
  ///
  Model* PZPS_Builder::build(Size gtSize, Size imSize) const
  {
    return new PZPS(gtSize, imSize);
  }

  string PZPS_Builder::name() const
  {
    return "PZPS_Builder";
  }

  PZPS_Builder::PZPS_Builder(double C, double area, shared_ptr< IHOGComputer_Factory > fc_factory, double minArea, double world_area_variance)
  {
  }
}
