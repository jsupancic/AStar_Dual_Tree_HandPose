/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_SCANNING_WINDOW
#define DD_SCANNING_WINDOW

#include <map>
#include <string>
#include "HeuristicTemplates.hpp"
#include "Model.hpp"

namespace deformable_depth
{
  using std::map;
  using std::string;

  static constexpr int ROTS_PER_WINDOW = 20;
  static constexpr int ROTS_TOTAL = 50;

  ///
  /// SECTION: Training
  ///
  struct ExtractedTemplates
  {
  public:
    map<string,string> sources;
    map<string,NNTemplateType> allTemplates;
    map<string,map<string,AnnotationBoundingBox> > parts_per_ex;
  };

  vector<AnnotationBoundingBox> metric_positive(MetaData&ex);
  ExtractedTemplates extract_templates(vector< shared_ptr< MetaData > >& training_set, 
				       Model::TrainParams train_params);
  Size MetricSize(string video_name);

  /// 
  /// SECTION: Testing
  ///
  void set_det_positions
  (const NNTemplateType&Tex,string pose,
   RotatedRect orig_bb, // X BB
   map<string,AnnotationBoundingBox> & parts,
   DetectorResult&det,float depth);
  void set_det_positions2
  (RotatedRect gtBB, shared_ptr<MetaData>&datum,string pose,
   RotatedRect orig_bb, // X BB
   map<string,AnnotationBoundingBox> & parts,
   DetectorResult&det,float depth);

  DetectionSet enumerate_windows(const ImRGBZ&im,DetectionFilter filter);
}

#endif

