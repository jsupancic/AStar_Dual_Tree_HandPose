/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_CONVEXITY_DEFECTS
#define DD_CONVEXITY_DEFECTS
#include "Model.hpp"
#include "FingerPoser.hpp"

namespace deformable_depth
{  
  /**
   * Use OpenCV's Convexity Defects function to detect the fingers 
   * in a given bounding box.
   **/
  typedef unordered_set<Point,
	    function<unsigned long(Point)>,
	    function<bool(const Point&,const Point&)> > ConvexExtrema;  
  
  class FingerConvexityDefectsPoser : public FingerPoser
  {
  protected:
    ConvexExtrema analyze_convexity(Rect bb,ImRGBZ&im);
    virtual BaselineDetection pose_cover_fingers(const BaselineDetection&src,ImRGBZ&im);
    virtual BaselineDetection pose_inform_error_correction(const BaselineDetection&src,ImRGBZ&im);

  public:
    virtual BaselineDetection pose(const BaselineDetection&src,ImRGBZ&im);
    bool is_hard_example(MetaData&metadata);
  };
}

#endif

