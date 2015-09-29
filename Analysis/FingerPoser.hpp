/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_FINGER_POSER
#define DD_FINGER_POSER

#include "BaselineDetection.hpp"

namespace deformable_depth
{
  /**
   * class takes a finger detection and (Tries to) correct the pose
   **/
  class FingerPoser
  {
  protected:    
    
  public:
    Mat segment_flood(Rect bb, const ImRGBZ&im,map<string,AnnotationBoundingBox>&poss);
    Mat segment_bounds(Rect bb, const ImRGBZ&im, float z_min = qnan, float z_max = qnan);
    Mat segment(Rect bb,const ImRGBZ&im);
    bool is_freespace(Rect bb, const ImRGBZ&im);
    virtual BaselineDetection pose(const BaselineDetection&src,
				    ImRGBZ&im) = 0;
  };
  
  class VoronoiPoser : public FingerPoser
  {
  public:
    virtual BaselineDetection pose(const BaselineDetection&src,
				    ImRGBZ&im);
  };

  class DumbPoser : public FingerPoser
  {
  public:
    virtual BaselineDetection pose(const BaselineDetection&src,ImRGBZ&im) override;
  };
}

#endif
