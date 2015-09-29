/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_POSELET
#define DD_POSELET

#include "MetaData.hpp"
#include "Detection.hpp"

namespace deformable_depth
{
  namespace Poselet
  {    
    struct Procrustean2D
    {
      double max_se;
      double min_ssd;
      double tx, ty, s;
      
      Procrustean2D() : max_se(qnan), min_ssd(qnan), tx(qnan),ty(qnan), s(qnan) {};

      inline bool operator< (const Procrustean2D&other) const
      {
	return min_ssd < other.min_ssd;
      }
    };
    
    vector<Point2d> keypoints(MetaData&metadata);
    double min_dist_procrustian(MetaData & dst/*vary param*/, 
				DetectorResult & src/*fixed param*/);
    Procrustean2D min_dist(const vector<Point2d>&xs/*dst*/,
		    const vector<Point2d>&ys/*src*/,vector<double> weights);
    double min_dist_simple(const vector<Vec3d>&xs/*dst*/,
		    const vector<Vec3d>&ys/*src*/);
    Procrustean2D min_dist(const map<string,Vec3d>& xs, const map<string,Vec3d>&ys,bool clamp = true);
    Procrustean2D min_dist(const map<string,AnnotationBoundingBox>&xs,const map<string,AnnotationBoundingBox>&ys,bool clamp = true);
    vector<double> make_weights(string partname, 
			    shared_ptr<MetaData> source,
			    vector<Point2d>&kp_src);
  }
  
  void poselet();
}

#endif
