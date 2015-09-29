/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_SPHERE_VOLUMETRY
#define DD_SPHERE_VOLUMETRY

#include "util_depth.hpp"
#include "util_mat.hpp"
#include "Visualization.hpp"

namespace deformable_depth
{
  class SphericalOccupancyMap
  {
  protected:    
    CustomCamera camera;
    Mat om;    

    void drawCellInterp(Mat&yProj,const CustomCamera&camera, double xIter, double zIter, Scalar color, double xStride, double zStride) const;

  public:
    SphericalOccupancyMap(const ImRGBZ&im);
    Mat proj_z() const;
    Mat proj_x() const;
    Mat proj_y() const;
    Mat get_OM() const;
    Mat slice_z(float z_min, float z_max) const;
    Visualization vis() const;
    size_t xSize() const;
    size_t ySize() const;
    size_t zSize() const;
  };

  Vec3d vecOf(double azimuth, double altitude);
  void test_spherical_volumetry();
}

#endif
