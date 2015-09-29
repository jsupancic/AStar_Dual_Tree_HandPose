/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "CSG.hpp"
#include <boost/multi_array/base.hpp>
#include "util_real.hpp"
#include "Log.hpp"
#include "util.hpp"
 
namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  deformable_depth::CSG_Workspace::CSG_Workspace(
    int xres, int yres, int zres, const map<string,Vec3d>&contains) : 
    volume(boost::extents[xres][yres][zres]),
    xres(xres), yres(yres), zres(zres)
  {
    x_min = y_min = z_min = +inf;
    x_max = y_max = z_max = -inf;
    
    for(auto & pt : contains)
    {
      double x = pt.second[0], y = pt.second[1], z = pt.second[2];
      x_min = std::min(x_min,x);
      y_min = std::min(y_min,y);
      z_min = std::min(z_min,z);
      x_max = std::max(x_max,x);
      y_max = std::max(y_max,y);
      z_max = std::max(z_max,z);
    }
    
    for(int zIter = 0; zIter < zres; zIter++)
      for(int yIter = 0; yIter < yres; yIter++)
	for(int xIter = 0; xIter < xres; xIter++)
	  volume[xIter][yIter][zIter] = Vec3b(0,0,0);
  }
  
  void CSG_Workspace::paint()
  {
    Mat vis(yres,xres,DataType<Vec3b>::type,Scalar::all(0));
    
    for(int yIter = 0; yIter < yres; yIter++)
      for(int xIter = 0; xIter < xres; xIter++)
	for(int zIter = 0; zIter < zres; zIter++)
	{
	  Vec3b color = volume[xIter][yIter][zIter];
	  if(color != Vec3b(0,0,0))
	    vis.at<Vec3b>(yIter,xIter) = color;
	}
	
    log_im("CSG_Workspace",vis);
  }
  
  // Bresenham's algorithm (see http://www.cb.uu.se/~cris/blog/index.php/archives/400)
  void CSG_Workspace::writeLine(Vec3d p1, Vec3d p2, int id)
  {
    Vec3d p = p1;
    Vec3d d = p2 - p1;
    // N = max(abs(d))
    double N = -inf;
    for(int iter = 0; iter < 3; ++iter)
      N = std::max(N,std::abs(d[iter]));
    Vec3d s = d/N;
    
    // compute the line
    put(p1,getColor(id));
    for(int iter = 1; iter <= N*static_cast<double>(xres)/(x_max-x_min); ++iter)
    {
      p += s;
      put(p,getColor(id));
    }
    put(p2,getColor(id));
  }
  
  void CSG_Workspace::dilate(int id)
  {
    Array3D new_volume = volume;
    Vec3b id_color = getColor(id);
    
    for(int y1 = 0; y1 < yres; y1++)
      for(int x1 = 0; x1 < xres; x1++)
	for(int z1 = 0; z1 < zres; z1++)
	  for(int y2 = std::max(y1 - 1,0); y2 <= std::min(y1+1,yres-1); y2++)
	    for(int x2 = std::max(x1 - 1,0); x2 <= std::min(x1+1,xres-1); x2++)
	      for(int z2 = std::max(z1 - 1,0); z2 <= std::min(z1 +1,zres-1); z2++)
	      {
		if(volume[x1][y1][z1] == id_color)
		  new_volume[x2][y2][z2] = id_color;
	      }
    
    volume = new_volume;
  }
  
  void CSG_Workspace::put(Vec3d raw_pos, Vec3b value)
  {
    int x = interpolate_linear(raw_pos[0],x_min,x_max,0,xres-1);
    int y = interpolate_linear(raw_pos[1],y_min,y_max,0,yres-1);
    int z = interpolate_linear(raw_pos[2],z_min,z_max,0,zres-1);
    volume[x][y][z] = value;
  }
}
