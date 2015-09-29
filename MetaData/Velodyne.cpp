/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Velodyne.hpp"
#include "util_real.hpp"
#include <fstream>

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  VelodyneData::VelodyneData(std::string filename)
  {
    ifstream ifdata(filename,ios::binary|ios::in);
    //cout << "VelodyneData::VelodyneData " << filename << endl;
    while(ifdata)
    {
      float x,y,z,r;
      x = y = z = r = qnan;
      ifdata.read(reinterpret_cast<char*>(&x),sizeof(float));
      ifdata.read(reinterpret_cast<char*>(&y),sizeof(float));
      ifdata.read(reinterpret_cast<char*>(&z),sizeof(float));
      ifdata.read(reinterpret_cast<char*>(&r),sizeof(float));
      if(isnan(x) || isnan(y) || isnan(z) || isnan(r))
	break;
      //cout << "got : " << x << " " << y << " " << z << " " << r << endl;
      points.push_back(Vec3d(x,y,z));
      reflectances.push_back(r);
    }
    
    //cout << "Loading " << points.size() << " Velodyne Points" << endl;
  }
  
  vector< Vec3d >& VelodyneData::getPoints()
  {
    return points;
  }

  vector< double >& VelodyneData::getReflectances()
  {
    return reflectances;
  }
}
