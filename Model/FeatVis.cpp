/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "FeatVis.hpp"

namespace deformable_depth
{
  /// SECTION FeatVis
  FeatVis::FeatVis(string source) : 
    source(source)
  {
    neg = Mat(0,0,DataType<Vec3b>::type);
    pos = Mat(0,0,DataType<Vec3b>::type);
  }
  
  FeatVis::FeatVis(string source, Mat pos, Mat neg) : 
    source(source)
  {
    setPos(pos);
    setNeg(neg);
  }
  
  const Mat& FeatVis::getNeg() const
  {
    return neg;
  }

  const Mat& FeatVis::getPos() const
  {
    return pos;
  }

  void FeatVis::setNeg(Mat neg) 
  {
    assert(neg.type() == DataType<Vec3b>::type);
    this->neg = neg;
  }

  void FeatVis::setPos(Mat pos) 
  {
    assert(pos.type() == DataType<Vec3b>::type);
    this->pos = pos;
  }
  
  string FeatVis::getSource() const
  {
    return source;
  }
}
