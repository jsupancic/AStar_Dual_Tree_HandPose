/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "FeatPyr.hpp"

#include <functional>

namespace deformable_depth
{
  using namespace std;
  
  bool FeatPyr::addIm(const ImRGBZ& im, Key sf)
  {
    unique_lock<mutex> exclusion(monitor);
    if(imPyr.find(sf) != imPyr.end())
      return true;    
    ImRGBZ im_here = im.resize(sf.scale);
    Rect roi = Rect(Point(sf.tx,sf.ty),Size(im_here.cols()-sf.tx,im_here.rows()-sf.ty));
    if(roi.area() < 2)
      return false;
    ImRGBZ im_crop = im_here(roi);
    require_equal<int>(im_crop.RGB.type(), DataType<Vec3b>::type);
    require_equal<int>(im_crop.Z.type(), DataType<float>::type);
    //require_gt<int>(im_crop.RGB.size().area(),0);
    imPyr.insert(pair<Key,const ImRGBZ>(sf, im_crop));
    return true;
  }  
  
  const ImRGBZ& FeatPyr::getIm(Key sf)
  {
    unique_lock<mutex> exclusion(monitor);
    const ImRGBZ&im = imPyr.at(sf);
    //require_equal<int>(im.RGB.type(), DataType<Vec3b>::type);
    //require_equal<int>(im.Z.type(), DataType<float>::type);
    return im;
  }
  
  DepthFeatComputer& FeatPyr::computeFeat(
    const ImRGBZ& im, function<DepthFeatComputer* (Size sz,int sbins)> build,
    DepthFeatComputer&base, Key sf)
  {
    unique_lock<mutex> exclusion(monitor);
    // return from cache
    if(computers.find(sf) != computers.end())
    {
      return *computers.at(sf);
    }
    // else recopmute
    
    // crop image to multiple of cell size.
    auto im_crop = base.cropToCells(im);    
    // create a separate base instance for this scale
    computers[sf] = unique_ptr<DepthFeatComputer>(
      build(im_crop->RGB.size(),base.getCellSize().width));
    //dynamic_cast<OccSliceFeature*>(&*computers.at(sf))->setRanges(min_depth,max_depth);
    assert(computers.at(sf)->cellsPerBlock() == 1);
    // get HOG from image
    featPyr[sf] = shared_ptr<FeatIm>(new FeatIm());
    computers.at(sf)->compute(*im_crop,*featPyr[sf]);
    
    return *computers.at(sf);
  }
  
  shared_ptr<FeatIm> FeatPyr::getFeat(Key sf)
  {
    unique_lock<mutex> exclusion(monitor);
    return featPyr.at(sf);
  }
  
  FeatPyr::FeatPyr(const ImRGBZ source_image) : source_image(source_image)
  {

  }

  const ImRGBZ& FeatPyr::getSrcImage()
  {
    return source_image;
  }
  
  ///
  /// Implements the partially functional Faux Image Pyramid
  ///
  FauxFeatPyr::FauxFeatPyr(const ImRGBZ source_image) : source_image(source_image)
  {
  }
  
  bool FauxFeatPyr::addIm(const ImRGBZ& im, Key id)
  {
    // nop
    return true;
  }

  DepthFeatComputer& FauxFeatPyr::computeFeat(
    const ImRGBZ& im, function<DepthFeatComputer* (Size sz,int sbins)> build, DepthFeatComputer& base, 
    Key id)
  {
    assert(false);
    throw std::logic_error("unimplemented");
  }

  const ImRGBZ& FauxFeatPyr::getIm(IFeatPyr::Key id)
  {
    imPyr.insert(pair<Key,const ImRGBZ>(id, source_image.resize(id.scale,true)));
    return imPyr.at(id);
  }

  shared_ptr< FeatIm > FauxFeatPyr::getFeat(IFeatPyr::Key id)
  {
    assert(false);
    throw std::logic_error("unimplemented");
  }

  const ImRGBZ& FauxFeatPyr::getSrcImage()
  {
    return source_image;
  }
}
