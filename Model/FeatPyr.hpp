/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_FEAT_PYR
#define DD_FEAT_PYR

#include "ThreadCompat.hpp"
#include "util_mat.hpp"
#include "DepthFeatures.hpp"
#include "DiskArray.hpp"

namespace deformable_depth
{
  /**
   * Class allows sharing of features and images between
   *  detectors. Note, it is the client's responsablity
   *  to ensure the features and images are identical between
   *  all clients sharing an instance of FeatPyr.
   * 
   * ID Format: scale, tx, ty
   **/
  struct FeatPyr_Key
  {
    double scale;
    double tx, ty;
    
    FeatPyr_Key(double scale, double tx, double ty) : 
      scale(scale), tx(tx), ty(ty)
    {};
    
    // return a total order?
    bool operator< (const FeatPyr_Key&other) const
    {
      if(scale < other.scale)
	return true;
      else if(scale > other.scale)
	return false;

      if(tx < other.tx)
	return true;
      else if(tx > other.tx)
	return false;

      if(ty < other.ty)
	return true;
      else if(ty > other.ty)
	return false;    
      
      // total equality
      return false;
    };
  };
  
  class IFeatPyr
  {
  public:
    typedef FeatPyr_Key Key;
    
  public:
    virtual const ImRGBZ&getSrcImage() = 0;
    virtual bool addIm(const ImRGBZ&im,Key id) = 0;
    virtual DepthFeatComputer& computeFeat(const ImRGBZ&im,
				   function<DepthFeatComputer* (Size sz,int sbins)> build,
				   DepthFeatComputer&base,Key id) = 0;
    virtual shared_ptr<FeatIm> getFeat(Key id) = 0;
    virtual const ImRGBZ& getIm(Key id) = 0;    
  };
  
  class FauxFeatPyr : public IFeatPyr
  {
  protected:
    const ImRGBZ source_image;
    map<Key/*scale*/,const ImRGBZ> imPyr;
    
  public:
    FauxFeatPyr(const ImRGBZ source_image);
    virtual const ImRGBZ&getSrcImage();
    virtual bool addIm(const ImRGBZ&im,Key id);
    virtual DepthFeatComputer& computeFeat(const ImRGBZ&im,
				   function<DepthFeatComputer* (Size sz,int sbins)> build,
				   DepthFeatComputer&base,Key id);
    virtual shared_ptr<FeatIm> getFeat(Key id);
    virtual const ImRGBZ& getIm(Key id);       
  };
  
  /**
   * Feature Pyramid for a single image
   **/
  class FeatPyr : public IFeatPyr
  {
  protected:
    map<Key/*scale*/,const ImRGBZ> imPyr;
    map<Key/*scale*/,shared_ptr<FeatIm> > featPyr;
    map<Key,unique_ptr<DepthFeatComputer> > computers;
    mutex monitor;    
    const ImRGBZ source_image;
    
  public:
    FeatPyr(const ImRGBZ source_image);
    const ImRGBZ&getSrcImage();
    bool addIm(const ImRGBZ&im,Key id);
    DepthFeatComputer& computeFeat(const ImRGBZ&im,
				   function<DepthFeatComputer* (Size sz,int sbins)> build,
				   DepthFeatComputer&base,Key id);
    shared_ptr<FeatIm> getFeat(Key id);
    const ImRGBZ& getIm(Key id);
  };
}

#endif
