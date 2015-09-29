/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_DEPTH_FEATURES
#define DD_DEPTH_FEATURES

#define use_speed_ 0
#include <opencv2/opencv.hpp>

#include "util_mat.hpp"
#include "params.hpp"
#include <memory>
#include <functional>
#include "FeatVis.hpp"

/**
 * This function is complicated by the fact that OpenCV uses 
 * 	a strange format for it's HOG feature. I essentially use
 * 	the same thing but elsewhere find it awkward so I have to define
 * 	remaper classes. 
 */
namespace deformable_depth
{
  using namespace cv;
  using std::shared_ptr;
  
  shared_ptr<const ImRGBZ> cropToCells(const ImRGBZ& im,int cellx, int celly);
    
  // These features have to be stored linearly in memory. 
  // I use the convention barrowed from OpenCV (which is stupid)
  // If using a for loop, from outer loop to inner loop we iterate as:
  // blockX blockY cellN bin
  class DepthFeatComputer
  {
    // abstract methods
  public:
    virtual int getNBins() = 0;
    virtual Size getBlockStride() = 0;
    virtual Size getCellSize() = 0;
    virtual Size getWinSize() = 0;
    virtual void compute(const ImRGBZ&im,vector<float>&feats) = 0;
    virtual size_t getDescriptorSize() = 0;
    virtual Mat show(const string&title,vector<double> feat) = 0;
    virtual vector<FeatVis> show_planes(vector<double> feat);
    virtual Size getBlockSize() = 0;
    virtual int cellsPerBlock() = 0;
    // concrete methods
  public:
    inline int getIndex(int blockX, int blockY, int cellN, int bin)
    {
      return 
	blockX*(blocks_y()*cellsPerBlock()*getNBins()) + 
	blockY*(cellsPerBlock()*getNBins()) + 
	cellN*getNBins() + 
	bin;
    }
    shared_ptr<const ImRGBZ> cropToCells(const ImRGBZ&im);
    virtual ~DepthFeatComputer() {};
    virtual int blocks_x();
    virtual int blocks_y();
    virtual string toString() const;
  };
  
  class NullFeatureComputer : public DepthFeatComputer
  {
  protected:
    Size win_size;
    Size block_size;
    Size block_stride;
    Size cell_size;
    
  public:
    NullFeatureComputer(
    		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8));
    
  public:
    virtual int getNBins();
    virtual Size getBlockStride();
    virtual Size getCellSize();
    virtual Size getWinSize();
    virtual void compute(const ImRGBZ&im,vector<float>&feats);
    virtual size_t getDescriptorSize();
    virtual Mat show(const string&title,vector<double> feat);
    virtual vector<FeatVis> show_planes(vector<double> feat);
    virtual Size getBlockSize();
    virtual int cellsPerBlock();    
    virtual string toString() const;
  };
    
  class HOGComputer_RGB : private cv::HOGDescriptor, public DepthFeatComputer
  {
  public:
    // RAII CTORS
    HOGComputer_RGB(
		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8), 
		    int nbins=9) : 
		    cv::HOGDescriptor(
		      win_size,
		      block_size,
		      block_stride,
		      cell_size,
		      nbins){};
    HOGComputer_RGB(HOGComputer_RGB&other) : cv::HOGDescriptor(other){};
  // Implmeentation of HOGComputer
  public:
    virtual int getNBins() ;
    virtual Size getBlockStride() ;
    virtual Size getCellSize() ;
    virtual Size getWinSize() ;
    virtual void compute(const ImRGBZ&im,vector<float>&feats) ;
    virtual size_t getDescriptorSize();
    virtual Mat show(const string&title,vector<double> feat);
    virtual Size getBlockSize();
    virtual string toString() const;
    virtual int cellsPerBlock();
  };
  
  typedef std::function<const Mat& (const ImRGBZ&)> im_select_fun; 
  
  class HOGComputer18x4_General : public DepthFeatComputer
  {
  public:
    typedef im_select_fun im_fun; 
    // RAII CTORS
    HOGComputer18x4_General(
		    im_fun use_fun,
		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8));
    HOGComputer18x4_General(HOGComputer18x4_General&other);
  private:
    Size win_size;
    Size block_size;
    Size block_stride;
    Size cell_size;
    int nbins;
    constexpr static bool contrast_sensitive = true;
    im_fun use_fun;    
  // Implmeentation of HOGComputer
  public:
    virtual int getNBins() ;
    virtual Size getBlockStride() ;
    virtual Size getCellSize() ;
    virtual Size getWinSize() ;
    virtual void compute(const ImRGBZ&im,vector<float>&feats) ;
    virtual size_t getDescriptorSize();   
    virtual Mat show(const string&title,vector<double> feat);
    virtual Size getBlockSize();
    virtual int cellsPerBlock();
    virtual string toString() const;
    virtual vector<FeatVis> show_planes(vector<double> feat);
  };
    
  class HistOfNormals : public DepthFeatComputer
  {
  private:
    Size cell_size;
    Size win_size;
    static constexpr int theta_bins = 10;
    static constexpr int phi_bins = 10;
  public:
    HistOfNormals(
		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8));
  // Implementation of DepthFeatComputer
  public:
    virtual int getNBins() ;
    virtual Size getBlockStride() ;
    virtual Size getCellSize() ;
    virtual Size getWinSize() ;
    virtual void compute(const ImRGBZ&im,vector<float>&feats) ;
    virtual size_t getDescriptorSize();   
    virtual Mat show(const string&title,vector<double> feat);
    virtual Size getBlockSize();
    virtual int cellsPerBlock();
    virtual string toString() const;
  };
  
  class HOGComputer_Area : public DepthFeatComputer
  {
  protected: // settings
    Size win_size;
    Size block_size;
    Size block_stride;
    Size cell_size;
    int nbins;
  protected: // hyperparmaeters
#ifdef DD_CXX11  
    constexpr static int BIN_COUNT = 8;
    constexpr static float MAX_AREA  = 50; // in cm^2, tuned hyperparameter
#endif
  public: // RAII CTORS
    HOGComputer_Area(
		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8));
    HOGComputer_Area(HOGComputer_Area&other);
    
    static double DistInvarDepth(double depth, const Camera&camera, double&max_area_out);
    static Mat DistInvarientDepths(const ImRGBZ&im,double&max_value);
    
  public: // Implmeentation of HOGComputer
    virtual int getNBins() ;
    virtual Size getBlockStride() ;
    virtual Size getCellSize() ;
    virtual Size getWinSize() ;
    virtual void compute(const ImRGBZ&im,vector<float>&feats) ;
    virtual size_t getDescriptorSize();   
    virtual Mat show(const string&title,vector<double> feat);
    virtual int cellsPerBlock();
    virtual Size getBlockSize();
    virtual string toString() const;
    virtual vector<FeatVis> show_planes(vector<double> feat);
  protected:
    // two different show implementations.
    virtual Mat show_max(const string&title,vector<double> feat);
    virtual Mat show_sample(const string&title,vector<double> feat);
  };
  Mat DistInvarientDepth(const ImRGBZ&im);
    
  class DistInvarDepth : public DepthFeatComputer
  {
  protected:
    Size win_size;
    
  public:
    DistInvarDepth(Size win_size = Size(64, 128), 
			Size block_size = Size(16, 16), 
			Size block_stride = Size(8, 8), 
			Size cell_size = Size(8, 8));
    virtual int getNBins();
    virtual Size getBlockStride();
    virtual Size getCellSize();
    virtual Size getWinSize();
    virtual void compute(const ImRGBZ&im,vector<float>&feats);
    virtual size_t getDescriptorSize();
    virtual Mat show(const string&title,vector<double> feat);
    virtual vector<FeatVis> show_planes(vector<double> feat);
    virtual Size getBlockSize();
    virtual int cellsPerBlock();
  };
  
  // here we control the type of HOG feature extracter craeted
  struct IHOGComputer_Factory
  {
    virtual DepthFeatComputer* build(Size winSize, int s_cell) const = 0;
  };
  template<typename CT>
  struct HOGComputer_Factory : public IHOGComputer_Factory
  {
    DepthFeatComputer* build(Size winSize, int s_cell)  const
    {
      return new 
	CT(winSize,Size(2*s_cell,2*s_cell),Size(s_cell,s_cell),Size(s_cell,s_cell));
    }
  };
  // adapter factory
  // FT: Factory we use
  // CT: type we produce
  template</*typename FT,*/typename CT>
  struct AdapterFactory : public IHOGComputer_Factory
  {
    AdapterFactory(shared_ptr<IHOGComputer_Factory> subordinate_factory) : 
    subordinate_factory(subordinate_factory){};
    shared_ptr<IHOGComputer_Factory> subordinate_factory;
    DepthFeatComputer* build(Size winSize, int s_cell)  const
    {
      return new CT(shared_ptr<DepthFeatComputer>(subordinate_factory->build(winSize,s_cell)));
    }    
  };
}

#include "MetaFeatures.hpp"

#endif
