/**
 * Copyright 2013: James Steven Supancic III
 **/

/**
 * Here I implement features which are computed using/on top of/ other features
 **/

#ifndef DD_META_FEATURES
#define DD_META_FEATURES

#include "DepthFeatures.hpp"
#include "util.hpp"
#include "vec.hpp"

namespace deformable_depth
{
  /**
   * Converts from OpenCV's internal HOG representation
   *  to one more suitable for convolution.
   **/
  class BlockCellRemapper : public DepthFeatComputer
  {
  protected:
    void map(std::vector< float >& from);
    void unmap(std::vector< double >& from);
    int ncells;
  protected:
    HOGComputer18x4_General computer;
  // RAII CTORs
  public:
  BlockCellRemapper(
      HOGComputer18x4_General::im_fun use_fun,
      Size win_size=Size(64, 128), 
      Size block_size=Size(16, 16), 
      Size block_stride=Size(8, 8), 
      Size cell_size=Size(8, 8));
  BlockCellRemapper(BlockCellRemapper&other);
  // virtual methods
  public:
    virtual int getNBins();
    virtual Size getBlockStride();
    virtual Size getCellSize();
    virtual Size getWinSize();
    virtual void compute(const ImRGBZ&im,vector<float>&feats);
    virtual size_t getDescriptorSize();
    virtual Mat show(const string&title,vector<double> feat);   
    virtual vector<FeatVis> show_planes(vector<double> feat);
    virtual int cellsPerBlock();
    virtual Size getBlockSize();
  };
  
  /**
   * Combines two forms of feature via concatination.
   * This function assumes the blocks == cells.
   **/
  class FeatureBinCombiner : public DepthFeatComputer
  {
  public:
    FeatureBinCombiner(
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
  protected:
    void normalize_weights();
    vector<shared_ptr<DepthFeatComputer> > computers;
    vector<double> weights; 
  };  
    
  class PCAFeature : public DepthFeatComputer
  {
  public:
    PCAFeature(
      Size win_size=Size(64, 128), 
      Size block_size=Size(16, 16), 
      Size block_stride=Size(8, 8), 
      Size cell_size=Size(8, 8));
    // abstract method implementations
  public:
    virtual int getNBins();
    virtual Size getBlockStride();
    virtual Size getCellSize();
    virtual Size getWinSize();
    virtual void compute(const ImRGBZ&im,vector<float>&feats);
    virtual size_t getDescriptorSize();
    virtual Mat show(const string&title,vector<double> feat);
    virtual Size getBlockSize();
    virtual int cellsPerBlock();    
  protected:
    typedef FeatureBinCombiner SubComputer;
    SubComputer unreduced_computer;
    static constexpr int FEAT_DIM = 42;
    Size win_size,block_size,block_stride,cell_size;
  protected:
    void ready_pca();
    void train_pca();
  };
  
  /**
   * The 18+4 HoG feature derived from the 18*4 HoG feature
   **/
  class HOGComputer18p4_General : public DepthFeatComputer
  {
  public:
    typedef im_select_fun im_fun;
    HOGComputer18p4_General(
		    im_fun use_fun,
		    Size win_size=Size(64, 128), 
		    Size block_size=Size(16, 16), 
		    Size block_stride=Size(8, 8), 
		    Size cell_size=Size(8, 8));  
    virtual ~HOGComputer18p4_General() {};
  // Implmeentation of HOGComputer
  public:
    virtual int getNBins() ;
    virtual Size getBlockStride() ;
    virtual Size getCellSize() ;
    virtual Size getWinSize() ;
    virtual void compute(const ImRGBZ&im,vector<float>&feats) ;
    virtual size_t getDescriptorSize();   
    virtual Mat show(const string&title,vector<double> feat);
    virtual vector<FeatVis> show_planes(vector<double> feat);
    virtual Size getBlockSize();
    virtual int cellsPerBlock();
    virtual string toString() const;    
  protected:
    vector< double > undecorate_feat(vector< double > feat);
    BlockCellRemapper hog18x4mapped;
  };  
  
  class HOGComputer18p4_Depth : public HOGComputer18p4_General
  {
  public:
    HOGComputer18p4_Depth(
	Size win_size=Size(64, 128), 
	Size block_size=Size(16, 16), 
	Size block_stride=Size(8, 8), 
	Size cell_size=Size(8, 8));       
  };
  
  class ComboComputer_Depth : public FeatureBinCombiner
  {
  public:
    ComboComputer_Depth(Size win_size = Size(64, 128), 
			Size block_size = Size(16, 16), 
			Size block_stride = Size(8, 8), 
			Size cell_size = Size(8, 8));
    virtual ~ComboComputer_Depth() {};
    virtual string toString() const;
  };  
  
  class ComboComputer_RGBPlusDepth : public FeatureBinCombiner
  {
  public:
    ComboComputer_RGBPlusDepth(Size win_size = Size(64, 128), 
			       Size block_size = Size(16, 16), 
			       Size block_stride = Size(8, 8), 
			       Size cell_size = Size(8, 8));
    virtual string toString() const;
  };
}

#ifndef WIN32
namespace deformable_depth
{ 
  typedef HOGComputer_Factory<NullFeatureComputer> NullFeat_FACT;
  typedef HOGComputer_Factory<DistInvarDepth> DistInvarDepth_FACT;
  //typedef HOGComputer_Factory<HOGComputer_Area> ZAREA_FACT; const ZAREA_FACT zarea_fact;
  typedef HOGComputer_Factory<HOGComputer18p4_Depth> ZHOG_FACT; //const ZHOG_FACT zhog_fact;
  //typedef HOGComputer_Factory<HistOfNormals> HNORM_FACT; const HNORM_FACT hnorm_fact; 
  typedef HOGComputer_Factory<FeatureBinCombiner> COMBO_FACT; 
  const COMBO_FACT combo_fact;
  typedef HOGComputer_Factory<ComboComputer_Depth> COMBO_FACT_DEPTH; 
  const COMBO_FACT_DEPTH combo_fact_depth;
  typedef HOGComputer_Factory<ComboComputer_RGBPlusDepth> COMBO_FACT_RGB_DEPTH; 
  const COMBO_FACT_RGB_DEPTH combo_fact_rgb_depth;
  
  // HOGComputer_Area or HOGComputer_RGB or HOGComputer_Depth FeatureBinCombiner OR PCAFeature
  typedef FeatureBinCombiner Default_Computer;   
  typedef HOGComputer_Factory<Default_Computer> Default_Computer_Factory;  
  const Default_Computer_Factory default_fact;  
}
#endif

#endif
