/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_OCCREASON_FEAT
#define DD_OCCREASON_FEAT

#include "MetaFeatures.hpp"

namespace deformable_depth
{
  /*
   * Represents the depths by cell in an image
   **/
  class CellDepths
  {
  public:
    // 5 regular (inside volume), +2 special (FG/BG)
    static constexpr int METRIC_DEPTH_BINS = 5 + 2;
    static constexpr float BETA_HISTOGRAM = 5;
    static constexpr float BETA_OCC = 25;
    static constexpr int BIN_OCC = 0;
    static constexpr int BIN_BG = 1;
    
    Mat upper_manifold;
    vector<vector<vector<float> > > raw_cell_depths; // y,x, list of depths
    
    CellDepths operator()( const Rect& roi ) const;
    CellDepths(const ImRGBZ&im,int nx, int ny);
    vector<double> depth_histogram(
      int row, int col, float min_depth, float max_depth) const;
    
  protected:
    CellDepths();
  };
  
  /**
   * Given a specific depth range, this thing
   * adds occlusion and bg features to each cell
   **/
  class OccSliceFeature : public DepthFeatComputer
  {
  public:
    OccSliceFeature(shared_ptr<DepthFeatComputer> subordinate);
    typedef function<void (const CellDepths&cell_depths,float&min_depth,float&max_depth)> DepthFn;
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
    void mark_context(Mat&vis,vector<double> feat,bool lrflip = false);
  public:
    void setRanges(float min_depth,float max_depth);
    void setDepthFn(DepthFn depthFn);
    vector<float > decorate_feat(const vector<float>&raw_pos,const CellDepths&cellDepths,
      float min_depth, float max_depth);
    vector<double> undecorate_feat(const vector<double>&orig);
    double occlusion(const vector<double>&orig);
    double background(const vector<double>&orig);
    double real(const vector<double>&orig);
    
    static constexpr int DECORATION_LENGTH = CellDepths::METRIC_DEPTH_BINS;
  protected:    
    mutable shared_ptr<DepthFeatComputer> subordinate;
    CellDepths mkCellDepths(const ImRGBZ&im) const;
    DepthFn comp_depths;
    int num_cells() ;
    friend class OccAwareLinearModel;
  };  
}

#endif
