/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_TREEDECORATORS
#define DD_TREEDECORATORS

#include "util.hpp"
#include "util_mat.hpp"
#include "opencv2/opencv.hpp"
#include "ZSpanTree.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  struct TreeStats;
  
  struct StatsUpdateArgs
  {
    const ImRGBZ&im;
    bool true_edge;
    bool oob_edge;
    float z_here,  z_nb, z_far;
    Direction dir;
    Vec2i here, neighbour;
    StatsUpdateArgs(const ImRGBZ&im,
	   bool true_edge,
	   bool oob_edge,
	   float z_here,  float z_nb, float z_far,
	   Direction dir,
	   Vec2i here, Vec2i neighbour) : 
	   im(im), true_edge(true_edge), oob_edge(oob_edge),
	   z_here(z_here),z_nb(z_nb), z_far(z_far),dir(dir),here(here), neighbour(neighbour)
	   {};
  };
    
  class TreeStatsDecorator
  {
  public:
    virtual void update_base(const StatsUpdateArgs args) = 0;
    virtual void update_recur(const StatsUpdateArgs args) = 0;
  };  
    
  /**
   * Use the color histogram for skin detection...
   **/
  class SkinDetectorDecorator : public TreeStatsDecorator
  {
  public:
    vector<Mat> meanProbLikes;
    Mat likelihood;
    TreeStats&ts_for_area;
  public:
    SkinDetectorDecorator(Mat RGB,TreeStats&ts_for_area);
    void show();
  public:
    // from parent class
    virtual void update_base(const StatsUpdateArgs args);
    virtual void update_recur(const StatsUpdateArgs args);    
  };  
  
  /**
   * Abstract decorator which computes the area of the tree
   * matching the predicate
   **/
  class FilteredAreaDecorator : public TreeStatsDecorator
  {
  public:
    enum AreaType
    {
      PIXEL, WORLD
    };
    AreaType areaType;
    vector<Mat> matching_areas;
    virtual void update_base(const StatsUpdateArgs args);
    virtual void update_recur(const StatsUpdateArgs args);
    virtual bool predicate(Vec2i pos) = 0;
    FilteredAreaDecorator(Mat&dt,AreaType areaType = WORLD);
    FilteredAreaDecorator();
  };
  
  /**
   * this is Not really a "feature" computer like the rest,
   * it's for figuring out which subtrees have a given label during training.
   **/
  class SupervisedDecorator : public FilteredAreaDecorator
  {
  public:
    Mat pos_mask;
  public: // virtual
    SupervisedDecorator(Mat&dt, Mat pos_mask);
    virtual bool predicate(Vec2i pos);
  };
  
  /**
   * Contains the bounding box for each sub-tree
   **/  
  class PixelAreaDecorator : public FilteredAreaDecorator
  {
  public:
    // bounding boxes
    vector<vector<vector<Rect> > > bbs;
  public:
    PixelAreaDecorator(Mat& dt);
    PixelAreaDecorator();
    virtual bool predicate(Vec2i pos);
    virtual void update_base(const StatsUpdateArgs args);
    virtual void update_recur(const StatsUpdateArgs args);
  };
  typedef PixelAreaDecorator BoundingBoxDecorator;
  void read(string filename,PixelAreaDecorator&paDec);
  void write(string filename,PixelAreaDecorator&paDec);
  
  class CentroidDecorator : public TreeStatsDecorator
  {
  public:
    vector<Mat> centroids; // Mat<Vec3d>
    TreeStats&ts_for_area;
    CentroidDecorator(const ImRGBZ&im,TreeStats&ts);
  public: // virtual
    virtual void update_base(const StatsUpdateArgs args);
    virtual void update_recur(const StatsUpdateArgs args);    
  public:
    Vec3d forward(Vec2i pos, Direction dir);
    Vec3d backward(Vec2i pos, Direction dir);
    Vec3d fbDelta(Vec2i pos, Direction dir);
    bool isFrontal(Vec2i pos, Direction dir);
    void show();
  };  
  
  class FaceAreaDecorator : public FilteredAreaDecorator
  {
  public:
    Mat facep;
    FaceAreaDecorator(const ImRGBZ&im,Mat&dt,GraphHash&treeMap);
    void show(std::vector< cv::Mat >& areas);
  public: // virtual
    virtual bool predicate(Vec2i pos);
  };  
  
  class EdgeTypeDecorator : public TreeStatsDecorator
  {
  public:
    Mat etLeft, etRight, etUp, etDown;
    EdgeTypeDecorator(Mat&dt,GraphHash&treeMap);
    Vec3i extrinsic_ET(int row, int col, Direction dir);
    Vec3i& et_for_dir(Direction dir, Vec2i pos);
    void show(cv::Mat active);
  public:
    // from parent class
    virtual void update_base(const StatsUpdateArgs args);
    virtual void update_recur(const StatsUpdateArgs args);
  };  
}

#endif
