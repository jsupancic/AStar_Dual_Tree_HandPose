/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_SUBTREES
#define DD_SUBTREES

#include "util_mat.hpp"
#include "util.hpp"
#include "TreeDecorators.hpp"
#include "ZSpanTree.hpp"

namespace deformable_depth
{
  struct Root
  {
    int row, col;
    Direction dir;
  };
  
  struct TreeStats
  {
    constexpr static float nan = numeric_limits<float>::quiet_NaN();
    Mat 
      active, dt,
      areaLeft, areaRight, areaUp, areaDown,
      perimLeft, perimRight, perimUp, perimDown;
    GraphHash treeMap;
    EdgeTypeDecorator etDecorator;
    FaceAreaDecorator faDecorator;
    CentroidDecorator centroidDecorator;
    SupervisedDecorator supDecorator;
    PixelAreaDecorator pixelAreaDecorator;
    SkinDetectorDecorator skinDetectorDecorator;
    const ImRGBZ im;
    
    typedef deformable_depth::Direction Direction;
    
  public:
    TreeStats(const ImRGBZ&im,Mat&dt,GraphHash&treeMap,Mat pos_mask = Mat());
    float& area_for_dir(Direction dir, Vec2i pos);
    float& perim_for_dir(Direction dir, Vec2i pos);
    Mat features(cv::Mat& X, cv::Mat& Y, cv::Rect gt_bb);
    Mat  features_positive(cv::Mat& X, cv::Mat& Y, cv::Rect gt_bb, deformable_depth::Root& pos);
    void features_negative(cv::Mat& X, cv::Mat& Y, cv::Rect gt_bb, deformable_depth::Root pos);
    Mat feature(int row, int col, Direction dir);
    void drawTree(const ImRGBZ&im, Mat&visited, string name = "Detection") const;
    void showTree(const ImRGBZ& im, 
			    int rIter, int cIter, int dir_raw, 
			     Mat&visited, bool show = true) const;
  protected:
    void mark_tree_region(int rIter, int cIter, int dir_raw, cv::Mat& visited) const;
    void compute_DFS(const int row, const int col,GraphHash&treeMap,const ImRGBZ&im);
    bool allSet(int row, int col, Direction dir = ALL);
    float extrinsicArea(int row, int col, Direction dir);
    float extrinsicPerim(int row, int col, Direction dir);
    void show(int delay = 0);
    void updateStats(Vec2i&here, const ImRGBZ&im, TreeStats::Direction dir);
  };  
  
  TreeStats treeStatsOfImage(const ImRGBZ&im, Rect posMask = Rect());
  TreeStats&treeStatsOfImage_cached(const ImRGBZ&im);
  PixelAreaDecorator&pixelAreas_cached(const ImRGBZ&im);
}

#endif
