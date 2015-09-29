/**
 * Copyright 2014: James Steven Supancic III
 **/ 
#ifndef DD_HOUGH_FOREST
#define DD_HOUGH_FOREST


#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <memory>
#include "util_depth.hpp"
#include "ThreadPool.hpp"

namespace std
{
  template <>
  struct hash < cv::Vec3b >
  {
  public :
    size_t operator()(const cv::Vec3b &x ) const;
  };
}

namespace deformable_depth
{
  using cv::Vec3d;
  using cv::Vec3b;
  using cv::Mat;
  using cv::Point;
  using cv::Point2d;
  using std::unordered_map;
  using std::vector;
  using std::shared_ptr;

  // statistics needed to make a prediction
  class PredictionStatistics
  {
  protected:
    // trained prediction at this branch
    Vec3d X; // the first moment
    Mat X2; // the second moment
    double N; 
    unordered_map<Vec3b,long> frequencies;
    double shannon_entropy() const;
    double differential_entropy() const;

  public:
    PredictionStatistics();
    void update(Vec3d sample);
    Vec3d predict() const; 
    unordered_map<Vec3b,long> posterior() const;
    bool splitable() const;    
    double entropy() const;
    double samples() const; 
  };

  //
  // Parameterized Split function & Its statistics
  // 
  class SplitFunction
  {
  protected:
    // data
    int function_code;
    double thresh;
    Point2d d1, d2;
    Point absolute;
    PredictionStatistics stats_above, stats_below;

    // methods
    double entropy() const;
    bool split_kinect_feat(const Mat&Z,Point pt) const;
    bool split_absolute_x(const Mat&Z,Point pt) const;
    bool split_absolute_y(const Mat&Z,Point pt) const;

  public:
    SplitFunction();
    void train(const Mat&sem,const Mat&Z,Point pt);
    double info_gain() const;
    bool split(const Mat&Z,Point pt) const;
  };

  // 
  // Implements a Stochastically trained Extremely Random Tree
  //
  class StochasticExtremelyRandomTree
  {
  protected:   
    // what we predict locally
    PredictionStatistics local_stats;
    // if we have a split function, this is it
    shared_ptr<SplitFunction> splitFn;
    vector<shared_ptr<SplitFunction> > split_candidates;
    // branches
    shared_ptr<StochasticExtremelyRandomTree> branch_above;
    shared_ptr<StochasticExtremelyRandomTree> branch_below;    
    static constexpr int MAX_DEPTH = 9;
    int depth;    

  public:
    StochasticExtremelyRandomTree(int depth = 0);
    Vec3d predict(const Mat&Z,Point pt) const;
    unordered_map<Vec3b,long> posterior(const Mat&Z,Point pt) const;
    void train(const Mat&sem,const Mat&Z,Point pt);
    size_t total_samples() const;
  };

  /// 
  /// Simple 2d hough forest implementation
  ///
  class HoughForest
  {
  protected:
    // data
    vector<StochasticExtremelyRandomTree> trees;

  public:
    HoughForest();
    Mat predict_one_part(const Mat&seg,const Mat&Z,const CustomCamera&camera) const;
    void train_one(Mat&Z,Point2d part_center,TaskBlock&train_trees,Mat seg=Mat());
  };
}

#endif


