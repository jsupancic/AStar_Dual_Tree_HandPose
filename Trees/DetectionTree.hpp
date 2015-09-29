/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_DETECTIONTREE
#define DD_DETECTIONTREE

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <functional>

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  class Decision
  {
  public:
    virtual bool decide(const Mat&x) = 0;
    virtual void println() = 0;
    virtual void update_uses(vector<int>&var_uses) = 0;
    virtual double boundary_dist(const Mat&x) = 0;
    double mean();
    double varience();
    void computeMoments();
    Decision(std::shared_ptr<std::vector< float > > values);
    Decision(double mu, double sigma);
    virtual ~Decision();
    virtual void write(FileStorage&fs);
  private:
    std::shared_ptr<std::vector<float> > values;
    double mu, sigma;
  };
  
  typedef std::shared_ptr<Decision> dptr;
  
  class DecisionThreshold : public Decision
  {
  public:
    DecisionThreshold(std::shared_ptr<std::vector<float> >, int best_var, float best_value);
    DecisionThreshold(double mu, double sigma, int best_var, float best_value);
    virtual bool decide(const Mat&x);
    virtual void println();    
    virtual void update_uses(vector<int>&var_uses);
    virtual double boundary_dist(const Mat&x);
    virtual void write(FileStorage&fs);
  private:
    int best_var;
    float best_value;
  };
  
  class DecisionRange : public Decision
  {
  public:
    DecisionRange(std::shared_ptr<std::vector<float> >,int var, float low, float high);
    DecisionRange(double mu, double sigma, int var, float low, float high);
    virtual bool decide(const Mat&x);
    virtual void println();    
    virtual void update_uses(vector<int>&var_uses);
    virtual double boundary_dist(const Mat&x);
    virtual void write(FileStorage&fs);
  private:
    int var;
    float low, high;
  };
  
  struct VarInfo;
  
  // a classifier
  class DetectionTree
  {
  protected:
    // general info
    double n_pos, n_neg, n_agg;
    enum NodeType
    {
      LEAF, INTERNAL
    };
    NodeType node_type;
    // if internal
    dptr decision;
    std::shared_ptr<DetectionTree> branch_true;
    std::shared_ptr<DetectionTree> branch_false;
  protected:
    // don't split smaller collections
    constexpr static int MIN_SPLIT = 1000;
    constexpr static int MAX_DEPTH = 8;
    constexpr static float POS_WEIGHT = 100;
  public:
    DetectionTree();
    DetectionTree(cv::Mat& X, cv::Mat& Y, vector<int> var_uses = vector<int>(), int depth = 0);
    double predict(Mat&x);
    void show(int depth = 0);
  protected:
    void train(cv::Mat& X, cv::Mat& Y, vector<int> var_uses = vector<int>(), int depth = 0);
    double predict_here();
    dptr choose_split(cv::Mat& X, cv::Mat& Y, std::vector< int >& var_uses) const;
    void split_data(Mat&XPrime, Mat&YPrime, Mat X, Mat Y, int side) const;
    void eval_thresholds(
      int var_idx,
      dptr&best_decsion,
      double&best_info_gain,
      VarInfo&info) const;
    void eval_ranges(int var_idx, dptr&best_decsion,double&best_info_gain,VarInfo&info) const;
    friend void write(FileStorage&fs, const string&, const DetectionTree&tree);
    template<typename Storage> friend void read(const Storage&node, 
						DetectionTree&tree, 
						const DetectionTree&default_value);
  };
  
  /// serialization
  // detection tree
  void write(FileStorage&fs, const string&, const DetectionTree&tree);
  void read(const FileStorage&node, DetectionTree&tree, const DetectionTree&default_value = DetectionTree());
  // decision
  void write(FileStorage&fs, const string&, const std::shared_ptr<Decision>&writeMe);
}

#endif
