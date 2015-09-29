/**
 * Copyright 2012: James Steven Supancic III
 **/


#ifndef DD_UTIL_RECT
#define DD_UTIL_RECT

#define use_speed_ 0
#include <opencv2/opencv.hpp>

#include <string>
#include "util_real.hpp"
#include "ThreadCompat.hpp"
#include <set>

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
#ifndef WIN32
  // forward declare
  class Detection;
  Detection&detection_affine(Detection&detection, Mat&affine_transform);
#endif

  struct Scores;
  
  float rect_max_intersect(vector<Rect> r1s, vector<Rect> r2s);
  float rectIntersect(const Rect&r1, const Rect&r2);
  Rect rectUnion(vector<Rect> rects);
  // scale a rectangle without a change in image size
  Rect rectResize(Rect r, float xScale, float yScale);
  // scale up a  rectangle with an change in image scale
  Size_<double> sizeFromAreaAndAspect(double area, double aspect);
  Rect_<double> rectScale(Rect_<double> r, float factor);
  Rect clamp(Rect container, Rect containee);
  Rect clamp(Mat container, Rect containee);
  bool rectContains(const Mat&container, Rect containee);
  Rect_<double> rectFromCenter(Point2d center, Size_<double> size);
  Point2d rectCenter(const Rect_<double>&r);
  // Deprecated, use add_detection
  bool rectScore(cv::Rect gt, cv::Rect det, double resp, Scores&score);
  bool rectCorrect(cv::Rect gt, cv::Rect det);
  Rect rectOfBlob(Mat&mask, int targetVal);
  Rect_<double> rect_Affine(Rect_<double> rect, const Mat&affine_transform);
  Mat affine_transform(Rect_<double>&src,Rect_<double>&dst);
  Mat affine_transform(Rect src,Rect dst);
  //Mat affine_transform(RotatedRect src, Rect dst);
  Mat affine_transform_rr(RotatedRect src, RotatedRect dst);
  Point_<double> point_affine(Point_<double> point, const Mat& affine_transform);
  Point3d point_affine(Point3d point, const Mat& affine_transform);
  Vec3d vec_affine(Vec3d, const Mat&affine);
  void write_corners(vector<Point2d >&out,const Rect_<double>&rect);
  Rect operator+ (const Rect& r1, const Rect& r2);
  Rect operator* (double scalar, const Rect& r);
  string to_string(const Rect & r);

  struct FrameDet
  { 
  protected:
    double m_score, m_subscore; 
    bool m_correct; // true if positive, false if negative
    
  public:
    FrameDet() {};
    FrameDet(bool correct, double score, double subscore);
    bool is_detection() const;
    bool operator< (const FrameDet&other) const;
    bool operator> (const FrameDet&other) const;
    double score() const;
    double subscore() const;
    bool correct() const;
  };
  void read(FileNode, FrameDet&, FrameDet);
  void write(FileStorage&, string, const FrameDet&);
  
  class IScores
  {
  public:
    virtual double p(double threshold = -inf) const = 0 ;
    virtual double r(double threshold = -inf) const = 0 ;
    virtual double v(double threshold = -inf) const = 0 ;
    virtual void compute_pr(vector< double >& P, vector< double >& R,vector<double>&V) const = 0;
  };
  
  struct Scores : public IScores
  {
  protected: // values
    // detection
    vector<FrameDet> detections;
    mutable mutex monitor;
    
  public: 
    // pose estimation
    double pose_correct;
    double pose_incorrect;    
    
    enum Type
    {
      TP,
      FP,
      FN,
      TN
    };
    
    // functions
    Scores() : pose_correct(0), pose_incorrect(0) {}
    Scores(const Scores&copy);
    Scores& operator=(const Scores&copy);
    Scores(vector<Scores> combine);
    double f1(double threshold = -inf) const;
    double pose_accuracy() const;
    string toString(double threshold) const;
    vector<FrameDet> getDetections() const;
    void add_detection(FrameDet frameDet);
    double tp(double threshold) const;
    double fp(double threshold) const;
    double fn(double threshold) const; 
    void score(Type type,double resp);

    virtual double p(double threshold = -inf)const;
    virtual double r(double threshold = -inf)const;
    virtual double v(double threshold = -inf)const;
    virtual void compute_pr(vector< double >& P, vector< double >& R,vector<double>&V)const;
    
    friend void write(FileStorage& fs, const string& , const Scores& score);
    friend void read(const FileNode& node, Scores& score,const Scores& default_value);
  };
  
  // 
  struct RectAtDepth : public Rect_<double>
  {
  public:
    RectAtDepth();
    RectAtDepth(const Rect_<double>&copy);
    RectAtDepth(Rect_<double> r, double z1,double z2);
    double depth() const;
    double&depth();

  protected:
    double z1;
    double z2;

  public:
    double volume() const;
    double dist(const RectAtDepth&other) const;
    RectAtDepth intersection(const RectAtDepth&other) const;
  };
  
#ifdef DD_CXX11
  // class which scores the results from a detector.
  class DetectorScores : public IScores
  {
  public:
    typedef function<bool (RectAtDepth gt, RectAtDepth Det, const string&frame)> MatchFn;
    typedef function<double (RectAtDepth gt, RectAtDepth Det)> SoftErrorFn;
    
  protected:
    struct ScoredBB
    {
      RectAtDepth BB;
      double resp;
      string filename;
      
      bool operator< (const ScoredBB&other) const;
    };
    
    std::multimap<string,RectAtDepth> ground_truth;
    std::multiset<ScoredBB> detections;
    std::multimap<string,ScoredBB> detections_per_frame;
    MatchFn matchFn;
    
  public:    
    DetectorScores(MatchFn matchFn);
    
    bool put_detection(string filename, RectAtDepth detection, double resp);
    void put_ground_truth(string filename, RectAtDepth gt);
    // returns average of soft errors per detection
    double compute_pr_simple(vector< double >& P, 
			   vector< double >& R,
			   vector<double>&V,
			   SoftErrorFn errFn = nullptr) const;
    void merge(DetectorScores&other);
    virtual double p(double threshold = -inf)const;
    virtual double r(double threshold = -inf)const;
    virtual double v(double threshold = -inf)const;
    virtual void compute_pr(vector< double >& P, vector< double >& R,vector<double>&V) const;
  };
#endif
  
  void write(FileStorage& fs, const string& , const Scores& score);
  void read(const FileNode& node, Scores& score,const Scores& default_value = Scores());
  ostream& operator<< (ostream &out, Rect &r);
  istream& operator>> (istream &in, Rect&r);
}

#endif
