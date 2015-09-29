/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_PLOT
#define DD_PLOT

#include <util_rect.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "BaselineDetection.hpp"

namespace deformable_depth
{
  using cv::Rect_;  
    
  struct PerformanceRecord
  {
  protected:
    Scores joint_pose_scores;
    DetectorScores finger_scores,agnostic_finger_scores, hand_scores;
    vector<double> finger_errors_cm;
    double gt_count;
    
  public:    
    PerformanceRecord();
    PerformanceRecord(
      DetectorScores hand_scores, 
      DetectorScores finger_scores,
      Scores joint_pose_scores,
      DetectorScores agnostic_finger_scores,
      vector<double> finger_errors_cm,
      double gt_count);
    void merge(PerformanceRecord&&other);
    void merge(PerformanceRecord&other);
    DetectorScores& getHandScores();
    DetectorScores& getFingerScores();
    Scores& getJointPoseScores();
    DetectorScores& getAgnosticFingerScores();
    double&getGtCount() ;
    vector<double>& getFingerErrorsCM();
  };  
  void write(string filename, map<string/*name*/,PerformanceRecord/*record*/> data);
 
  /**
   * Coordinates the evaluation of a single method.
   * We must score multiple of these to then generate
   * comparative plots.
   **/
  struct BaseLineTest 
  { 
    vector<string> videos;
    vector<string> labels;
    string title; // eg the method name
    PerformanceRecord record; // how well did it do?
    // some flags
    bool curve;
    bool comparative;
    bool ablative; 
    bool eval_joint_pose;
  };      
  
  class MatplotlibPlot
  {
  protected:
    string save_file;
    ofstream plot_script;
    
    MatplotlibPlot(string save_file);
    virtual ~MatplotlibPlot();
    void put(string s);
    void header();
    void footer();
  };
  
  // uses YARD not Matplotlib?
  class PrecisionRecallPlot
  {
  protected:
    string filename;
    map<string,PerformanceRecord> records;
    map<string,PerformanceRecord> points;
    string type;
    
  protected:
    void plot_one(ostream&out, bool curve, string title, PerformanceRecord& record, int number);
    
  public:
    PrecisionRecallPlot(string filename, string type);
    virtual ~PrecisionRecallPlot();
    void add_plot(PerformanceRecord record,string label);
    void add_point(string title, PerformanceRecord record);    
  };
  
  class ROC_Plot : protected MatplotlibPlot
  {
  public:
    typedef function<Scores (PerformanceRecord&)> SelectScoresFn;
    
  protected:
    string save_file;
    map<string,PerformanceRecord> records;
    map<string,Vec2d> points;
    SelectScoresFn scoresFn;
    
  public:
    ROC_Plot(string save_file,SelectScoresFn);
    void add_plot(PerformanceRecord record, string label);
    void add_point(string title, double p, double r);
    virtual ~ROC_Plot();
  };
  
  class FingerErrorCumulativeDist : protected MatplotlibPlot
  {
  protected:
    string save_file;
    map<string,vector<double> > finger_errors;
    
  public:
    FingerErrorCumulativeDist(string save_file);
    void add_error_plot(vector<double> finger_errors, string label);
    virtual ~FingerErrorCumulativeDist();
  };    
  
  void comparative_video();
  // generate 
  void comparative_video_assemble();
  // generate a comparative video from track
  void comparative_video_track();
  // take an existing PR curve and bootstrap it.
  void bootstrap_pr_curve();

  // for drawing the video
  typedef function<void (Vec3b&)> ColorFn;
  
  void colorFn_ours(Vec3b&);
  Mat draw_exemplar_overlay(
    Mat&exemplar,Mat&background,
    BaselineDetection&det, ColorFn colorFn = colorFn_ours, Mat segmentation = Mat());
}

#endif

