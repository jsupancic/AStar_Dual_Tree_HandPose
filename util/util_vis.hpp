/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_UTIL_VIS
#define DD_UTIL_VIS

#include "DepthFeatures.hpp"

namespace deformable_depth
{ 
  enum KEY_CODES
  {
    ENTER = 10,
    ESCAPE = 27    
  };  

  bool have_display();
  struct PictureHOGOptions
  {
    bool contrast_sensitive;
    bool check_errors;
    PictureHOGOptions(bool contrast_sensitive, bool check_errors):
      contrast_sensitive(contrast_sensitive), check_errors(check_errors) {} ;
  };
  class MetaData;
  class DetectionSet;
  Mat drawDets(MetaData&metadata,const DetectionSet&dets, int shape = 0, int bg = 0);
  void logMissingPositives(MetaData&ex,const DetectionSet&dets);
  void split_feat_pos_neg(const vector<double>&feat,vector<double>&tpos,vector<double>&tneg);
  Mat picture_HOG_one(DepthFeatComputer& hog, std::vector< double > T,PictureHOGOptions opts = PictureHOGOptions(true,true));
  FeatVis picture_HOG_pn(DepthFeatComputer& hog, std::vector< double > T,PictureHOGOptions opts = PictureHOGOptions(true,true));
  Mat picture_HOG(DepthFeatComputer&hog,vector<double> T,PictureHOGOptions opts = PictureHOGOptions(true,true));
  Mat display(string title, Mat image,bool show = true, bool resize = true);
  Mat imagesc(const char*winName,Mat image, bool verbose = false, bool show = true);
  Mat imagesc(const char*winName,Mat_<float> image, bool verbose = false, bool show = true);
  Mat imageeq(const char*winName,Mat_<float> im, 
	      bool show = true, bool resize = true, bool verbose = false);
  Mat eq(const Mat&m);
  Mat imageeq(const Mat&m);
  Mat image_freqs(const Mat&m); // visualize the frequencies in an image using the fft
  Mat imagehog(string title, DepthFeatComputer&hog,vector<double> T,PictureHOGOptions opts = PictureHOGOptions(true,true));
  Mat im_merge(const Mat&r, const Mat&g, const Mat&b);
  Mat image_datum(MetaData&datum,bool write = false);
  Mat imagefloat(string title, Mat image, bool show = true, bool resize = true);
  Mat image_safe(string title, Mat im,bool resize = true);
  void waitKey_safe(int delay);
  Mat imVGA(Mat im, int interpolation = cv::INTER_LINEAR);  
  Mat image_text(const Mat&image, string text, Vec3b color = Vec3b(255,255,255));
  Mat image_text(string text,Vec3b color = Vec3b(255,255,255),Vec3b bg = Vec3b(0,0,0));
  Mat image_text(vector<string> lines);
  Mat replace_matches(const Mat&im0, const Mat&im1,const Mat&repl);
  Mat replace_matches(const Mat&im0, const Vec3b color,const Mat&repl);
  Mat rotate_colors(const Mat&orig,const Mat&rot);  
  void line(Mat&mat,Point p1, Point p2, Scalar color1, Scalar color2);
  Mat monochrome(Mat&m,Vec3b chrome);
  Mat mm_quant(const Mat&m);

  class ProgressBars_Impl; 
  class ProgressBars
  {
  protected:    
    std::unique_ptr<ProgressBars_Impl> pimpl;    

  public:
    ProgressBars();    
    void set_progress(const std::string&title,int value, int total);
  };
  extern ProgressBars*progressBars;
  
  class ProgressBar
  {
  protected:
    std::string title;
    int max;

  public:
    virtual ~ProgressBar();
    ProgressBar(const std::string&title,int max);
    void set_progress(int index);
  };

  // get a point from the uesr.
  Point2i getPt(string winName,bool*visible,char*abort_code = nullptr);
  Rect getRect(std::string winName, cv::Mat bg, cv::Rect init = Rect(), bool allow_null = false, bool*set = nullptr);  
  
  void test_vis();

  extern recursive_mutex exclusion_high_gui;
}

#endif


