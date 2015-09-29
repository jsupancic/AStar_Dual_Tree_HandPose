/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_FEAT_VIS
#define DD_FEAT_VIS

#include <string>
#include <opencv2/opencv.hpp>

namespace deformable_depth
{
	using std::string;
	using cv::Mat;
	using cv::DataType;
	using cv::Vec3b;

  // Visualization of a Faeture vector. pos contains the positive 
  // componenet while neg contains the negative component
  struct FeatVis
  {
  public:
    FeatVis(string source);
    FeatVis(string source,Mat pos, Mat neg);
    const Mat& getPos() const;
    const Mat& getNeg() const;
    void setPos(Mat pos) ;
    void setNeg(Mat neg) ;
    string getSource() const;
  protected:
    void append_transform_name(string name);
    Mat pos;
    Mat neg;
    string source;
  };
}

#endif
