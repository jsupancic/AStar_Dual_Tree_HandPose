/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_VISUALIZATION
#define DD_VISUALIZATION

#include <opencv2/opencv.hpp>
#include <string>
#include <map>

namespace deformable_depth
{
  // exiv2 pr -p a out/exiv2video_dets_NYU_Hands_8020_6e8ef99d-fe8d-4ee9-a1cc-43aa43cc7cc1.jpg
  class Visualization
  {
  protected:
    std::map<std::string,cv::Mat> images;    
    std::map<std::string,cv::Rect> layout;

    void update_layout();

  public:
    Visualization(const std::string&filename);
    Visualization();
    Visualization(const cv::Mat&image,const std::string&title);
    Visualization(const Visualization&v1,const std::string&prefix1,const Visualization&v2,const std::string&prefix2);

    void insert(const Visualization&&vis, const std::string&prefix);
    void insert(const cv::Mat&image,const std::string&title);
    void write(const std::string&prefix) const;
    cv::Mat image() const;
    cv::Mat at(const std::string&id);
  };  
}

#endif
