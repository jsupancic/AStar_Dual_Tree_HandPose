/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_CYBER_GLOVE
#define DD_CYBER_GLOVE

#include <istream>
#include <string>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

namespace deformable_depth
{
  using std::istream;
  using std::string;
  using std::map;
  using std::vector;
  using cv::VideoCapture;
  using cv::Mat;
  
  /**
   * -- in the *.dat file each row is a sample containing:
   * a timestamp starting from 0
   * 6 reals from the FoB tracker
   * 9 zeros (disabled sensors)
   * 22 8 bit integers from the Cybergglove
   * 4 16 bit integers from the pressure sensors (only the first is meaningful)
   **/
  class UnigeParamterSequence
  {
  public:
    UnigeParamterSequence(string filename);
    Mat getFrame(size_t index);
    vector<double> getFoB(size_t index);
    vector<double> getCyberGlove(size_t index);
    size_t length();
    
  protected:
    // from the .dat
    map<double/*time stamp*/,vector<double> > fob;
    map<double,vector<double> > disabled;
    map<double,vector<double> > cyberglove;
    map<double,vector<double> > pressure;
    
    // from the .stmp
    map<double,double> timestamp2frame;
    map<double,double> frame2timestamp;
    
    // from the .avi
    VideoCapture video;
  };
  
  void cyberglove_reverse_engineer();
}

#endif
