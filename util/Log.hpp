/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_LOG
#define DD_LOG

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <functional>

namespace deformable_depth
{
  using std::ofstream;
  using std::string;
  using cv::Mat;
  
  extern ofstream log_file;

  void init_logs(const string&command);
  bool is_power_of_two(long number);
  void ensure(bool condition_must_be_true);
  void log_im_decay_freq(string prefix, std::function<Mat (void)> im);
  void log_im_decay_freq(string prefix, Mat im);
  void log_text_decay_freq(string counter, std::function<string (void)> text);
  string log_im(string prefix, Mat im);
  void log_once(string message,string extra_unconsidered = "");  
  void log_locked(string message);
  string uuid();
}

#endif
