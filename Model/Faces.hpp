/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_FACES_MODEL
#define DD_FACES_MODEL

#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "Detector.hpp"
#include "Detection.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;

  class SimpleFaceDetector
  {
  public:
    SimpleFaceDetector();
    vector<Rect> detect(const ImRGBZ&im);
    Mat detect_and_show(const ImRGBZ&im);
    Mat show(const ImRGBZ&im,vector<Rect>&detections);
    DetectionSet filter(DetectionSet& src, 
				  const ImRGBZ&im, float thresh = .5);
    
    static void build_cache(const ImRGBZ&example);    
  };
  
  class CachingFaceDetector
  {
  public:
    CachingFaceDetector();
    vector<Rect> detect(const ImRGBZ&im);
    Mat detect_and_show(const ImRGBZ&im);
    Mat show(const ImRGBZ&im,vector<Rect>&detections);
    DetectionSet filter(DetectionSet& src, 
				  const ImRGBZ&im, float thresh = .5);
    
    static void build_cache(const ImRGBZ&example);
  private:
  };
  
  typedef SimpleFaceDetector FaceDetector;
}

#endif
