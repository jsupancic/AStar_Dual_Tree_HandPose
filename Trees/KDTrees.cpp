/**
 * Copyright 2013: James Steven Supancic III
 **/

#include <memory>
#include <opencv2/opencv.hpp>

#include "KDTrees.hpp"
#include "Orthography.hpp"
#include "MetaData.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  shared_ptr<cv::flann::Index> kdTreeXYZS(DetectionSet&fingerTipDetections,ImRGBZ&im)
  {
    cout << "building KD-Tree" << endl;
    Mat sparceFingerDetections(fingerTipDetections.size(),4,DataType<float>::type,
			       Scalar::all(std::numeric_limits<float>::quiet_NaN()));
    for(int fingerDetIter = 0; fingerDetIter < fingerTipDetections.size(); fingerDetIter++)
    {
      Rect_<double> bb = fingerTipDetections[fingerDetIter]->BB;
      float z = medianApx(im.Z,bb);
      float x,y; map2ortho_cm(im.camera,bb.x+bb.width/2,bb.y+bb.height/2,z,x,y);
      float s = log2(bb.area()); // area range 1 to inf => range  0 to inf
      sparceFingerDetections.at<float>(fingerDetIter,0) = x;
      sparceFingerDetections.at<float>(fingerDetIter,1) = y;
      sparceFingerDetections.at<float>(fingerDetIter,2) = z;
      // TODO: experiment with a scale penalty?
      sparceFingerDetections.at<float>(fingerDetIter,3) = 0; //s;
    }
    cv::flann::KDTreeIndexParams kd_index_params;
    const cv::flann::IndexParams& index_params = kd_index_params;
    return shared_ptr<cv::flann::Index>(
      new cv::flann::Index(sparceFingerDetections,index_params));
  }
  
  shared_ptr<cv::flann::Index> kdTreeXYWH(DetectionSet&detections)
  {
    Mat points(detections.size(),4,DataType<float>::type,
	       Scalar::all(std::numeric_limits<float>::quiet_NaN()));
    for(int detIter = 0; detIter < detections.size(); detIter++)
    {
      Detection& detection = *detections[detIter];
      auto BB = detection.BB;
      points.at<float>(detIter,0) = BB.x;
      points.at<float>(detIter,0) = BB.y;
      points.at<float>(detIter,0) = BB.width;
      points.at<float>(detIter,0) = BB.height;
    }
    
    return shared_ptr<cv::flann::Index>(
      new cv::flann::Index(points,cv::flann::KDTreeIndexParams()));
  }
}

