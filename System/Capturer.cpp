/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "Capturer.hpp"
#include "util.hpp"

namespace deformable_depth
{
  Capturer::Capturer() : capture(CV_CAP_OPENNI)
  { 
    // init OpenNI
    //capture.set(CV_CAP_PROP_FRAME_WIDTH,640/2);
    //capture.set(CV_CAP_PROP_FRAME_HEIGHT,480/2);
    capture.set(CV_CAP_PROP_OPENNI_REGISTRATION, CV_CAP_PROP_OPENNI_REGISTRATION_ON);
    cout << "capture.isOpened() == " << capture.isOpened() << endl;
    assert(capture.isOpened());
    
    // grab the histograms
    FileStorage skinHists("skin_hist.yml", FileStorage::READ);
    skinHists["fg_hist"] >> skin_hist;
    skinHists["bg_hist"] >> bg_hist;
    skinHists.release();
  }
  
  void doOpenNI(
    VideoCapture&capture,Mat&depthOut,
    Mat&rgbImage)
  {
    Mat depthMap, depthValid;

    capture.grab(); 
    capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP );
    capture.retrieve( rgbImage, CV_CAP_OPENNI_BGR_IMAGE );
    capture.retrieve( depthValid, CV_CAP_OPENNI_VALID_DEPTH_MASK);
    
    // get circles
    //vector<Vec3f> circles;
    //HoughCircles(depthMap,circles, CV_HOUGH_GRADIENT, 2,
    //	 depthMap.rows/4, 200, 100);
    //for(int iter = 0; iter < circles.size(); iter++)
    //{
    //  Point center(cvRound(circles[iter][0]),cvRound(circles[iter][1]));
    //  int radius = cvRound(circles[iter][2]);
    //  circle(rgbImage, center, radius, Scalar(0,0,255), 3, 8, 0);
    //}
    
    // noramalize the depth image
    double depth_min, depth_max;
    //cout << "min = " << depth_min << " max = " << depth_max << endl;
    depthMap.convertTo(depthMap,cv::DataType<float>::type);
    depthMap.setTo(numeric_limits<float>::quiet_NaN(),1-depthValid);
    //cout << depthMap << endl;
      
    // get edges
    //Mat depthEdges = autoCanny(255*depthMap);
    //imshow("Captured: Depth edges",depthEdges);   
        
    depthOut = depthMap/10; // convert mm to cm
    depthOut = fillDepthHoles(depthOut);
  }
  
  void Capturer::doIteration(Mat&Zcap,Mat&RGBcap)
  {
    doOpenNI(capture,Zcap,RGBcap);  
  }
}
