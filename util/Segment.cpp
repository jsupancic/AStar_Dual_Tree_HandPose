/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Segment.hpp"
#include "Renderer.hpp"
#include "Capturer.hpp"
#include "Pipe.hpp"
#include <highgui.h>
#include <GL/glut.h>

namespace deformable_depth
{
  /// SECTION: Segemntation  
  struct Message_segment_caputre_to_circle
  {
    Mat ZDT, RGBcap;
  public:
    Message_segment_caputre_to_circle(Mat ZDT,Mat RGBcap)
    {
      this->ZDT = ZDT.clone();
      this->RGBcap = RGBcap.clone();
    }
  };
  
  static Pipe<Message_segment_caputre_to_circle> 
    pipe_segment_capture_to_circles;
  
  Mat ZDT(Mat Zcap)
  {
    // find the edges and take the distance transform.
    cv::Mat dxZ, dyZ, edgesZ ,magZ, ZDT, magZnorm, magZthresh;
    //cv::Scharr(Zcap,dxZ,CV_32F,1,0);
    //cv::Scharr(Zcap,dyZ,CV_32F,0,1);
    filter2D(Zcap,dxZ,-1,params::dxFilter);
    filter2D(Zcap,dyZ,-1,params::dyFilter);
    magZ = sqrt((Mat_<float>)(dxZ.mul(dxZ) + dyZ.mul(dyZ)));
    // as we get further from the camera, the x-y distance
    // between adjacent pixels increases so we must normalize.
    magZnorm = magZ / Zcap;
    cv::threshold(magZnorm,magZthresh,params::DEPTH_EDGE_THRESH,1,cv::THRESH_BINARY);
    edgesZ = magZthresh | isnan(Zcap);
    //imagesc("EDGES!",edgesZ);
    
    // take the DT
    edgesZ = 1 - edgesZ;
    edgesZ.convertTo(edgesZ,cv::DataType<uchar>::type);    
    cv::distanceTransform(edgesZ,ZDT,CV_DIST_L2,CV_DIST_MASK_PRECISE);
    
    //imagesc("ZDT!",ZDT);
    return ZDT;
  }
    
  void segment_iteration_caputre()
  {
    static Renderer renderer;
    static Capturer capturer;
    Mat Zrnd,RGBrnd,Zcap,RGBcap; 
    capturer.doIteration(Zcap,RGBcap);
    renderer.doIteration(Zrnd,RGBrnd);
        
    // send it to the next stage
    pipe_segment_capture_to_circles.push
      (Message_segment_caputre_to_circle(ZDT(Zcap),RGBcap));
  }
  
  struct Message_segment_circle_to_paths
  {
  public:
    Mat ZDT, RGBcap;
    vector<Circle> circles;
    Message_segment_circle_to_paths(Mat ZDT, Mat RGBcap, vector<Circle>&circles)
    {
      this->ZDT = ZDT;
      this->RGBcap = RGBcap;
      this->circles = circles;
    }
  };
  
  static Pipe<Message_segment_circle_to_paths> 
    pipe_segment_circles_paths;
  
  void segment_iteration_circles()
  {
    while(true)
    {
      cout << "segment_iteration_circles" << endl;
      auto msg = pipe_segment_capture_to_circles.pull(true);
      Mat ZDT = msg.ZDT;
      Mat RGBcap = msg.RGBcap;
      
      // segment!
      std::vector<Circle> circles = segment(ZDT);
      Mat rgbSegs = RGBcap.clone();
      for(Circle circle : circles)
      {
// 	rgbSegs = draw(rgbSegs,circle);
// 	cout << "x: " << circle.getX() 
// 	  << " y: " << circle.getY()
// 	  << " r: " << circle.getR() << endl;
      }  
      
      pipe_segment_circles_paths.push(
	Message_segment_circle_to_paths(ZDT,RGBcap,circles));
    }
  }
  
  struct Message_segment_paths_to_detect
  {
  public:
    vector<Circle> circles;
    vector<Edge> MST;
    Mat RGBcap, ZDT;
    Message_segment_paths_to_detect(
      Mat RGBcap, Mat ZDT,vector<Circle>&circles,vector<Edge>&MST)
    {
      this->circles = circles;
      this->MST = MST;
      this->RGBcap = RGBcap;
      this->ZDT = ZDT;
    }
  };
 
  static Pipe<Message_segment_paths_to_detect> 
    pipe_segment_paths_detect;  
  
  // find the widdest paths in the graph.
  void segment_iteration_paths()
  {
    while(true)
    {
      cout << "segment_iteration_paths" << endl;
      auto msg = pipe_segment_circles_paths.pull(true);
      vector<Circle> circles = msg.circles;
      Mat ZDT = msg.ZDT;
      // start all pairs as having width = 0, try to increase this
  
      /// compute the edge weights 
      Mat_<float> pwWidths;
      // this is the slowest line...
      pwWidths = deformable_depth::pwWidths(ZDT,circles);
      Mat_<float> pwDists = deformable_depth::pwDists(ZDT,circles);
      pwDists.setTo(inf,pwWidths < 4);
      
      /// find the MST (maximum spanning tree)
      vector<Edge> mst = MST_Kruskal(-pwDists,-inf);
      
      // draw lines between the circles
      const bool DRAW = true;
      if(DRAW)
      {
	Mat rgbLines = msg.RGBcap.clone();
	for(Circle c : circles)
	  rgbLines = draw(rgbLines,c);
	for(Edge edge : mst)
	{
	  Point2i 
	    p1 = circles[edge.v1].center(),
	    p2 = circles[edge.v2].center();
	  cout << "weight = " << edge.weight << endl;
	  cv::line(rgbLines,p1,p2,Scalar(255,0,0));	
	}  
	imshow("graph",rgbLines);
	cvWaitKey(1);
      }
      
      // to the next stage of the pipeline!
      pipe_segment_paths_detect.push(
	Message_segment_paths_to_detect(msg.RGBcap,msg.ZDT,circles,mst));
    }
  }
  
  void segment_iteration_detect()
  {
    while(true)
    {
      auto msg = pipe_segment_paths_detect.pull(true);
      cout << "+segment_iteration_detect" << endl;
      Mat ZDT = msg.ZDT;
      Mat RGBcap = msg.RGBcap;
      vector<Circle> circles = msg.circles;
      vector<Edge> MST = msg.MST;
      
      
    }
  }
  
  void segment(int argc, char**argv)
  {
    // start the segmentation
    std::thread(segment_iteration_circles).detach();
    std::thread(segment_iteration_paths).detach();
    std::thread(segment_iteration_detect).detach();
    
    // start the glut
    init_glut(argc, argv, segment_iteration_caputre);
    glutMainLoop();    
  }  
}
