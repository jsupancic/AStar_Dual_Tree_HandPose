/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "ConvexityDefects.hpp"
#include "Log.hpp"
#include <opencv2/opencv.hpp>

namespace deformable_depth
{
  using namespace cv;
  
  // return true if the pt is on the boarder of bb
  bool onBoarder(Rect bb, Point pt)
  {
    return 
      pt.x <= bb.tl().x + 1 ||
      pt.y <= bb.tl().y + 1 ||
      pt.x >= bb.br().x - 1 ||
      pt.y >= bb.br().y - 1;
  }
    
  ConvexExtrema FingerConvexityDefectsPoser::analyze_convexity(Rect bb, ImRGBZ& im)
  {
    // (1) Segement the foreground from the background
    Mat segmentation = segment(bb,im);
    Mat inHand = 255 - segmentation;
    //Mat vis_seg; //inHand.convertTo(vis_seg,DataType<Vec3b>::type);
    Mat vis_seg; cvtColor(inHand,vis_seg,CV_GRAY2BGR);
    Mat vis = Mat(inHand.rows, inHand.cols, DataType<Vec3b>::type, Scalar::all(255));
    rectangle(vis,bb.tl(),bb.br(),Scalar(0,0,0));
    double root_z = extrema(im.Z(bb)).min;
    
    // (2a) analyize the convexity defects
    // find the contours
    vector<vector<Point> > contours;
    // CV_CHAIN_APPROX_NONE vs. CV_CHAIN_APPROX_SIMPLE
    cv::findContours(inHand,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    log_file << "number of contours: " << contours.size() << endl;
    cv::drawContours(vis,contours,-1,Scalar(255,0,255));
    
    // find the convex hulls
    vector<vector<int> > convex_hulls_i(contours.size());
    vector<vector<Point> > convex_hulls_p(contours.size());
    for(int iter = 0; iter < contours.size(); ++iter)
    {
      cv::convexHull(Mat(contours[iter]),convex_hulls_i[iter],false,false);
      cv::convexHull(Mat(contours[iter]),convex_hulls_p[iter],false,true);
    }
    cv::drawContours(vis,convex_hulls_p,-1,Scalar(0,255,0));
    
    // find the damn convexity defects
    function<unsigned long(Point)> hashFn = [](Point p)->unsigned long
    {
      return std::hash<int>()(p.x) ^ std::hash<int>()(p.y);
    };
    function<bool(const Point&,const Point&)> eqFn = [](const Point&p1,const Point&p2)
    {
      return p1 == p2;
    };
    ConvexExtrema likely_fingers(0,hashFn,eqFn);
    vector<vector<Vec4i > > defect_indices(contours.size());
    for(int iter = 0; iter < contours.size(); ++iter)
    {
      log_file << "contour size: " << contours[iter].size() << endl;
      log_file << "convex hull size: " << convex_hulls_i[iter].size() << endl;
      if(convex_hulls_i[iter].size() > 3)
      {
	cv::convexityDefects(contours[iter],convex_hulls_i[iter],defect_indices[iter]);
      }
      log_file << printfpp("Got %d defects",(int)defect_indices[iter].size()) << endl;
      // visulize the defects
      for(Vec4i & defect_index : defect_indices[iter])
      {
	Point start_point = contours[iter][defect_index[0]];
	Point end_point = contours[iter][defect_index[1]];
	Point defect_point = contours[iter][defect_index[2]];
	double defect_depth = im.camera.DistPixtoWorldCM(root_z,defect_index[3]/256.0);
	
	if(defect_depth > 2.5)
	{
	  if(!onBoarder(bb,start_point))
	  {
	    cv::circle(vis,start_point,4,Scalar(255,0,0));
	    likely_fingers.insert(start_point);
	  }
	  if(!onBoarder(bb,end_point))
	  {
	    cv::circle(vis,end_point,4,Scalar(255,0,0));
	    likely_fingers.insert(end_point);
	  }
	  //cv::circle(vis,defect_point,3,Scalar(0,255,0));
	}
	else
	  cv::circle(vis,start_point,1,Scalar(0,0,255));
      }
    }
    log_im_decay_freq("ConvexityDefects",horizCat(vis,im.RGB));
    
    return likely_fingers;
  }
  
  BaselineDetection FingerConvexityDefectsPoser::pose(const BaselineDetection&src,ImRGBZ&im)
  {
    return pose_inform_error_correction(src,im);
  }

  BaselineDetection FingerConvexityDefectsPoser::pose_inform_error_correction(const BaselineDetection&src,ImRGBZ&im)
  {
    BaselineDetection pose_corrected(src);
    Mat inHand = segment(pose_corrected.bb,im);
    Mat dt, bp; voronoi(inHand,dt,bp);
    Mat vis = im.RGB.clone();

    ConvexExtrema likely_finger_set = analyze_convexity(pose_corrected.bb,im);
    
    for(auto && part : pose_corrected.parts)    
    {
      Point2d old_center = rectCenter(part.second.bb);
      cv::circle(vis,rectCenter(part.second.bb),5,Scalar(0,0,255));
      double dist_from_seg = dt.at<float>(old_center.y,old_center.x);
      if(dist_from_seg > 2)
      {
	double min_dist = inf;
	for(Point2d likely_finger : likely_finger_set)
	{
	  double ddist = likely_finger.ddot(rectCenter(part.second.bb));
	  if(ddist < min_dist)
	  {
	    min_dist = ddist;
	    part.second.bb = rectFromCenter(likely_finger,part.second.bb.size());
	  }
	}
      }
      cv::circle(vis,rectCenter(part.second.bb),5,Scalar(255,0,0));
      cv::line(vis,old_center,rectCenter(part.second.bb),Scalar(0,255,0));
    }
    log_im("corrections",vis);

    return pose_corrected;
  }

  BaselineDetection FingerConvexityDefectsPoser::pose_cover_fingers(const BaselineDetection& src, ImRGBZ& im)
  {
    BaselineDetection detection(src);
    
    // check that the segmentation is reasonable
    Mat segmentation = segment(detection.bb,im);
    Mat inHand = 255 - segmentation;
    double segmentation_ratio = cv::sum(inHand/255)[0] / detection.bb.area();
    log_file << "segmentation_ratio: " << segmentation_ratio << endl;
    if(segmentation_ratio < .10)
    {
      BaselineDetection empty_det;
      return empty_det;
    }
   
    ConvexExtrema likely_finger_set = analyze_convexity(detection.bb,im);
    vector<Point> likely_fingers(likely_finger_set.begin(),likely_finger_set.end());
    auto finger_covered = [&](int iter)->bool
    {
      Point&finger_loc = likely_fingers[iter];
      for(auto&&part_det : detection.parts)
	if(part_det.second.bb.contains(finger_loc))
	  return true;
      return false;
    };
    auto part_covers = [&](string part_name)->bool
    {
      Rect part_bb = detection.parts[part_name].bb;
      for(int iter = 0; iter < likely_fingers.size(); ++iter)
	if(part_bb.contains(likely_fingers[iter]))
	  return true;
      return false;
    };
    
    // try to cover all convex points with part detections
    for(int iter = 0; iter < likely_fingers.size(); ++iter)
    {
      if(finger_covered(iter))
	continue;
      
      // find the nearest un-covering BB and move it atop.
      string nearest_uncovering_bb = "";
      double min_sq_dist = inf;
      for(auto && part : detection.parts)
      {
	if(part_covers(part.first))
	  continue;
	
	double sq_dist = likely_fingers[iter].ddot(rectCenter(part.second.bb));
	if(sq_dist < min_sq_dist)
	{
	  min_sq_dist = sq_dist;
	  nearest_uncovering_bb = part.first;
	}
      }
      
      if(nearest_uncovering_bb != "")
      {
	Rect& update_bb = detection.parts[nearest_uncovering_bb].bb;
	//detection.parts[nearest_uncovering_bb].resp = inf;
	update_bb = rectFromCenter(likely_fingers[iter],update_bb.size());	
      }
    }
    
    // update responces of covering detections
    for(auto && part : detection.parts)
    {
      log_once(printfpp("FingerConvexityDefectsPoser part = %s",part.first.c_str()));
      if(part_covers(part.first))
	; //detection.parts[part.first].resp += .005;
      else
	; //detection.parts[part.first].resp -= .005;
    }
    
    return detection;
  }
  
  bool FingerConvexityDefectsPoser::is_hard_example(MetaData& metadata)
  {
    Rect handBB = metadata.get_positives()["HandBB"];
    shared_ptr<ImRGBZ> im = metadata.load_im();
    ConvexExtrema likely_fingers = analyze_convexity(handBB, *im);
    
    if(likely_fingers.size() >= 2)
    {
      // easy
      log_im("Easy",im->RGB);
      return false;
    }
    else
    {
      // hard
      log_im("Hard",im->RGB);
      return true;
    }
  }
}

