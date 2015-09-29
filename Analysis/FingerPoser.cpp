/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "FingerPoser.hpp"
#include "OcclusionReasoning.hpp"
#include "Log.hpp"
#include <queue>

namespace deformable_depth
{
  Mat FingerPoser::segment_bounds(Rect bb, const ImRGBZ&im, float z_min, float z_max)
  {
    Mat inHand(im.rows(),im.cols(),DataType<uchar>::type);
    for(int yIter = 0; yIter < im.rows() - 1; yIter++)
      for(int xIter = 0; xIter < im.cols(); xIter++)
      {
	float imZ = im.Z.at<float>(yIter,xIter);
	bool close_enough = goodNumber(z_max)?imZ < z_max:true;
	bool far_enough   = goodNumber(z_min)?z_min < imZ:true;
	bool good_depth = close_enough and far_enough;
	bool good_bb    = bb.contains(Point2i(xIter,yIter));
	float rightZ = im.Z.at<float>(yIter,xIter+1);
	if(good_depth && good_bb and (rightZ != imZ))
	  inHand.at<uchar>(yIter,xIter) = 0; // 0 for in hand
	else
	  inHand.at<uchar>(yIter,xIter) = 255; // 255 for not in hand
      }
    
    return inHand;
  }

  static double dist(Point2d p1, Point2d p2)
  {
    return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
  }

  Mat FingerPoser::segment_flood(Rect bb, const ImRGBZ&im,map<string,AnnotationBoundingBox>&poss)
  {
    bb = clamp(im.Z, bb);
    Point2d handCenter = rectCenter(poss.at("HandBB"));
    // for(auto && pair : poss)
    //   cout << pair.first << endl;
    // Point2d wristCenter = rectCenter(poss.at("wrist"));

    //double root_z = extrema(im.Z(bb)).min;
    double min_z = manifoldFn_apxMin(im,bb).front() - 1;
    double max_z = min_z + params::obj_depth() + 1;    
    Mat inHand(im.rows(),im.cols(),DataType<uchar>::type,Scalar::all(255));    

    std::queue<Point> q;
    Point2d p0 = rectCenter(bb);
    inHand.at<uchar>(p0.y,p0.x) = 0;
    q.push(rectCenter(bb));    
    while(not q.empty())
    {
      Point p = q.front(); q.pop();
      
      auto tryInsertPt = [&](int x, int y)
      {
	if(x < bb.tl().x  or y < bb.tl().y or 
	   bb.br().x <= x or bb.br().y <= y)
	  return;
	if(inHand.at<uchar>(y,x) < 100)
	  return;
	float z_here = im.Z.at<float>(p.y,p.x);
	float z_next = im.Z.at<float>(y,x);
	if(std::abs(z_here - z_next) > 2)
	  return;	
	//if(dist(Point2d(x,y),wristCenter) < dist(Point2d(x,y),handCenter))
	//return;

	inHand.at<uchar>(y,x) = 0;
	q.push(Point(x,y));
      };
      // left
      tryInsertPt(p.x+1,p.y);
      // right
      tryInsertPt(p.x-1,p.y);
      // up
      tryInsertPt(p.x,p.y-1);
      // down
      tryInsertPt(p.x,p.y+1);

      assert(q.size() < 10*im.rows()*im.cols());
    }

    return inHand;
  }

  Mat FingerPoser::segment(Rect bb,const ImRGBZ&im)
  { 
    bb = clamp(im.Z, bb);
    //double root_z = extrema(im.Z(bb)).min;
    double min_z = manifoldFn_apxMin(im,bb).front() - 1;
    double max_z = min_z + params::obj_depth() + 1;
    return segment_bounds(bb,im,qnan,max_z);
  }

  bool FingerPoser::is_freespace(Rect bb, const ImRGBZ&im)
  {
    Mat seg = segment(bb,im); // uchar, 0 in hand
    int mal_seg_corners = 0;

    bb = clamp(im.Z,bb);

    if(seg.at<uchar>(bb.tl().y,bb.tl().x) < 100)
      mal_seg_corners++;
    if(seg.at<uchar>(bb.tl().y,bb.br().x-1) < 100)
      mal_seg_corners++;
    if(seg.at<uchar>(bb.br().y-1,bb.tl().x) < 100)
      mal_seg_corners++;
    if(seg.at<uchar>(bb.br().y-1,bb.br().x-1) < 100)
      mal_seg_corners++;

    bool is_freespace = mal_seg_corners <= 1;
    if(is_freespace)
    {
      //log_im("Freespace",imageeq("",im.Z,false,false));
      log_im(safe_printf("Freespace_%_",mal_seg_corners),seg);
    }
    else
    {
      //log_im("MalSeg",imageeq("",im.Z,false,false));
      log_im(safe_printf("MalSeg_%_",mal_seg_corners),seg);
    }
    return is_freespace;
  }
  
  BaselineDetection VoronoiPoser::pose(const BaselineDetection& src,ImRGBZ&im)
  {
    if(src.bb == Rect())
      return src;
    
    // segment the hand
    BaselineDetection pose_corrected(src);
    Mat inHand = segment(pose_corrected.bb,im);
    Mat vis = im.RGB.clone();
    
    // compute the discrete voroni diagram
    Mat dt, bp;
    voronoi(inHand, dt, bp);
    
    for(auto && part : pose_corrected.parts)
    {
      // get the part baseline
      string part_name = part.first;
      BaselineDetection&part_det = part.second;
      Point2d old_center = rectCenter(part_det.bb);
      old_center.x = clamp<int>(0,old_center.x,dt.cols-1);
      old_center.y = clamp<int>(0,old_center.y,dt.rows-1);
      double dist_from_seg = dt.at<float>(old_center.y,old_center.x);
      cv::circle(vis,rectCenter(part.second.bb),5,Scalar(0,0,255)); // old positions in red

      if(false && dist_from_seg > std::sqrt(static_cast<double>(src.bb.size().area())/20.0))
      {
	part_det.bb = rectFromCenter(rectCenter(src.bb),part_det.bb.size());
	part.second.remapped = true;
      }
      else if(dist_from_seg > 0)
      {
	// re-map it using our BP matrix
	Vec2i newCenter = bp.at<Vec2i>(old_center.y,old_center.x);
	Point2d p_new_center(newCenter[1],newCenter[0]);
	part_det.bb = rectFromCenter(p_new_center,part_det.bb.size());
	part.second.remapped = true;
      }      
      
      // draw any movement...
      cv::circle(vis,rectCenter(part.second.bb),5,Scalar(255,0,0)); // new positions in blue
      cv::line(vis,old_center,rectCenter(part.second.bb),Scalar(0,255,0));
    }
    
    //log_im_decay_freq("Voronoi_pose_update",vis);    
    inHand.convertTo(inHand,DataType<Vec3b>::type);
    log_im("Voronoi_pose_update",vertCat(vis,imageeq("",inHand,false,false)));    
    return pose_corrected;
  }

  ///
  /// SECTION: DumbPoser
  ///
  BaselineDetection DumbPoser::pose(const BaselineDetection&src,ImRGBZ&im)
  {
    BaselineDetection pose_corrected(src);
    Mat inHand = segment(pose_corrected.bb,im);
    Mat dt, bp; voronoi(inHand,dt,bp);

    for(auto && part : pose_corrected.parts)
    {
      Point2d old_center = rectCenter(part.second.bb);
      old_center.x = clamp<int>(0,old_center.x,dt.cols-1);
      old_center.y = clamp<int>(0,old_center.y,dt.rows-1);
      double dist_from_seg = dt.at<float>(old_center.y,old_center.x);
      if(dist_from_seg > 1)
	part.second.bb = Rect(0,0,1,1);
    }

    return pose_corrected;
  }
}
