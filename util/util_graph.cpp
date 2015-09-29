/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "util_graph.hpp"
#include <queue>

namespace deformable_depth
{
  vector<Edge> MST_Kruskal(Mat G, float minEdgeWeight)
  {
    std::map<int,int> vertexToCluster;
    std::map<int/*cluster id*/,vector<int>/*vertexes*/> clusters;
    std::vector<Edge> MST;
    std::priority_queue<Edge> edgeQueue;
    // add all the edges
    for(int iter = 0; iter < G.rows; iter++)
    {
      vertexToCluster[iter] = iter;
      clusters[iter] = {iter};
      for(int jter = iter + 1; jter < G.rows; jter++)
	edgeQueue.push(Edge(iter,jter,G.at<float>(iter,jter)));
    }
      
    while(!edgeQueue.empty() 
	  && MST.size() < G.cols - 1 &&
	  edgeQueue.top().weight > minEdgeWeight)
    {
      // get an edge
      Edge e = edgeQueue.top();
      edgeQueue.pop();
      
      // can we use it?
      int v1 = e.v1;
      int v2 = e.v2;
      int cluster1 = vertexToCluster[v1];
      int cluster2 = vertexToCluster[v2];
      if(cluster1 != cluster2)
      {
	// if yes, add it
	MST.push_back(e);
	
	// and merge the clusters
	for(int vertex : clusters[cluster2])
	{
	  clusters[cluster1].push_back(vertex);
	  vertexToCluster[vertex] = cluster1;
	}
	clusters.erase(cluster2);
      }
    }
      
    return MST;
  }
  
  /// Circle Impl.
  Circle::Circle(float x, float y, float r)
  {
    this->x = x;
    this->y = y;
    this->r = r;
  }

  float Circle::getR()
  {
    return r;
  }

  float Circle::getX()
  {
    return x;
  }

  float Circle::getY()
  {
    return y;
  }
  
  bool Circle::operator<(const Circle& other) const
  {
    return this->r < other.r;
  }
  
  Point Circle::center() const
  {
    return Point(x,y);
  }
  
  Mat draw(Mat src, Circle circle)
  {
    Mat dst = src.clone();
    
    cv::circle(dst, 
	       cv::Point2i(cvRound(circle.getX()),cvRound(circle.getY())), 
	       cvRound(circle.getR()), 
	       Scalar(0,0,255), 1, 8, 0);
    
    return dst;
  }
      
  vector< Circle > segment(const Mat& ZDTc)
  {
    const float MIN_RAD = 5;
    Mat ZDT = ZDTc.clone();
    vector<Circle> circles;
    
    // construct the heap
    std::priority_queue<Circle> Q;
    for(int yIter = 0; yIter < ZDT.rows; yIter+=4)
      for(int xIter = 0; xIter < ZDT.cols; xIter+=4)
      {
	Q.push(Circle(xIter,yIter,ZDT.at<float>(yIter,xIter)));
      }
    
    while(!Q.empty())
    {
      // find a circle
      Circle circle = Q.top();
      Point2i center(cvRound(circle.getX()),cvRound(circle.getY()));
      Q.pop();
      if(ZDT.at<float>(circle.getY(),circle.getX()) > MIN_RAD)
      {
	//prior to the next iteration, we must fill
	//the DT to prevent reselection of the same circle.
	cv::circle(ZDT,
		 center,circle.getR(),
		 cv::Scalar::all(0),
		 -1,8,0);
      // I want to do this but it's to slow
//   for(int xIter = 0; xIter < ZDT.cols; xIter++)
// 	for(int yIter = 0; yIter < ZDT.rows; yIter++)
// 	{
// 	  float deltaX = center.x - xIter;
// 	  float deltaY = center.y - yIter;
// 	  float dist = ::sqrt(deltaX*deltaX + deltaY*deltaY);
// 	  float ZDT_Old = ZDT.at<float>(yIter,xIter);
// 	  float ZDT_New = dist - radius;
// 	  if(ZDT_Old > ZDT_New)
// 	    ZDT.at<float>(yIter,xIter) = ZDT_New;
// 	}
      
	// store the circle.
	circles.push_back(circle);
      }
    }
    
    return circles;
  }
  
  Mat_<float> pwDists(const Mat&ZDT,const vector<Circle>&circles)
  {
    Mat_<float> pwWidths = Mat_<float>::zeros(circles.size(),circles.size());
    for(int iter = 0; iter < circles.size(); iter++)
      for(int jter = 0; jter < circles.size(); jter++)
      {
	Circle ci = circles[iter];
	Circle cj = circles[jter];
	float zi = ZDT.at<float>(ci.getY(),ci.getX());
	float zj = ZDT.at<float>(cj.getY(),cj.getX());
	float deltaX = ci.getX() - cj.getX();
	float deltaY = ci.getY() - cj.getY();
	float deltaZ = zi - zj;
	float dist = dist = ::sqrt(deltaX*deltaX + deltaY*deltaY);
	pwWidths.at<float>(iter,jter) = dist;
	pwWidths.at<float>(jter,iter) = dist;
      }
      
    return pwWidths;
  }
  
  Mat_<float> pwWidths(const Mat&ZDT, const vector<Circle>&circles)
  {
    Mat_<float> pwWidths = Mat_<float>::zeros(circles.size(),circles.size());
  
    const int MIN_WIDTH = 4;
    vector<int> width_schedule = {MIN_WIDTH};
    for(int width : width_schedule)
    {
      cout << "segment_iteration_paths width = " << width << endl;
      // threshold the image
      Mat edges;
      cv::threshold(ZDT,edges,width-1,1,cv::THRESH_BINARY);
      
      // find connected components
      Mat_<int> ccs = connected_components<float,int>(edges,-1);
      //imagesc("CCS: ",ccs);
      //if(width == MIN_WIDTH+1)
      //  cvWaitKey(-1);
      
      // update pw dists
      for(int iter = 0; iter < circles.size(); iter++)
	for(int jter = iter+1; jter< circles.size(); jter++)
	{
	  Circle ci = circles[iter];
	  Circle cj = circles[jter];
	  int xi = ci.getX(), yi = ci.getY(),
	      xj = cj.getX(), yj = cj.getY();
	  float curWidth = pwWidths.at<float>(iter,jter);
	  float newWidth = 
	    (ccs.at<int>(yi,xi) == ccs.at<int>(yj,xj) && 
	      ZDT.at<float>(yi,xi) >= width &&
	      ZDT.at<float>(yj,xj) >= width)
	    ?width:0;
	  if(newWidth > curWidth)
	  {
	    pwWidths.at<float>(iter,jter) = newWidth;
	    pwWidths.at<float>(jter,iter) = newWidth;
	  }
	}
    }
    
    return pwWidths;
  }  
}
