/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_UTIL_GRAPH
#define DD_UTIL_GRAPH

#define use_speed_ 0
#include <opencv2/opencv.hpp>

#include <vector>

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  class Circle
  {
  private:
    float x,y,r;
  public:
    Circle(float x, float y, float r);
    float getX();
    float getY();
    float getR();
    bool operator<(const Circle&other) const;
    Point center() const;
  };
    
  Mat draw(Mat src, Circle circle);  
  
  vector<Circle> segment(const Mat&ZDTc); 
  
  // label connected_components with DFS
  template<typename T, typename I>
  Mat_<int> connected_components(const Mat src, I ignore = -1, int connectivity = 8)
  {
    Mat_<int> labels = Mat_<int>::zeros(src.size());
    int curLabel = 0;
    std::vector<Point2i> stack;
    
    for(int yIter = 0; yIter < src.rows; yIter++)
      for(int xIter = 0; xIter < src.cols; xIter++)
      {
	// prepare to do a DFS.
	if(labels.at<int>(yIter,xIter) > .5)
	  continue;
	bool ignoreValueSet = (src.at<T>(yIter,xIter) == ignore);
	if(ignore != -1 && ignoreValueSet)
	  continue;
	curLabel++;
	stack.push_back(Point2i(xIter,yIter));
	
	while(!stack.empty())
	{
	  Point2i p = stack.back();
	  stack.pop_back();
	  int x = p.x, y = p.y; 
	  labels.at<int>(y,x) = curLabel;
	  
// 	  cout << "connected_components: labeled (" << x << 
// 	    ", " << y << ") = " << curLabel << endl;
	  
	  for(int yNear = y - 1; yNear <= y + 1; yNear++)
	    for(int xNear = x - 1; xNear <= x + 1; xNear++)
	    {
	      bool match = 
		labels.at<int>(yNear,xNear) < .5 &&
		(src.at<T>(yNear,xNear) == src.at<T>(y,x));
	      bool valid = 
		!(xNear == x && yNear == y) && 
		xNear < src.cols 
		&& yNear < src.rows 
		&& xNear >= 0 && yNear >= 0;
	      bool valid4 = (yNear == y || xNear == x);
	      // neighbour coord in image
	      // and currently unlabeled.
	      // and has the same value in src
	      if(match && valid && (connectivity == 8 || valid4))
	      {
// 		cout << "scheduling neghbour: (" << xNear << 
// 		  ", " << yNear << ")" << endl;
		stack.push_back(Point2i(xNear,yNear));
	      }
	    }
	}
      }
    
    return labels;
  }
  
  struct Edge
  {
  public:
    int v1, v2;
    float weight;
    Edge(int v1, int v2, float weight)
    {
      this->v1 = v1;
      this->v2 = v2;
      this->weight = weight;
    }
    bool operator<(const Edge&other) const
    {
      return this->weight < other.weight;
    }
  };
  vector<Edge> MST_Kruskal(Mat G,float minEdgeWeight);  
  
  Mat_<float> pwWidths(const Mat&ZDT, const vector<Circle>&circles);
  Mat_<float> pwDists(const Mat&ZDT,const vector<Circle>&circles);
}

#endif
