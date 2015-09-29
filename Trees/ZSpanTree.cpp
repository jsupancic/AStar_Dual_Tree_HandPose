/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "ZSpanTree.hpp"
#include <opencv2/opencv.hpp>
#include "util.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace boost;
  
  GraphHash hashMST(const ImRGBZ&imsc, vector<Edge_Type> &mst)
  {
    GraphHash treeMap;
	for(int iter = 0; iter < mst.size(); iter++)
    {
	  Edge_Type&edge = mst[iter];
      int src_row, src_col; coordOf(imsc,src_row,src_col,edge.m_source);
      int dst_row, dst_col; coordOf(imsc,dst_row,dst_col,edge.m_target);
      treeMap[Vec4i(src_row,src_col,dst_row,dst_col)] = edge;
      treeMap[Vec4i(dst_row,dst_col,src_row,src_col)] = edge;
    }
    return treeMap;  
  }
  
  void pruneMST(const ImRGBZ& im, Mat zdt, vector< Edge_Type >& mst) 
  {
    vector<Edge_Type> keep;
    for(int iter = 0; iter < mst.size(); iter++)
    {
	  Edge_Type&edge = mst[iter];
      Vec2i p1, p2;
      coordOf(im,p1[0],p1[1],edge.m_source);
      coordOf(im,p2[0],p2[1],edge.m_target);
      
      // edges must not leave the image...
      if(p1[0] < 0 || p1[0] >= zdt.rows ||
	 p2[0] < 0 || p2[0] >= zdt.rows ||
	 p1[1] < 0 || p1[1] >= zdt.cols || 
	 p2[1] < 0 || p2[1] >= zdt.cols)
	continue;
      
      if(zdt.at<float>(p1[0],p1[1]) >= .5 && zdt.at<float>(p2[0],p2[1]) >= .5)
	keep.push_back(edge);
    }    
    mst = keep;
  }  
  
  vector< Edge_Type > calcMST(Mat values) 
  {
    printf("Computing a MST\n");
    // compute a MST over the depth image    
    // build the graph
    auto indexOf = [&values](int row, int col) -> int
    {
      return col + row * values.cols;
    };
    int N = values.rows*values.cols;
    vector<Edge_Connectivity_Type> edges;
    vector<double> weights;
    // connect each vertex to right and below is possible
    for(int row = 0; row < values.rows; row++)
      for(int col = 0; col < values.cols; col++)
      {
	// down
	if(row < values.rows - 1)
	{
	  float value1 = values.at<float>(row,col);
	  float value2 = values.at<float>(row+1,col);
	  edges.push_back(Edge_Connectivity_Type(indexOf(row,col),indexOf(row+1,col)));
	  weights.push_back(1/(std::max(value1,value2)));
	}
	// right
	if(col < values.cols - 1)
	{
	  float value1 = values.at<float>(row,col);
	  float value2 = values.at<float>(row,col+1);
	  edges.push_back(Edge_Connectivity_Type(indexOf(row,col),indexOf(row,col+1)));
	  weights.push_back(1/(std::max(value1,value2)));
	}
      }
    
    // allocate the graph and find MST...
    Graph_Type g(edges.begin(),edges.end(),weights.begin(),N);
    std::vector < Edge_Type > mst;
    kruskal_minimum_spanning_tree(g, std::back_inserter(mst));
    printf("Computed a MST!!!\n");
    return mst;
  }  
  
  Mat drawMST(const ImRGBZ& im, vector< Edge_Type >& mst, Scalar color, bool show) 
  {
    auto indexOf = [&im](int row, int col) -> int {return col + row * im.Z.cols; };
    auto coordOf = [&im,&indexOf](int&row, int&col, int idx) -> void 
    {
      row = idx / im.Z.cols;
      col = idx % im.Z.cols;
      assert(idx == indexOf(row,col));
    };
    
    double sf = 8;
    ImRGBZ bg = im.resize(sf);
    for(int iter = 0; iter < mst.size(); iter++)
    {
      Edge_Type&edge = mst[iter];
      Point2i p1, p2;
      coordOf(p1.y,p1.x,edge.m_source);
      coordOf(p2.y,p2.x,edge.m_target);
      p1.x = sf*p1.x + sf/2;
      p1.y = sf*p1.y + sf/2;
      p2.x = sf*p2.x + sf/2;
      p2.y = sf*p2.y + sf/2;
      line(bg.RGB,p1,p2,color);
    }
#ifdef DD_CXX11  
    if(show)
      image_safe("MST",bg.RGB);
#endif
	return bg.RGB;
  }
}

