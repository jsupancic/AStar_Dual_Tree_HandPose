/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_ZSPANTREE
#define DD_ZSPANTREE

#include <functional>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <opencv2/opencv.hpp>
#include <unordered_map>

#include "util_mat.hpp"

namespace deformable_depth
{
  using boost::adjacency_list;
  using boost::graph_traits;
  using boost::vecS;
  using boost::undirectedS;
  using boost::property;
  using boost::no_property;
  using boost::edge_weight_t;
  using namespace std;
  using namespace cv;
  
  struct hash_fn : public unary_function<Vec4i,size_t>
  {
    size_t operator() (const Vec4i&value) const
    {
      long int sum = value[0]+value[1]+value[2]+value[3];
      return boost::hash<long int>()(sum);
    }
  };
  
  // Boost is very flexible, I need to choose simple defaults... we are simplified :-)
  typedef std::pair<int,int> Edge_Connectivity_Type;
  typedef adjacency_list<vecS,vecS,undirectedS,no_property,property<edge_weight_t,double > > Graph_Type;
  typedef graph_traits<Graph_Type>::edge_descriptor Edge_Type;
  typedef graph_traits<Graph_Type>::vertex_descriptor Vertex_Type;
  typedef std::unordered_map<Vec4i,Edge_Type,hash_fn > GraphHash;  
  
  // functions we export
  void pruneMST(const ImRGBZ& im, Mat zdt, vector< Edge_Type >& mst);
  vector< Edge_Type > calcMST(Mat values);
  Mat drawMST(const ImRGBZ& im, 
	      std::vector< deformable_depth::Edge_Type >& mst, 
	      cv::Scalar color = Scalar(0,255,0), bool show = true);
  GraphHash hashMST(const ImRGBZ&imsc, vector<Edge_Type> &edges);
}

#endif


