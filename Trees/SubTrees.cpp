/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "SubTrees.hpp"
#include "Faces.hpp"
#include "Segment.hpp"
#include "Log.hpp"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include "Cache.hpp"

namespace deformable_depth
{      
  /**
   * Section: TreeStats
   * 	This class computes and contains statistics for an entire spanning forest.
   **/
  TreeStats::TreeStats(const ImRGBZ& im, cv::Mat& dt, GraphHash& treeMap, cv::Mat pos_mask) : 
    // store the image
    im(im),
    // areas
    areaLeft(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    areaRight(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    areaUp(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    areaDown(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    // parimeters
    perimLeft(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    perimRight(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    perimUp(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    perimDown(dt.rows, dt.cols, DataType<float>::type, Scalar::all((float)nan)),
    // graph structure
    treeMap(treeMap),
    // ET
    etDecorator(dt,treeMap),
    // Face Areas
    faDecorator(im,dt,treeMap),
    // centroids
    centroidDecorator(im,*this),
    // supervision
    supDecorator(dt,pos_mask),
    // area in pixel coordinates (not world coordinates)
    pixelAreaDecorator(dt),
    // skin detection scores
    skinDetectorDecorator(im.RGB,*this),
    // dt
    dt(dt)
  {         
    // for this task, I'll be using DFS.
    for(int rIter = 0; rIter < dt.rows; rIter++)
      for(int cIter = 0; cIter < dt.cols; cIter++)
      {
	//printf("DFS from %d %d\n",rIter,cIter);
	if(!allSet(rIter,cIter))
	{
	  active = Mat(dt.rows,dt.cols,DataType<uchar>::type,Scalar::all(0));
	  compute_DFS(rIter,cIter,treeMap,im);
	  //assert(stats.allSet(rIter,cIter));
	}
      }
      
    // attempt some visulizations...
    //show();
  };
      
  void TreeStats::compute_DFS(const int row, const int col, GraphHash& treeMap, const ImRGBZ& im) 
  {
    assert(row >= 0 && col >= 0 && row < im.Z.rows && col < im.Z.cols);
    
    Vec2i here(row,col);
    active.at<uchar>(row,col) = 1;
    
    /// Stage 1: Fill in as much data as we can using DFS
    auto fillNeighbour = [this,&treeMap,&here,&im](Vec2i neighbour,TreeStats::Direction dir) -> void
    {
      // we can only follow extant edges
      if(treeMap.find(Vec4i(here[0],here[1],neighbour[0],neighbour[1])) == treeMap.end())
	return;
      
      // don't loop back on ourselves or we'll never terminate (infinite loop, oh my!)
      if(active.at<uchar>(neighbour[0],neighbour[1]))
	return;
      
      // finally check, that there is some info to be gained by visiting the node (it
      // hasn't been fully completed via a prior iteration)
      if(allSet(neighbour[0],neighbour[1],dir))
	return;
      
      // otherwise, go ahead and visit the node
      compute_DFS(neighbour[0],neighbour[1],treeMap,im);
    };
	
    /// Stage 2: use this data to fill in as much info about ourself as possible.
    /// we are only certain to fill in all data for the root node. Thus, this remains
    /// linear time.

    
    //printf("Updating %d %d\n",row,col);
    if(std::isnan(areaLeft.at<float>(row,col)))
    {
      fillNeighbour(Vec2i(row,col-1),LEFT);
      updateStats(here,im,LEFT);
    }
    if(std::isnan(areaRight.at<float>(row,col)))
    {
      fillNeighbour(Vec2i(row,col+1),RIGHT);
      updateStats(here,im,RIGHT);
    }
    if(std::isnan(areaUp.at<float>(row,col)))
    {
      fillNeighbour(Vec2i(row-1,col),UP);
      updateStats(here,im,UP);
    }
    if(std::isnan(areaDown.at<float>(row,col)))
    {
      fillNeighbour(Vec2i(row+1,col),DOWN);
      updateStats(here,im,DOWN);
    }
    
    active.at<uchar>(row,col) = 0;
  }    
	
  bool TreeStats::allSet(int row, int col, Direction dir)
  {
    bool set_left = !std::isnan(areaLeft.at<float>(row,col));
    bool set_right = !std::isnan(areaRight.at<float>(row,col));
    bool set_up = !std::isnan(areaUp.at<float>(row,col));
    bool set_down = !std::isnan(areaDown.at<float>(row,col));
    
    return 
      (set_left || dir == RIGHT) && 
      (set_right || dir == LEFT) && 
      (set_up || dir == DOWN) && 
      (set_down || dir == UP);
  }
  
  float TreeStats::extrinsicArea(int row, int col, Direction dir)
  {
    if(row < 0 || col < 0 || row >= areaRight.rows || col >= areaRight.cols)
      return 0;
    
    float ans = 0;
    
    if(dir != LEFT)
      ans += areaRight.at<float>(row,col);
    if(dir != RIGHT)
      ans += areaLeft.at<float>(row,col);
    if(dir != UP)
      ans += areaDown.at<float>(row,col);
    if(dir != DOWN)
      ans += areaUp.at<float>(row,col);
    
    if(std::isnan(ans))
    {
      printf("extrinsicArea at %d %d: assertion failure\n",row,col);
      printf("Right = %f\n",areaRight.at<float>(row,col));
      printf("Left = %f\n",areaLeft.at<float>(row,col));
      printf("Up = %f\n",areaUp.at<float>(row,col));
      printf("Down = %f\n",areaDown.at<float>(row,col));
      assert(!std::isnan(ans));
    }
    return ans;
  }
  
  float TreeStats::extrinsicPerim(int row, int col, Direction dir)
  {
    float ans = 0;
    
    if(dir != LEFT)
      ans += perimRight.at<float>(row,col);
    if(dir != RIGHT)
      ans += perimLeft.at<float>(row,col);
    if(dir != UP)
      ans += perimDown.at<float>(row,col);
    if(dir != DOWN)
      ans += perimUp.at<float>(row,col);
    
    return ans;
  }
      
  float& TreeStats::area_for_dir(Direction dir, Vec2i pos)
  {
    int row = pos[0], col = pos[1];
    
    if(dir == RIGHT)
      return areaRight.at<float>(row,col);
    else if(dir == LEFT)
      return areaLeft.at<float>(row,col);
    else if(dir == UP)
      return areaUp.at<float>(row,col);
    else if(dir == DOWN)
      return areaDown.at<float>(row,col);
    else
      throw std::exception();
  }
  
  float& TreeStats::perim_for_dir(Direction dir, Vec2i pos)
  {
    int row = pos[0], col = pos[1];
    
    if(dir == RIGHT)
      return perimRight.at<float>(row,col);
    else if(dir == LEFT)
      return perimLeft.at<float>(row,col);
    else if(dir == UP)
      return perimUp.at<float>(row,col);
    else if(dir == DOWN)
      return perimDown.at<float>(row,col);
    else
      throw std::exception();      
  }  
  
  void TreeStats::show(int delay)
  {
    //imageeq("left",areaLeft);
    //imageeq("right",areaRight);
    //imageeq("up",areaUp);
    //imageeq("down",areaDown);
    
    //imageeq("left",perimLeft);
    //imageeq("right",perimRight);
    //imageeq("up",perimUp);
    //imageeq("down",perimDown);
    
    vector<Mat> areas = {areaLeft,areaRight,areaUp,areaDown};
    
    //etDecorator.show(active);
    //faDecorator.show(areas);
    centroidDecorator.show();
    skinDetectorDecorator.show();
    
    //waitKey_safe(delay);      
  }      
  
  void TreeStats::updateStats(Vec2i&here, const ImRGBZ&im, TreeStats::Direction dir)
  {
    Vec2i vdir = dir2vec(dir);
    Vec2i neighbour = here + vdir;
    
    auto oobP = [&im](Vec2i p) -> bool
    {
      return p[0] < 0 || p[1] < 0 || 
	p[0] >= im.RGB.rows || p[1] >= im.RGB.cols;	
    };
    
    // read the depth for the position.
    bool neighbour_oob = oobP(neighbour);
    bool true_edge = 
      neighbour_oob || 
      dt.at<float>(neighbour[0],neighbour[1]) <= 0;// ||
      //dt.at<float>(here[0],here[1]) <= 0;
    float z = !neighbour_oob?im.Z.at<float>(neighbour[0],neighbour[1]):0;
    bool far_oob = oobP(Vec2i(here[0]+4*vdir[0],here[1]+4*vdir[1]));
    float z_far = far_oob?z:im.Z.at<float>(here[0]+4*vdir[0],here[1]+4*vdir[1]);
    float z_here = im.Z.at<float>(here[0],here[1]);
    float z_mu = (z+z_here)/2;
    
    // three cases
    // (1) no edge in this direction
    if(treeMap.find(Vec4i(here[0],here[1],neighbour[0],neighbour[1])) == treeMap.end())
    {
      // up against a wall!
      // area 
      area_for_dir(dir,here) = 0;
      
      // absense of a tree edge implies
      // (A) Edge in the image
      // (B) false edge
      StatsUpdateArgs args(im,true_edge,neighbour_oob,z_here,z,z_far,dir,here,neighbour);
      etDecorator.update_base(args);
      faDecorator.update_base(args);
      centroidDecorator.update_base(args);
      supDecorator.update_base(args);
      pixelAreaDecorator.update_base(args);
      skinDetectorDecorator.update_base(args);
      if(true_edge)
      {
	// parim
	if(dir == TreeStats::Direction::LEFT || dir == TreeStats::Direction::RIGHT)
	  perim_for_dir(dir,here) = im.camera.heightAtDepth(z_here);
	else
	  perim_for_dir(dir,here) = im.camera.widthAtDepth(z_here);
      }
      else
      {
	perim_for_dir(dir,here) = 0;
      }
      
      return;
    }
    
    // (2) the direction is active (thus missing information)
    if(active.at<uchar>(neighbour[0],neighbour[1]))
      return;
    
    // (3) we have all the info we require.
    // area
    area_for_dir(dir,here) = 
      extrinsicArea(neighbour[0],neighbour[1],dir) + im.camera.pixAreaAtDepth(z);
    // parim
    perim_for_dir(dir,here) = 
      extrinsicPerim(neighbour[0],neighbour[1],dir);
    // edge type
    StatsUpdateArgs args(im,true_edge,neighbour_oob,z_here,z,z_far,dir,here,neighbour);
    etDecorator.update_recur(args);
    faDecorator.update_recur(args);
    centroidDecorator.update_recur(args);
    supDecorator.update_recur(args);
    pixelAreaDecorator.update_recur(args);
    skinDetectorDecorator.update_recur(args);
    assert(!std::isnan(area_for_dir(dir,here)));
  };    
  
  const int D = 8;
  
  /**
   * Compute the feature for one tree.
   **/
  Mat TreeStats::feature(int rIter, int cIter, TreeStats::Direction dir)
  {
    // These are the elemental features.
    float agg_area  = area_for_dir(dir,Vec2i(rIter,cIter));
    float face_area = faDecorator.matching_areas[dir].at<float>(rIter,cIter);
    float perim     = perim_for_dir(dir,Vec2i(rIter,cIter));
    Vec3i et        = etDecorator.et_for_dir(dir,Vec2i(rIter,cIter));
    Vec3d centroid  = centroidDecorator.centroids[dir].at<Vec3d>(rIter,cIter);
    float skin_lik  = skinDetectorDecorator.meanProbLikes[dir].at<double>(rIter,cIter);
    
    // compute the derivative features
    float percent_face = (agg_area<=0)?0:face_area/agg_area;
    Vec3d centroid_offset = centroidDecorator.fbDelta(Vec2i(rIter,cIter),dir);    
    // the centroid offset can take nan values in case of zero area trees.
    centroid_offset[0] = std::isnan(centroid_offset[0])?0:centroid_offset[0];
    centroid_offset[1] = std::isnan(centroid_offset[1])?0:centroid_offset[1];
    centroid_offset[2] = std::isnan(centroid_offset[2])?0:centroid_offset[2];
    // pfg = percent foreground
    double percent_fg = et[0]/static_cast<double>(et[0]+et[2]);
    if(et[0]+et[2] == 0) percent_fg = numeric_limits<double>::infinity();
    
    // check for nans
    assert(!std::isnan(face_area));
    assert(!std::isnan(perim));
    assert(!std::isnan(agg_area));
    assert(!std::isnan(percent_face));
    assert(!std::isnan(centroid_offset[0]));
    assert(!std::isnan(centroid_offset[1]));
    assert(!std::isnan(centroid_offset[2]));
    assert(!std::isnan(percent_fg));  
    
    // write the feature
    int idx = 0;
    Mat x(1,D,DataType<float>::type,Scalar::all(0));
    x.at<float>(idx++) = agg_area;
    x.at<float>(idx++) = percent_face;
    x.at<float>(idx++) = perim;
    x.at<float>(idx++) = percent_fg;
    x.at<float>(idx++) = centroid_offset[0];
    x.at<float>(idx++) = centroid_offset[1];
    x.at<float>(idx++) = centroid_offset[2];
    x.at<float>(idx++) = skin_lik;
    
    assert(!any(isnan(x)));
    
    return x;
  }
  
  void TreeStats::features_negative(Mat& X, Mat& Y, Rect gt_bb, Root pos)
  {
    Mat not_negative(dt.rows,dt.cols,DataType<uchar>::type,Scalar::all(0));
    
    for(int rIter = 0; rIter < dt.rows; rIter++)
      for(int cIter = 0; cIter < dt.cols; cIter++)
	for(Direction dir : card_dirs())
	{
	  Rect tree_bb = pixelAreaDecorator.bbs[rIter][cIter][dir];  
	  Rect true_bb = pixelAreaDecorator.bbs[pos.row][pos.col][pos.dir];
	  float overlap = rectIntersect(true_bb,tree_bb);
	  if(overlap >= .60)
	  {
	    mark_tree_region(rIter,cIter,dir,not_negative);
	    continue;
	  }
	  
	  Mat y(1,1,DataType<float>::type,Scalar::all(-1.0f));
	  Mat x = feature(rIter,cIter,dir);
	  X.push_back<float>(x);
	  Y.push_back<float>(y);
	}
	
    image_safe("not_negative",255*not_negative);
  }

  Mat TreeStats::features_positive(Mat& X, Mat& Y, Rect gt_bb, Root&best_root)
  {
    // find the best root...
    float best_overlap = -numeric_limits<float>::infinity();
    for(int rIter = 0; rIter < dt.rows; rIter++)
      for(int cIter = 0; cIter < dt.cols; cIter++)
	for(Direction dir : card_dirs())
	{
	  // label
	  Rect  tree_bb      = pixelAreaDecorator.bbs[rIter][cIter][dir];  
	  Vec3i et           = etDecorator.et_for_dir(dir,Vec2i(rIter,cIter));
	  //float overlap      = pos_area/(pixel_area+pos_pixel_area-pos_area);
	  float overlap      = rectIntersect(gt_bb,tree_bb);
	  //printf("pos_area = %f\n",pos_area);
	  //printf("pos_pixel_area = %f\n",pos_pixel_area);
	  //printf("percent_pos = %f\n",percent_pos);
	 
	  if(overlap > best_overlap && et[0] > et[2])
	  {
	    best_overlap = overlap;
	    Root root{rIter,cIter,dir};
	    best_root = root;
	  }
	}
	
    // write the feature
    Mat y(1,1,DataType<float>::type,Scalar::all(+1.0f));
    Mat x = feature(best_root.row,best_root.col,best_root.dir);
    X.push_back<float>(x);
    Y.push_back<float>(y);    
    
    // update the visualization
    Mat extracted_positives(dt.rows,dt.cols,DataType<uchar>::type,Scalar::all(0));
    mark_tree_region(best_root.row,best_root.col,best_root.dir,extracted_positives);
    return extracted_positives;
  }
  
  /**
   * Compute all the features in the image.
   **/
  Mat TreeStats::features(Mat& X, Mat& Y, Rect gt_bb)
  {
    X = Mat(0,D,DataType<float>::type);
    Y = Mat(0,1,DataType<float>::type);
    
    Root best_root;
    Mat extracted_positives = features_positive(X,Y,gt_bb,best_root);
    features_negative(X,Y,gt_bb,best_root);

    return extracted_positives;
  }
  
  static void showTree_DFS (int row, int col,const TreeStats&stats,Mat &visited)
  {
    visited.at<uchar>(row,col) = 1;
    
    bool edgeLeft = stats.treeMap.find(Vec4i(row,col,row,col-1)) != stats.treeMap.end();
    bool edgeRight = stats.treeMap.find(Vec4i(row,col,row,col+1)) != stats.treeMap.end();
    bool edgeUp = stats.treeMap.find(Vec4i(row,col,row-1,col)) != stats.treeMap.end();
    bool edgeDown = stats.treeMap.find(Vec4i(row,col,row+1,col)) != stats.treeMap.end();
    
    if(	(edgeLeft && col - 1 < 0) || 
	(edgeRight && col + 1 >= visited.cols) || 
	(edgeUp && row - 1 < 0) ||
	(edgeDown && row + 1 >= visited.rows))
    {
      printf("edgeLeft = %d\n",(int)edgeLeft);
      printf("edgeRight = %d\n",(int)edgeRight);
      printf("edgeUp = %d\n",(int)edgeUp);
      printf("edgeDown = %d\n",(int)edgeDown);
      printf("visited.size = %d %d\n",visited.rows, visited.cols);
      printf("row = %d col = %d\n",row,col);     
      assert(false);
    }
    
    if(!visited.at<uchar>(row,col-1) && edgeLeft)
      showTree_DFS(row,col-1,stats,visited);
    if(!visited.at<uchar>(row,col+1) && edgeRight)
      showTree_DFS(row,col+1,stats,visited);
    if(!visited.at<uchar>(row-1,col) && edgeUp)
      showTree_DFS(row-1,col,stats,visited);
    if(!visited.at<uchar>(row+1,col) && edgeDown)
      showTree_DFS(row+1,col,stats,visited);
  };  
  
  void TreeStats::mark_tree_region(int rIter, int cIter, int dir_raw, cv::Mat& visited) const
  {
    TreeStats::Direction dir((Direction)dir_raw);
    if(visited.empty())
      visited = Mat(dt.rows,dt.cols,DataType<uchar>::type,Scalar::all(0));
    visited.at<uchar>(rIter,cIter) = 1;
        
    if(dir == LEFT && cIter - 1 >= 0)
      showTree_DFS(rIter,cIter-1,*this,visited);
    else if(dir == RIGHT && cIter + 1 < dt.cols)
      showTree_DFS(rIter,cIter+1,*this,visited);
    else if(dir == UP && rIter -1 >= 0)
      showTree_DFS(rIter-1,cIter,*this,visited);
    else if(dir == DOWN && rIter + 1 < dt.rows)
      showTree_DFS(rIter+1,cIter,*this,visited);
  }
  
  void TreeStats::drawTree(const ImRGBZ& im, cv::Mat& visited, std::string name) const
  {
    Mat showme = im.RGB.clone();
    for(int rIter = 0; rIter < showme.rows; rIter++)
      for(int cIter = 0; cIter < showme.cols; cIter++)
      {
	if(visited.at<uchar>(rIter,cIter))
	{
	  Vec3b&col = showme.at<Vec3b>(rIter,cIter);
	  col[1] = 255;
	}
      }
    float sf = std::sqrt((648*480.0f)/showme.size().area());
    resize(showme,showme,Size(),sf,sf);
    image_safe(name,showme);
  }
  
  void TreeStats::showTree(const ImRGBZ& im, 
			   int rIter, int cIter, int dir_raw, Mat&visited, bool show) const
  {
    mark_tree_region(rIter,cIter,dir_raw,visited);
    
    if(show)
      drawTree(im,visited);
  }  
  
  TreeStats treeStatsOfImage(const ImRGBZ&im, Rect BB)
  {
    // the positive format correction
    Mat posMask(im.rows(),im.cols(),DataType<uchar>::type,Scalar::all(0));
    rectangle(posMask,BB.tl(),BB.br(),Scalar::all(255),CV_FILLED/*fill*/);
    
    // Segment the imnage
    Mat zdt = ZDT(im.Z);
    vector<Edge_Type> mst = calcMST(zdt);
    pruneMST(im,zdt,mst);
    log_im("PZPS",vertCat(imageeq("",im.Z,false,false),drawMST(im,mst)));
    GraphHash treeMap = hashMST(im,mst);
    
    // compute the TS
    if(BB == Rect())
      return TreeStats(im,zdt,treeMap);
    else
      return TreeStats(im,zdt,treeMap,posMask);
  }
  
  TreeStats&treeStatsOfImage_cached(const ImRGBZ&im)
  {
    static Cache<shared_ptr<TreeStats> > cache;
    return *cache.get(im.filename,[&]()
    {
      return make_shared<TreeStats>(treeStatsOfImage(im));
    });
    log_once(printfpp("subtree cache size = %d",(int)cache.size()));
  }
  
  static PixelAreaDecorator pixelAreas_disk_cached(const ImRGBZ&im)
  {
    string cache_file = string("cache/pix_areas_") + hash(vector<string>{im.filename}) + ".bin";
    static mutex m; 
    
    // Try to load critical section
    {
      unique_lock<mutex> l(m);
      if(boost::filesystem::exists(cache_file))
      {
	l.unlock();
	log_once(printfpp("loaded %s from cache",im.filename.c_str()));
	PixelAreaDecorator pixelAreaDecorator; 
	read(cache_file,pixelAreaDecorator);
	return pixelAreaDecorator;
      }
    }
    
    // compute the tree stats
    TreeStats&stats = treeStatsOfImage_cached(im);
    
    // write the cache critical section
    {
      unique_lock<mutex> l(m);
      if(!boost::filesystem::exists(cache_file))
	write(cache_file,stats.pixelAreaDecorator);
      return stats.pixelAreaDecorator;
    }
  }  
  
  PixelAreaDecorator& pixelAreas_cached(const ImRGBZ&im)
  {
    static map<string/*filename*/,PixelAreaDecorator> cache;
    static mutex m; 
    
    {
      // make this a critical section
      unique_lock<mutex> l(m);
      if(cache.find(im.filename) != cache.end())
	return cache.at(im.filename);
    }
    
    PixelAreaDecorator pixelAreaDecorator = pixelAreas_disk_cached(im);
    
    {
      // make this a critical section
      unique_lock<mutex> l(m);
      if(cache.find(im.filename) == cache.end())
	cache.insert(std::pair<string,PixelAreaDecorator>(im.filename,pixelAreaDecorator));
      log_once(printfpp("pixelArea cache size = %d",(int)cache.size()));
      return cache.at(im.filename);
    }    
  }
}

