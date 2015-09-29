/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "TreeDecorators.hpp"
#include "SubTrees.hpp"
#include "Faces.hpp"
#include "util_file.hpp"
#include <fstream>
#include "Log.hpp"
#include "Skin.hpp"

namespace deformable_depth
{
  /**
   * Section: Centroid Decorator
   **/  
  CentroidDecorator::CentroidDecorator(const ImRGBZ& im, TreeStats& ts) :
    ts_for_area(ts)
  {
    double nan = numeric_limits<double>::quiet_NaN();
    for(Direction dir : card_dirs())
      centroids.push_back(Mat(im.rows(),im.cols(),DataType<Vec3d>::type,Scalar::all(nan)));
  }

  void CentroidDecorator::update_base(const StatsUpdateArgs args)
  {
    // this point has zero area and thus zero weight so it doesn't matter much...
    double x = args.here[1]; // col
    double y = args.here[0]; // row
    double z = args.z_here;
    centroids[args.dir].at<Vec3d>(args.here[0],args.here[1]) = Vec3d(x,y,z);
  }

  void CentroidDecorator::update_recur(const StatsUpdateArgs args)
  {
    double totalArea = 0.0;
    Vec3d  totalMass(0,0,0);
    
    // extrinsic
    for(Direction cur_dir : card_dirs())
      if(cur_dir != opp_dir(args.dir))
      {
	double cur_area = ts_for_area.area_for_dir(cur_dir,args.neighbour);
	totalArea += cur_area;
	totalMass += cur_area*centroids[cur_dir].at<Vec3d>(args.neighbour);
      }
    
    // intrinsic
    double x = args.here[1];
    double y = args.here[0];
    double z = args.z_here;
    double cur_area = args.im.camera.areaAtDepth(z);
    Vec3d cenHere(x,y,z);
    
    // combine and store
    totalArea += cur_area;
    totalMass += cur_area*cenHere;
    Vec3d finalCenter = totalMass/totalArea;
    centroids[args.dir].at<Vec3d>(args.here[0],args.here[1]) = finalCenter;
  }
  
  void CentroidDecorator::show()
  {
    // show nearest subtree at pixel
    Mat showme(centroids[0].rows,centroids[0].cols,DataType<float>::type,Scalar::all(inf));
    
    for(int rIter = 0; rIter < showme.rows; rIter++)
      for(int cIter = 0; cIter < showme.cols; cIter++)
	for(Direction dir : card_dirs())
	{
	  double z = centroids[dir].at<Vec3d>(rIter,cIter)[2];
	  if(showme.at<float>(rIter,cIter) > z)
	    showme.at<float>(rIter,cIter) = z;
	}
    
    imageeq("nearest subtree depth",showme);
  }
    
  Vec3d CentroidDecorator::backward(Vec2i pos, Direction dir)
  {
    //get backward centroid
    Vec3d cen_backward(0,0,0);
    double totalArea = 0;
    for(Direction cur_dir : card_dirs())
      if(cur_dir != opp_dir(dir))
      {
	double cur_area = ts_for_area.area_for_dir(cur_dir,pos);
	totalArea += cur_area;
	cen_backward += cur_area*centroids[cur_dir].at<Vec3d>(pos);
      }
    cen_backward /= totalArea;
    return cen_backward;
  }

  Vec3d CentroidDecorator::forward(Vec2i pos, Direction dir)
  {
    // get forward centroid
    Vec3d cen_forward = centroids[dir].at<Vec3d>(pos[0],pos[1]);
    return cen_forward;
  }
  
  Vec3d CentroidDecorator::fbDelta(Vec2i pos, Direction dir)
  {
    return forward(pos,dir) - backward(pos,dir);
  }	

  bool CentroidDecorator::isFrontal(Vec2i pos, Direction dir)
  {
    // compare and decide
    return forward(pos,dir)[2] < backward(pos,dir)[2]; // compare z coordinates.
  }
  
  /**
   * Section SupervisedDecorator
   **/
  bool SupervisedDecorator::predicate(Vec2i pos)
  {
    if(pos_mask.empty())
      return false;
    else
      return pos_mask.at<uchar>(pos[0],pos[1]) > 0;
  }
  
  SupervisedDecorator::SupervisedDecorator(cv::Mat& dt, cv::Mat pos_mask): 
    FilteredAreaDecorator(dt,PIXEL), pos_mask(pos_mask)
  {
  }
  
  /**
   * SECTION: SkinDecorator
   **/
  SkinDetectorDecorator::SkinDetectorDecorator(Mat RGB,TreeStats&ts_for_area) :
    ts_for_area(ts_for_area)
  {
    #pragma omp critical
    {
      Mat fgProb, bgProb;
      likelihood = skin_detect(RGB,fgProb,bgProb);
      
      // init the tree stat stores
      for(Direction cur_dir : card_dirs())
	meanProbLikes.push_back(
	  Mat(RGB.rows,RGB.cols,DataType<double>::type,
	      Scalar::all(numeric_limits<double>::quiet_NaN())));
    }
  }

  void SkinDetectorDecorator::update_base(const StatsUpdateArgs args)
  {
    // up against a wall
    meanProbLikes[args.dir].at<double>(args.here[0],args.here[1]) = 0;
  }

  void SkinDetectorDecorator::update_recur(const StatsUpdateArgs args)
  {
    // configure
    double&writeOut = meanProbLikes[args.dir].at<double>(args.here[0],args.here[1]);
    writeOut = 0;
    double total_area = 0;
    
    // intrinsic
    // weight neighbour by his area at depth
    total_area += args.im.camera.areaAtDepth(args.z_nb);
    writeOut += args.im.camera.areaAtDepth(args.z_nb)*
      likelihood.at<double>(args.neighbour);
    
    // extrinsic
    for(Direction cur_dir : card_dirs())
    {
      if(cur_dir == opp_dir(args.dir))
	continue;
      
      total_area += ts_for_area.area_for_dir(cur_dir,args.neighbour);
      writeOut += ts_for_area.area_for_dir(cur_dir,args.neighbour)*
	meanProbLikes[cur_dir].at<double>(args.neighbour);
    }
    
    writeOut /= total_area;
  }
  
  void SkinDetectorDecorator::show()
  {
    Mat showme(likelihood.rows,likelihood.cols,DataType<float>::type,
	       Scalar::all(-numeric_limits<float>::infinity()));
    
    for(int rIter = 0; rIter < showme.rows; rIter++)
      for(int cIter = 0; cIter < showme.cols; cIter++)
	for(Direction dir : card_dirs())
	  if(showme.at<float>(rIter,cIter) < meanProbLikes[dir].at<double>(rIter,cIter))
	    showme.at<float>(rIter,cIter) = meanProbLikes[dir].at<double>(rIter,cIter);
    
    imageeq("SkinDetectorDecorator",showme);
    waitKey_safe(1);
  }
  
  /**
   * SECTION: PixelAreaDecorator
   **/
  PixelAreaDecorator::PixelAreaDecorator(Mat& dt): 
    FilteredAreaDecorator(dt, PIXEL)
  {
    for(int rIter = 0; rIter < dt.rows; rIter++)
    {
      bbs.push_back(vector<vector<Rect> >());
      for(int cIter = 0; cIter < dt.cols; cIter++)
      {
	bbs[rIter].push_back(vector<Rect>());
	for(Direction cur_dir : card_dirs())
	  bbs[rIter][cIter].push_back(Rect(Point2i(cIter,rIter),Size(0,0)));
      }
    }
  }
  
  PixelAreaDecorator::PixelAreaDecorator()
  {
  }

  bool PixelAreaDecorator::predicate(Vec2i pos)
  {
    return true;
  }
  
  void PixelAreaDecorator::update_base(const StatsUpdateArgs args)
  {
    deformable_depth::FilteredAreaDecorator::update_base(args);
    
    // no additional processing required
  }

  void PixelAreaDecorator::update_recur(const StatsUpdateArgs args)
  {
    deformable_depth::FilteredAreaDecorator::update_recur(args);
    
    // 
    for(Direction cur_dir : card_dirs())
      if(cur_dir != opp_dir(args.dir))
	bbs[args.here[0]][args.here[1]][args.dir] |= 
	bbs[args.neighbour[0]][args.neighbour[1]][cur_dir];
  }

  void read(string filename,PixelAreaDecorator&paDec )
  {
    ifstream ifs(filename,ios::in | ios::binary);
    int rows; //ifs >> rows; 
    ifs.read(reinterpret_cast<char*>(&rows),sizeof(rows));
    int cols; //ifs >> cols;
    ifs.read(reinterpret_cast<char*>(&cols),sizeof(cols));
    log_once(printfpp("ReadPixAreaCache [%s] rows = %d cols = %d",
		      filename.c_str(),(int)rows,(int)cols));
    assert(rows <= 5000 && cols <= 5000); 
    
    // allcoate the array
    for(int yIter = 0; yIter < rows; yIter++)
    {
      paDec.bbs.push_back(vector<vector<Rect> >());
      for(int xIter = 0; xIter < cols; xIter++)
      {
	paDec.bbs[yIter].push_back(vector<Rect>());
	for(Direction cur_dir : card_dirs())
	{
	  Rect loaded_rect; 
	  //ifs >> loaded_rect;
	  ifs.read(reinterpret_cast<char*>(&loaded_rect),sizeof(loaded_rect));
	  paDec.bbs[yIter][xIter].push_back(loaded_rect);
	}
      }
    }
    
    ifs.close();
  }

  void write(string filename,PixelAreaDecorator&paDec)
  {
    ofstream ofs(filename,ios::out | ios::binary);
    //ofs << (int)paDec.bbs.size() << endl;
    //ofs << (int)paDec.bbs[0].size() << endl;
    int rows = (int)paDec.bbs.size(), cols = (int)paDec.bbs[0].size();
    ofs.write(reinterpret_cast<char*>(&rows),sizeof(rows));
    ofs.write(reinterpret_cast<char*>(&cols),sizeof(cols));
    assert(rows <= 5000 && cols <= 5000); 
    
    for(int yIter = 0; yIter < paDec.bbs.size(); yIter++)
      for(int xIter = 0; xIter < paDec.bbs[0].size(); xIter++)
	for(Direction cur_dir : card_dirs())
	{
	  //ofs << paDec.bbs[yIter][xIter][cur_dir] << endl;
	  Rect& data = paDec.bbs[yIter][xIter][cur_dir];
	  ofs.write(reinterpret_cast<char*>(&data),sizeof(data));
	}
    ofs.close();
  }
  
  /**
   * Section: Filtered Area Decorator
   **/
  FilteredAreaDecorator::FilteredAreaDecorator()
  {
  }
  
  FilteredAreaDecorator::FilteredAreaDecorator(cv::Mat& dt, 
					       deformable_depth::FilteredAreaDecorator::AreaType areaType) :
	areaType(areaType)
  {
    float nan = std::numeric_limits<float>::quiet_NaN();
    // init the mats
    for(Direction dir : card_dirs())
    {
      matching_areas.push_back(Mat(dt.rows,dt.cols,DataType<float>::type,Scalar::all(nan)));
    }
  }

  void FilteredAreaDecorator::update_base(const StatsUpdateArgs args)
  {
    // up against a wall
    matching_areas[args.dir].at<float>(args.here[0],args.here[1]) = 0;
  }

  void FilteredAreaDecorator::update_recur(const StatsUpdateArgs args)
  {
    // intrinsic area_face
    float intrinsic_fa = 0;
    /*DEBUG: Make this produce same result*///true || 
    if(predicate(args.neighbour))
    {
      float z = args.im.Z.at<float>(args.neighbour[0],args.neighbour[1]);
      intrinsic_fa = areaType==PIXEL?1:args.im.camera.pixAreaAtDepth(z);
    }
    
    // compute the extrinsic area.
    float extrinsic_fa = 0;
    for(Direction cur_dir : card_dirs())
      if(cur_dir != args.dir)
	extrinsic_fa += matching_areas[opp_dir(cur_dir)].at<float>(args.neighbour[0],args.neighbour[1]);
    
    matching_areas[args.dir].at<float>(args.here[0],args.here[1]) = extrinsic_fa + intrinsic_fa;
  }
  
  /**
   * Section: Face Area Decorator
   **/  
  FaceAreaDecorator::FaceAreaDecorator(const ImRGBZ& im, 
				       cv::Mat& dt, 
				       GraphHash& treeMap) :
	FilteredAreaDecorator(dt)
  {    
    // compute which pixels are in a face
    facep = Mat(dt.rows,dt.cols,DataType<uchar>::type,Scalar::all(0));
    vector<Rect> face_dets = FaceDetector().detect(im);
    for(const Rect& det : face_dets)
      rectangle(facep,det.tl(),det.br(),Scalar::all(255),-1/*fill*/);
  }
  
  bool FaceAreaDecorator::predicate(Vec2i pos)
  {
    //assert(facep.type() == DataType<uchar>::type);
    return facep.at<uchar>(pos[0],pos[1]);
  }
  
  void FaceAreaDecorator::show(vector<Mat>&areas)
  {
    float nan = numeric_limits<float>::quiet_NaN();
    Mat showMe(matching_areas[0].rows,matching_areas[0].cols,DataType<float>::type,Scalar::all(nan));
    
    for(int rIter = 0; rIter < showMe.rows; rIter++)
      for(int cIter = 0; cIter < showMe.cols; cIter++)
      {
	float max_p_area = 0;
	float max_area = 0;
	for(Direction dir : card_dirs())
	{
	  float area_face = matching_areas[dir].at<float>(rIter,cIter);
	  float area_raw  = areas[dir].at<float>(rIter,cIter);
	  float p_area = area_face/area_raw;
	  if(p_area > max_p_area)
	    max_p_area = p_area;
	  if(area_face > max_area)
	    max_area = area_face;
	}
	
	//printf("max_p_area = %f\n",max_p_area);
	showMe.at<float>(rIter,cIter) = max_p_area; //max_area;
      }
    
    image_safe("facep",facep);
    imageeq("max area face",showMe);
  }
  
  /**
   * Section: Edge Type Decorator
   **/  
  void EdgeTypeDecorator::show(Mat active)
  {
    Mat showMe(active.rows,active.cols,DataType<Vec3b>::type,Scalar::all(0));
    for(int rIter = 0; rIter < showMe.rows; rIter++)
      for(int cIter = 0; cIter < showMe.cols; cIter++)
      {
	Vec3i ets = 0;
	//for(Direction dir : card_dirs())
	  //ets += et_for_dir(dir,Vec2i(rIter,cIter));
	ets += et_for_dir(UP,Vec2i(rIter,cIter));
	
	Vec3b&pixel = showMe.at<Vec3b>(rIter,cIter);
	
	bool all0 = true;
	for(Direction dir : card_dirs())
	{
	  // if any valid
	  Vec3i etd = et_for_dir(dir,Vec2i(rIter,cIter));
	  if(etd[0] > etd[2])
	    pixel = Vec3b(0,255,0);
	  if(etd[0] > 0)
	    all0 = false;
	  
	  assert(etd[0] != -1);
	  assert(etd[2] != -1);
	}
	
	if(all0)
	  pixel = Vec3b(0,0,255); // RED
	
	// 	  // blue
// 	  if(ets[0] > ets[1] && ets[0] > ets[2])
// 	    pixel = Vec3b(255,0,0);
// 	  // green
// 	  else if(ets[1] > ets[2])
// 	    pixel = Vec3b(0,255,0);
// 	  // red
// 	  else
// 	    pixel = Vec3b(0,0,255);
      }
    image_safe("ets - up",showMe);
  }
  
  EdgeTypeDecorator::EdgeTypeDecorator(Mat& dt, GraphHash& treeMap) : 
    // edge types 
    etLeft(dt.rows, dt.cols, DataType<Vec3i>::type, Scalar::all(-1)),
    etRight(dt.rows, dt.cols, DataType<Vec3i>::type, Scalar::all(-1)),
    etUp(dt.rows, dt.cols, DataType<Vec3i>::type, Scalar::all(-1)),
    etDown(dt.rows, dt.cols, DataType<Vec3i>::type, Scalar::all(-1))
  {
  }
  
  Vec3i EdgeTypeDecorator::extrinsic_ET(int row, int col, Direction dir)
  {
    Vec3i ans(0,0,0);
    
    if(dir != LEFT)
    {
      ans += etRight.at<Vec3i>(row,col);
    }
    if(dir != RIGHT)
    {
      ans += etLeft.at<Vec3i>(row,col);
    }
    if(dir != UP)
    {
      ans += etDown.at<Vec3i>(row,col);
    }
    if(dir != DOWN)
    {
      ans += etUp.at<Vec3i>(row,col);
    }
    
    bool valid = ans[0] != -1 && ans[1] != -1 && ans[2] != -1;
    if(!valid)
    {
      cout << row <<  " " << col << endl;
      cout << ans << endl;
    }
    assert(valid);
    assert(ans != Vec3i(0,0,0));
    return ans;
  } 
  
  Vec3i& EdgeTypeDecorator::et_for_dir(Direction dir, Vec2i pos)
  {
    int row = pos[0], col = pos[1];
    
    if(dir == RIGHT)
      return etRight.at<Vec3i>(row,col);
    else if(dir == LEFT)
      return etLeft.at<Vec3i>(row,col);
    else if(dir == UP)
      return etUp.at<Vec3i>(row,col);
    else if(dir == DOWN)
      return etDown.at<Vec3i>(row,col);
    else
      throw std::exception();            
  }  
  
  void EdgeTypeDecorator::update_base(const StatsUpdateArgs args)
  {
    if(args.true_edge)
    {
      // edge type
      if(args.oob_edge)
	et_for_dir(args.dir,args.here) = Vec3i(0,0,1);
      else if(args.z_here > args.z_far)
	et_for_dir(args.dir,args.here) = Vec3i(0,0,1);
      else
	et_for_dir(args.dir,args.here) = Vec3i(1,0,0);        
    }
    else
    {
      et_for_dir(args.dir,args.here) = Vec3i(0,1,0);      
    }
  }

  void EdgeTypeDecorator::update_recur(const StatsUpdateArgs args)
  {
    Vec3i et_hist = extrinsic_ET(args.neighbour[0],args.neighbour[1],args.dir);
    et_for_dir(args.dir,args.here) = et_hist;
    assert(et_for_dir(args.dir,args.here) == et_hist);
  }  
}
