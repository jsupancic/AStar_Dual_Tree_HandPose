/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Colors.hpp"
#include "SphericalVolumes.hpp"
#include "Orthography.hpp"
#include "Visualization.hpp"
#include "Video.hpp"
#include "TestModel.hpp"
#include "util_mat.hpp"

namespace deformable_depth
{
  SphericalOccupancyMap::SphericalOccupancyMap(const ImRGBZ&im)
  {
    camera = im.camera;    
    //om = depthToPointCloud(im);
    //om = sphericalToCartesian(im.Z,im.camera);
    om = cartesianToSpherical(im.Z,im.camera);
    om = im.Z.clone();

    log_im("convert",horizCat(imageeq("",im.Z,false,false),imageeq("",om,false,false)));
  }

  Mat SphericalOccupancyMap::get_OM() const
  {
    return om.clone();
  }
  
  int VIS_RES_ROWS = 2*1024;
  int VIS_RES_COLS = 2*1280;

  void drawCell(Mat&yProj,double t0, double t1, double z1, double z2,Scalar color)
  {
    //cout << safe_printf("drawing (% %)r (% %)d",t0,t1,z1,z2) << endl;
    
    // get the direction
    Vec2d src(yProj.cols/2,yProj.rows);
    t0 -= params::PI/2;
    t1 -= params::PI/2;
    Vec2d direc0(std::cos(t0),std::sin(t0));
    direc0 /= std::sqrt(direc0.ddot(direc0));
    Vec2d direc1(std::cos(t1),std::sin(t1));
    direc1 /= std::sqrt(direc1.ddot(direc1));
      
    // compute the corners.
    Vec2d tl = src + z2*direc0;
    Vec2d bl = src + z1*direc0;
    Vec2d tr = src + z2*direc1;
    Vec2d br = src + z1*direc1;
    vector<Point> pts{Point(tl[0],tl[1]),Point(bl[0],bl[1]),Point(br[0],br[1]),Point(tr[0],tr[1])};
    //cout << safe_printf("corners = % %",tl,br) << endl;

    // draw lines
    //Scalar color = toScalar(getColor(colorIndex++));      
    if(color == Scalar::all(-1))
    {
      cv::line(yProj,Point(tl[0],tl[1]),Point(bl[0],bl[1]),toScalar(GREEN));
      cv::line(yProj,Point(tr[0],tr[1]),Point(br[0],br[1]),toScalar(GREEN));
      cv::line(yProj,Point(tl[0],tl[1]),Point(tr[0],tr[1]),toScalar(GREEN));
      cv::line(yProj,Point(bl[0],bl[1]),Point(br[0],br[1]),toScalar(GREEN));
    }
    else
    {
      cv::fillConvexPoly(yProj,&pts[0],pts.size(),color);
    }
  }
        
  void SphericalOccupancyMap::drawCellInterp(Mat&yProj,const CustomCamera&camera, double xIter, double zIter, Scalar color, double xStride, double zStride) const
  {
    double x0 = xIter;
    double x1 = xIter + xStride;
    double t_min = deg2rad(+camera.hFov()/2);
    double t_max = deg2rad(-camera.hFov()/2);
    double t0 = interpolate_linear_prop(x0,0,xSize(),t_min,t_max);
    double t1 = interpolate_linear_prop(x1,0,xSize(),t_min,t_max);
    double z0 = zIter;
    double z1 = zIter + zStride;
    double d0 = interpolate_linear_prop(z0,0,zSize(),0,yProj.rows);
    double d1 = interpolate_linear_prop(z1,0,zSize(),0,yProj.rows);
	
    drawCell(yProj,t0,t1, d0, d1, color);
  }

  Mat SphericalOccupancyMap::proj_y() const
  {
    Mat yProj(VIS_RES_ROWS,VIS_RES_COLS,DataType<Vec3b>::type,toScalar(INVALID_COLOR));
    double X_STRIDE = 1;
    double Z_STRIDE = .25;
    static int colorIndex = 0;
    
    // draw the data
    for(double xIter = 0; xIter < xSize() - X_STRIDE; xIter += X_STRIDE)
      for(double zIter = 0; zIter < zSize() - Z_STRIDE; zIter += Z_STRIDE)
      {
	double yDist = (int)ySize() - 1;
	for(int yIter = (int)ySize() - 1; yIter >= 0; yIter--)
	{
	  if(om.at<float>(yIter,xIter) < zIter)
	    yDist = std::min<double>(yIter,yDist);
	}
	double y_rate = yDist/ySize();
	//cout << "y_rate = " << y_rate << endl;
	drawCellInterp(yProj,camera,xIter,zIter,(y_rate)*Scalar(255,255,255),X_STRIDE,Z_STRIDE);
      }

    // draw the grid
    for(double xIter = 0; xIter < (int)xSize() - 10; xIter += 10)
      for(double zIter = 0; zIter < (int)zSize() - 1; zIter += 1)
      {
	drawCellInterp(yProj,camera,xIter,zIter,Scalar::all(-1),10,1);
      }

    return yProj;
  }

  Mat SphericalOccupancyMap::proj_x() const
  {
    Mat xProj(VIS_RES_COLS,VIS_RES_ROWS,DataType<Vec3b>::type,toScalar(INVALID_COLOR));
    double Y_STRIDE = 1;
    double Z_STRIDE = .25;
    static int colorIndex = 0;
    
    // draw the data
    for(double yIter = 0; yIter < ySize() - Y_STRIDE; yIter += Y_STRIDE)
      for(double zIter = 0; zIter < zSize() - Z_STRIDE; zIter += Z_STRIDE)
      {
	double xDist = (int)xSize() - 1;
	for(int xIter = (int)xSize() - 1; xIter >= 0; xIter--)
	{
	  if(om.at<float>(yIter,xIter) < zIter)
	    xDist = std::min<double>(xIter,xDist);
	}
	double x_rate = xDist/xSize();
	//cout << "y_rate = " << y_rate << endl;
	drawCellInterp(xProj,camera,yIter,zIter,(x_rate)*Scalar(255,255,255),Y_STRIDE,Z_STRIDE);
      }

    // draw the grid
    for(double yIter = 0; yIter < (int)ySize() - 10; yIter += 10)
      for(double zIter = 0; zIter < (int)zSize() - 1; zIter += 1)
      {
	drawCellInterp(xProj,camera,yIter,zIter,Scalar::all(-1),10,1);
      }

    return xProj.t();
  }

  Mat SphericalOccupancyMap::proj_z() const
  {
    Mat zvis = imageeq("",om,false,false);
    cv::resize(zvis,zvis,Size(VIS_RES_COLS,VIS_RES_ROWS));
    return zvis;
  }

  Visualization SphericalOccupancyMap::vis() const
  {
    Visualization vis;
    vis.insert(imVGA(proj_y(),INTER_LANCZOS4),"SphericalOccupancyMap_proj_y");
    vis.insert(imVGA(proj_z(),INTER_LANCZOS4),"SphericalOccupancyMap_proj_z");
    vis.insert(imVGA(proj_x(),INTER_LANCZOS4),"SphericalOccupancyMap_proj_x");
    return vis;
  }
    
  size_t SphericalOccupancyMap::xSize() const
  {
    return om.cols;
  }
  
  size_t SphericalOccupancyMap::ySize() const
  {
    return om.rows;
  }
  
  size_t SphericalOccupancyMap::zSize() const
  {
    return params::MAX_Z();
  }

  void test_spherical_volumetry_one(ImRGBZ&im)
  {
    // Do the test for the image
    RectLinearOccupancyMap rom(im);	  
    SphericalOccupancyMap som(im);
    Mat deptheq = imageeq("",im.Z,false,false);
    log_im("som",horizCat(deptheq,horizCat(som.proj_y(),rom.proj_y())));
  }

  void test_spherical_volumetry()
  {
    // generate a special test image
    if(false)
    {
      Mat Z(480,640,DataType<float>::type,Scalar::all(inf));
      cv::line(Z,Point(120,300),Point(520,300),Scalar::all(16));
      cv::line(Z,Point(220,200),Point(420,200),Scalar::all(32));
      cv::line(Z,Point(370,100),Point(270,100),Scalar::all(64));
      cv::line(Z,Point(345,50),Point(295,50),Scalar::all(128));
      Mat RGB = imageeq("",Z,false,false);
      CustomCamera camera(74,58,640,480);
      ImRGBZ im(RGB,Z,uuid(),camera);
      test_spherical_volumetry_one(im);
    }

    // test on all frames
    TaskBlock test_all_frames("test_all_frames");
    for(string video_file : test_video_filenames())
    {
      shared_ptr<Video> video = load_video(video_file);
      for(int iter = 0; iter < video->getNumberOfFrames(); ++iter)
      {
	if(video->is_frame_annotated(iter))
	{
	  test_all_frames.add_callee([&,video,iter]
				     {
				       shared_ptr<MetaData> datum = video->getFrame(iter,true);
				       shared_ptr<ImRGBZ> im = datum->load_im();
				       test_spherical_volumetry_one(*im);
				     });
	  //return;
	}
      }
    }
    test_all_frames.execute();
  }

  Mat SphericalOccupancyMap::slice_z(float z_min, float z_max) const
  {
    float z_top = z_max - z_min;

    Mat slice = om.clone();
    for(int yIter = 0; yIter < slice.rows; ++yIter)
      for(int xIter = 0; xIter < slice.cols; ++xIter)
      {
	float&z = slice.at<float>(yIter,xIter);	
	//z = clamp<float>(0,z - z_min,z_top);
	z = z - z_min;
	if(z < 0)
	  z = -inf;
	else if(z > z_top)
	  z = inf;
      }

    return slice;
  }
}

