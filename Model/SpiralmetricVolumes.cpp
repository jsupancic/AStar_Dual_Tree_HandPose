/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "SpiralmetricVolumes.hpp"
#include "Visualization.hpp"
#include "Video.hpp"
#include "TestModel.hpp"
#include "Colors.hpp"
#include "SphericalVolumes.hpp"
#include "HornAbsOri.hpp"

namespace deformable_depth
{
  const double a = 1;
  const double b = 1;
  const double lg_base = 1.09;

  static void test_spiral_volumetry_one(ImRGBZ&im)
  {
    cout << "++test_spiral_volumetry_one" << endl;
    Mat vis(2*1024,2*1280,DataType<Vec3b>::type,toScalar(INVALID_COLOR));
    Vec2d center(vis.rows/2,vis.cols/2);

    vector<Point> pts;
    for(double theta = 1; theta < 30*params::PI; theta += .05*params::PI)
    {
      Vec2d direc0(std::cos(theta),std::sin(theta));
      direc0 /= std::sqrt(direc0.ddot(direc0));
      
      double r = a * std::pow(lg_base,b * theta);
      Vec2d pt = center + r * direc0;
      cout << "theta = " << theta << " r = " << r << " pt = " << pt << endl;
      
      pts.push_back(Point(pt[0],pt[1]));

      //if(0 <= pt[0] && 0 <= pt[1] && pt[0] < vis.cols && pt[1] < vis.rows)
      //vis.at<Vec3b>(pt[1],pt[0]) = GREEN;
    }

    for(int iter = 0; iter < pts.size() - 1; ++iter)
    {
      cv::line(vis,pts.at(iter),pts.at(iter+1),toScalar(BLUE));
    }

    for(double theta = 0; theta <= 2*params::PI; theta += .05*params::PI)
    {
      Vec2d direc0(std::cos(theta),std::sin(theta));
      direc0 /= std::sqrt(direc0.ddot(direc0));
      Vec2d pt2 = center + vis.cols*direc0;

      cv::line(vis,Point(center[0],center[1]),Point(pt2[0],pt2[1]),toScalar(BLUE));
    }

    log_im("vis",vis);
    cout << "--test_spiral_volumetry_one" << endl;
  }

  static void test_spiral_surface()
  {
    Mat vis(2*1024,2*1280,DataType<Vec3b>::type,toScalar(INVALID_COLOR));
    Vec3d center(vis.rows/2,vis.cols/2,0);

    vector<Point3d> pts;
    vector<double> phis, thetas;
    auto spiral_fun = [&](double phi, double theta)
    {
      Vec3d direc0 = vecOf(phi,theta);
      direc0 /= std::sqrt(direc0.ddot(direc0));
      
      double r = a * std::pow(lg_base,b * (theta + phi));
      Vec3d pt = center + r * direc0;
      //cout << "theta = " << theta << " phi = " << phi << " r = " << r << " pt = " << pt << endl;
      return pt;
    };
    for(double phi = 1; phi < 30*params::PI; phi += .05*params::PI)
      for(double theta = 1; theta < 30*params::PI; theta += .05*params::PI)
      {	
	Vec3d pt = spiral_fun(phi,theta);
	
	pts.push_back(Point3d(pt[0],pt[1],pt[2]));
	phis.push_back(phi);
	thetas.push_back(theta);
	
	//if(0 <= pt[0] && 0 <= pt[1] && pt[0] < vis.cols && pt[1] < vis.rows)
	//vis.at<Vec3b>(pt[1],pt[0]) = GREEN;
      }

    auto samplePts = [&]()
    {      
      double phi1 = phis.at(thread_rand()%phis.size());
      double theta1 = thetas.at(thread_rand()%thetas.size());
      double phi2 = phi1 + sample_in_range(0,params::PI/8);
      double theta2 = theta1 + sample_in_range(0,params::PI/8);
      
      return vector<Vec3d>{
	spiral_fun(phi1,theta1),
	  spiral_fun(phi1,theta2),
	  spiral_fun(phi2,theta1),
	  spiral_fun(phi2,theta2),
	  spiral_fun(phi1+2*params::PI,theta1+2*params::PI),
	  spiral_fun(phi1+2*params::PI,theta2+2*params::PI),
	  spiral_fun(phi2+2*params::PI,theta1+2*params::PI),
	  spiral_fun(phi2+2*params::PI,theta2+2*params::PI)};
    };

    for(int iter = 0; iter < 10; ++iter)
    {
      vector<Vec3d> shape1 = samplePts();
      vector<Vec3d> shape2 = samplePts();
      AbsoluteOrientation abs_ori = distHornAO(shape1,shape2);
      cout << abs_ori.distance << endl;
    }
  }

  void test_spiral_volumetry()
  {
    cout << "++test_spiral_volumetry" << endl;
    //test_spiral_surface();
    //return;

    // test on all frames
    for(string video_file : test_video_filenames())
    {
      shared_ptr<Video> video = load_video(video_file);
      for(int iter = 0; iter < video->getNumberOfFrames(); ++iter)
      {
	if(video->is_frame_annotated(iter))
	{
	  shared_ptr<MetaData> datum = video->getFrame(iter,true);
	  shared_ptr<ImRGBZ> im = datum->load_im();
	  test_spiral_volumetry_one(*im);
	  return;
	}
      }
    }

    cout << "--test_spiral_volumetry" << endl;
  }
}


