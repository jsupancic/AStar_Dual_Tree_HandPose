/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "util_depth.hpp"
#include <opencv2/opencv.hpp>
#include "util_mat.hpp"
#include "params.hpp"
#include "util_real.hpp"
#include "util_rect.hpp"
#include "Log.hpp"
#include "util.hpp"
#include "util_vis.hpp"

namespace deformable_depth
{
  using namespace cv;

  Mat depth_float2uint16(const Mat&inDepth)
  {
    Mat Depth_uint16 = inDepth.clone(); 
    Depth_uint16 *= 10; // cm to mm
    Depth_uint16.convertTo(Depth_uint16,DataType<uint16_t>::type);    
    return Depth_uint16;
  }

#if CV_MAJOR_VERSION >= 3 || (CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4)   
  template<typename T>
  Mat fillHoles(const Mat Zin, Mat_<uchar> invalid, double max_fill_dist)
  {    
    //imagesc("Invalid",invalid);
    Mat dt, labels;
    
    // we use distance transform to indentify NN of each invalid pixel. 
    //printMatType(invalid.type());
    cv::distanceTransform(invalid, dt, labels, CV_DIST_L2, 5, cv::DIST_LABEL_PIXEL );
    assert(dt.type() == DataType<float>::type);
    //imagesc("dt",dt);
    //imagesc("labels",labels); //cvWaitKey(0);
    
    // PASS 1: Map labels to coords
    std::vector<T> valueForLabel
      (Zin.rows * Zin.cols + 1);
    for(int rIter = 0; rIter < Zin.rows; rIter++)
      for(int cIter = 0; cIter < Zin.cols; cIter++)
	// if valid pixel, valid value for component
	if(!invalid.at<uchar>(rIter,cIter))
	{
	  int label = labels.at<int>(rIter,cIter);
	  T value = Zin.at<T>(rIter,cIter);
	  valueForLabel[label] = value;
	}      
    
    // PASS 2: fill in nans with correct coors.
    // create and fill in the output
    //std::set<T> fill_values;
    Mat Zout = Zin.clone();
    for(int rIter = 0; rIter < Zout.rows; rIter++)
      for(int cIter = 0; cIter < Zout.cols; cIter++)
      {
	bool is_invalid = invalid.at<uchar>(rIter,cIter) > 0;
	if(is_invalid && dt.at<float>(rIter,cIter) <= max_fill_dist)
	{
	  int label = labels.at<int>(rIter,cIter);
	  T fill_value = valueForLabel[label];
	  //fill_values.insert(fill_value);
	  Zout.at<T>(rIter,cIter) = fill_value;
	}
	else if(is_invalid && dt.at<float>(rIter,cIter) > max_fill_dist)
	{
	  Zout.at<T>(rIter,cIter) = inf;
	}
      }
	
    //for(const T & fill_value : fill_values)
      //std::cout << "Using fill value: " << fill_value << std::endl;
	
    return Zout;     
  }    
  
  template
  Mat fillHoles<Vec2f>(const Mat Zin, Mat_<uchar> invalid, double max_fill_dist);
  template
  Mat fillHoles<float>(const Mat Zin, Mat_<uchar> invalid, double max_fill_dist);
  template
  Mat fillHoles<Vec3b>(const Mat Zin, Mat_<uchar> invalid, double max_fill_dist);  
#endif
  
#if CV_MAJOR_VERSION >= 3 || (CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4)   
  Mat_<float> fillDepthHoles(const Mat_<float> Zin, Mat_<uchar> invalid, Mat_<uchar> conf)
  {
    Mat_<float> Zconf = fillDepthHoles(Zin,inv(conf));
    Mat dt_conf; distanceTransform(inv(conf),dt_conf,CV_DIST_L2,CV_DIST_MASK_PRECISE);
    Mat_<float> Zvalid = fillDepthHoles(Zin,invalid);
    Mat_<float> Zvalblur; medianBlur(Zvalid,Zvalblur,5);
    Mat Zout = Zin.clone();
    
    for(int rIter = 0; rIter < Zin.rows; rIter++)
      for(int cIter = 0; cIter < Zin.cols; cIter++)
      {
	float depth_valid = Zvalid.at<float>(rIter,cIter);
	float depth_neighbour = Zvalblur.at<float>(rIter,cIter);
	if(dt_conf.at<float>(rIter,cIter) > 5)
	//if(depth_neighbour > 100)
	//if(depth_valid < 2.5*depth_neighbour)
	  Zout.at<float>(rIter,cIter) = Zvalid.at<float>(rIter,cIter);
	else
	  Zout.at<float>(rIter,cIter) = Zconf.at<float>(rIter,cIter);
      }
    
    //imagesc("inv(conf)",inv(conf));
    return Zout;
  }
  
  Mat_<float> fillDepthHoles(const Mat_<float> Zin, Mat_<uchar> invalid, double max_fill_dist)
  {    
    return fillHoles<float>(Zin,invalid,max_fill_dist);
  }
  
  Mat_<float> fillDepthHoles(const Mat_<float> Zin,double max_fill_dist)
  {
    // uint8 which is zero where 
    Mat invalid = isBad(Zin);
    cv::dilate(invalid,invalid,Mat::ones(3,3,DataType<float>::type));
    //imageeq("invalid",invalid,true,false);
    invalid.convertTo(invalid,cv::DataType<uchar>::type);
    invalid *= 255;
    //image_safe("InvalidRGB",invalid);
    return fillDepthHoles(Zin, invalid,max_fill_dist);
  }
#endif
  
  float Camera::areaAtDepth(float depth) const
  {
    return widthAtDepth(depth)*heightAtDepth(depth);
  }

  Size Camera::sizeAtDepth(float depth) const
  {
    return Size(widthAtDepth(depth),heightAtDepth(depth));
  }
    
  float Camera::pixAreaAtDepth(float depth) const
  {
    //cout << "metric_correction = " << metric_correction() << endl;
    return areaAtDepth(depth)/(vRes()*hRes());
  }
    
  double Camera::DistPixtoWorldCM(double depth, double pix_dist) const
  {
    float pix_area_cm2 = pixAreaAtDepth(depth);
    float pix_side_len_cm = std::sqrt(pix_area_cm2);
    return pix_dist * pix_side_len_cm;
  }
    
  float Camera::widthAtDepth(float depth) const
  {
    // old
    // return 2*depth*std::tan((hFov()/2)*params::PI/180);
    
    return distance_geodesic(Point2d(hRes()/2,0), depth, Point2d(hRes()/2,vRes()), depth);

    // new
    //return std::sqrt(metric_correction())*2*depth*hRes()/focalX();
    //   2*depth*hRes()/focalX() 
    // = 2*depth*hRes()/[.5*hRes()/tan(params::PI/180.0f*hFov()/2)]
    // = 4*depth*tan(params::PI/180.0f*hFov()/2)
  }
  
  float Camera::heightAtDepth(float depth) const
  {
    // old
    // return 2*depth*std::tan((vFov()/2)*params::PI/180);
    
    return distance_geodesic(Point2d(0,vRes()/2), depth, Point2d(hRes(),vRes()/2), depth);

    // by similar triangles
    // H/h = Z/f
    //return std::sqrt(metric_correction())*2*depth*vRes()/focalY();
  }

  float Camera::focalX() const
  {
    double fx = 2 * std::tan(0.5 * deg2rad(hFov())) ;
    assert(fx > 0);
    return fx;    
  }

  float Camera::focalY() const
  {
    //double fy = (.5*vRes()/tan(params::PI/180.0f*vFov()/2));
    double fy = 2 * std::tan(0.5 * deg2rad(vFov())) ;
    require_true<double>(fy > 0, vFov(), fy);
    return fy;    
  }
  
  float CustomCamera::focalX() const
  {
    //double fx = .5*hRes()/tan(params::PI/180.0f*hFov()/2);
    if(goodNumber(m_fx))
      return m_fx;
    else
    {
      return Camera::focalX();
    }
  }

  float CustomCamera::focalY() const
  {
    if(goodNumber(m_fy))
      return m_fy;
    else
    {
      return Camera::focalY();
    }
  }  
  
  float Camera::metric_correction() const
  {
    return 1;
  }
  
  Rect_<double> Camera::bbForDepth(float z, int rIter, int cIter, float width, float height) const
  {
    return bbForDepth(z,Size(hRes(),vRes()),rIter,cIter,width,height);
  }
  
  bool Camera::is_orhographic() const
  {
#ifdef WIN32
	return _isnan(vFov()) || _isnan(hFov());
#else
    return std::isnan(vFov()) || std::isnan(hFov());
#endif
  }
  
  float Camera::worldAreaForImageArea(float depth, cv::Rect_< float > bb) const
  {
    if(is_orhographic())
      return bb.area();
    else
    {
      float bb_ratio = bb.area()/(hRes()*vRes());
      return bb_ratio*areaAtDepth(depth);
    }    
  }

  Rect_<double> Camera::bbForDepth(
    float depth, Size im_size, int rIter, int cIter, float width, float height,
    bool clampp) const
  {
    // camera 
    float rel_width  = width/(widthAtDepth(depth));
    float rel_height = height/(heightAtDepth(depth));
    float pix_width = im_size.width*rel_width;
    float pix_height = im_size.height*rel_height;
    int x1 = cIter - pix_width/2;
    int y1 = rIter - pix_height/2;
    int x2 = cIter + pix_width/2;
    int y2 = rIter + pix_height/2;
    //printf("Sample = (%d, %d)  (%d, %d)\n",x1,y1,x2,y2);
    int x1p = clamp(0,x1,im_size.width-1);
    int y1p = clamp(0,y1,im_size.height-1);
    int x2p = clamp(0,x2,im_size.width-1);
    int y2p = clamp(0,y2,im_size.height-1);
    if(clampp)
      return Rect(Point2i(x1p,y1p),Point2i(x2p,y2p));
    else
      return Rect(Point2i(x1,y1),Point2i(x2,y2));
  }    
  
  Rect_<double> Camera::bbForDepth(Mat Z, int rIter, int cIter, float width, float height) const
  {
    return bbForDepth(Z.at<float>(rIter,cIter),Z.size(),rIter,cIter,width,height);
  }    
  
  /// SECTION: RGB and Depth versions of the above
  float RGBCamera::hFov() const
  {
    return params::H_RGB_FOV;
  }

  float RGBCamera::vFov() const
  {
    return params::V_RGB_FOV;
  }

  float RGBCamera::hRes() const
  {
    return params::hRes;
  }

  float RGBCamera::vRes() const
  {
    return params::vRes;
  }
  
  float DepthCamera::hFov() const
  {
    return params::H_Z_FOV;
  }

  float DepthCamera::vFov() const
  {
    return params::V_Z_FOV;
  }
  
  float DepthCamera::hRes() const
  {
    return params::depth_hRes;
  }

  float DepthCamera::vRes() const
  {
    return params::depth_vRes;
  }
  
  ///
  /// SECTION: Custom camera implementation
  ///
  
  CustomCamera::CustomCamera() : 
    m_hFov(qnan), m_vFov(qnan), m_hRes(qnan), m_vRes(qnan), m_metric_correction(1),
    m_fx(qnan), m_fy(qnan)
  {
  }
  
  CustomCamera::CustomCamera(float hFov, float vFov, float hRes, float vRes, 
			     float metric_correction, float fx, float fy) :
    m_hFov(hFov), m_vFov(vFov), m_hRes(hRes), m_vRes(vRes), 
    m_metric_correction(metric_correction),
    m_fx(fx), m_fy(fy)
  {
    assert(hFov > 0 or !goodNumber(hFov));
    assert(vFov > 0 or !goodNumber(vFov));
  }

  float CustomCamera::hFov()  const
  {
    return m_hFov;
  }

  float CustomCamera::vFov()  const
  {
    return m_vFov;
  }

  float CustomCamera::hRes()  const
  {
    return m_hRes;
  }

  float CustomCamera::vRes()  const
  {
    return m_vRes;
  }
  
  float CustomCamera::metric_correction() const
  {
      return m_metric_correction;
  }
  
  void CustomCamera::scaleFov(double hScale, double vScale)
  {
    double theta1 = 2*rad2deg(std::atan(hScale*std::tan(deg2rad(hFov())/2)));
    double phi1   = 2*rad2deg(std::atan(vScale*std::tan(deg2rad(vFov())/2)));
    
//     {
//       static mutex m; lock_guard<mutex> l(m);
//       log_file << "hscale = " << hScale << " vscale " << vScale << endl;
//       log_file << "hfov = " << hFov() << " vfov = " << vFov() << endl;
//       log_file << "theta = " << theta1 << " phi = " << phi1 << endl;
//     }
    
    setFov(theta1,phi1);
  }
  
  CustomCamera CustomCamera::crop(Rect_< double > roi) const
  {
    CustomCamera result;
    
    // update the resolution
    result.m_hRes = roi.width;
    result.m_vRes = roi.height;
    
    // update the fov
    result.m_hFov = m_hFov;
    result.m_vFov = m_vFov;
    result.scaleFov(result.m_hRes/m_hRes,result.m_vRes/m_vRes);
    
//     {
//       static mutex m; lock_guard<mutex> l(m);
//       log_file << printfpp("old res = %f %f",m_hRes,m_vRes) << endl;
//       log_file << printfpp("new res = %f %f",result.m_hRes,result.m_vRes) << endl;
//     }
//     
//     log_file << printfpp("camera fov %f %f => %f %f",
// 			 m_hFov,m_vFov,result.m_hFov,result.m_vFov) << endl;
    
    return result;
  }
  
  void CustomCamera::setFov(float hFov, float vFov)
  {
    this->m_hFov = hFov;
    this->m_vFov = vFov;
  }
  
  void CustomCamera::setRes(float hRes, float vRes)
  {
    this->m_hRes = hRes;
    this->m_vRes = vRes;
  }
  
  void CustomCamera::setMetricCorrection(float factor)
  {
    this->m_metric_correction = factor;
  }
  
  void read(const FileNode& node, CustomCamera& cam, CustomCamera )
  {
    node["hFov"] >> cam.m_hFov;
    node["vFov"] >> cam.m_vFov;
    node["hRes"] >> cam.m_hRes;
    node["vRes"] >> cam.m_vRes;
    if(!node["metric_correction"].empty())
      node["metric_correction"] >> cam.m_metric_correction;
  }

  void write(FileStorage& node, string& , const CustomCamera& cam)
  {
    node << "{";
    node << "hFov" << cam.hFov();
    node << "vFov" << cam.vFov();
    node << "hRes" << cam.hRes();
    node << "vRes" << cam.vRes();
    node << "metric_correction" << cam.metric_correction();
    node << "}";
  }

#ifdef DD_CXX11
  float medianApx(const Mat& Z, Rect bb, float order)
  {
    // setup
    bb = clamp(Z,bb);
    float min_z = params::MIN_Z();
    float max_z = params::MAX_Z();
    
    // init a RNG
    std::mt19937 sample_seq;
    sample_seq.seed(bb.x);
    std::uniform_int_distribution<int> x_dist(bb.x,bb.x+bb.width-1);
    std::uniform_int_distribution<int> y_dist(bb.y,bb.y+bb.height-1);   
    
    // sample some depths
    int samples = std::min<float>(20,bb.size().area());
    vector<float> depths;
    for(int iter = 0; iter < samples; iter++)
    {
      int sample_x = x_dist(sample_seq);
      int sample_y = y_dist(sample_seq);
      float sample_z = Z.at<float>(sample_y,sample_x);
      if(!goodNumber(sample_z) || sample_z <= min_z || sample_z >= max_z)
      {
	//iter--; // skip invalid samples, but don't resample to avoid infinite loop
	continue;
      }
      depths.push_back(sample_z);
    }
    
    if(depths.size() > 0)
    {
      std::sort(depths.begin(),depths.end());
      int index = order * depths.size();
      double apx_median = depths[clamp<int>(0,index,depths.size()-1)];
      return 
	clamp<float>(
	  params::MIN_Z(),
	  apx_median,
	  params::MAX_Z());
    }
    else
    {
      return qnan;
    }
  }
#endif

  Vec3d vecOf(double azimuth, double altitude)
  {
    double z = sin(altitude);
    double hyp = cos(altitude);
    double y = hyp*cos(azimuth);
    double x = hyp*sin(azimuth);
    return Vec3d(x,y,z);
  }

  double radius(const Vec3d&vec)
  {
    return std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  }

  double azimuth(const Vec3d&vec)
  {
    return std::atan(vec[1]/vec[0]);
  }

  double altitude(const Vec3d&vec)
  {
    return std::acos(vec[2]/radius(vec));
  }

  double Camera::depth_Cartesian_to_Spherical(double z,Point2d uv) const
  {
    Vec3d zenith = vecOf(0,0);
    double xIter = uv.x;
    double yIter = uv.y;

    double angle = distance_angular(Point2d(hRes()/2,vRes()/2),uv);

    return z / std::sin(deg2rad(90) - angle);
  }

  double Camera::depth_Spherical_to_Cartesian(double z,Point2d uv) const
  {
    double xIter = uv.x;
    double yIter = uv.y;
    
    double angle = distance_angular(Point2d(hRes()/2,vRes()/2),uv);
    
    return z * std::sin(deg2rad(90) - angle);
  }
  
  double Camera::distance_angular(Point2d image_xy1, Point2d image_xy2) const
  {
    double azimuth1 = deg2rad(interpolate_linear_prop(image_xy1.x,0,hRes(),-hFov()/2,+hFov()/2));
    double altitude1  = deg2rad(interpolate_linear_prop(image_xy1.y,0,vRes(),-vFov()/2,+vFov()/2));
    Vec3d dir1 = vecOf(azimuth1,altitude1);

    double azimuth2 = deg2rad(interpolate_linear_prop(image_xy2.x,0,hRes(),-hFov()/2,+hFov()/2));
    double altitude2  = deg2rad(interpolate_linear_prop(image_xy2.y,0,vRes(),-vFov()/2,+vFov()/2));
    Vec3d dir2 = vecOf(azimuth2,altitude2);

    return std::acos(dir1.ddot(dir2)/(std::sqrt(dir2.ddot(dir2))*std::sqrt(dir1.ddot(dir1))));
  }

  double Camera::distance_geodesic(Point2d image_xy1, double z1, Point2d image_xy2, double z2) const
  {
    // use law of cosines
    double a = depth_Cartesian_to_Spherical(z1,image_xy1);
    double b = depth_Cartesian_to_Spherical(z2,image_xy2);
    double theta = distance_angular(image_xy1,image_xy2);
    return std::sqrt(a*a + b*b - 2 * a * b * std::cos(theta));
  }

  Size Camera::imageSizeForMetricSize(float depth,Size metricSize) const
  {
    // (1) metric to angle @ depth
    // geo_dist = std::sqrt(2*d^2 - 2 * d^2 * std::cos(theta));
    // geo_dist^2 = 2*d^2 (1 - std::cos(theta))
    // geo_dist^2/2*d^2 = (1 - std::cos(theta))
    // std::cos(theta) = 1 - geo_dist^2/2*d^2w
    // theta = acos(1 - geo_dist^2/2*d^2))
    double theta = std::acos(1 - std::pow(metricSize.width ,2)/(2*depth*depth));
    double phi   = std::acos(1 - std::pow(metricSize.height,2)/(2*depth*depth));
    
    // (2) angle to image (generic computation)
    double im_width  = theta/deg2rad(hFov()) * hRes();
    double im_height = phi  /deg2rad(vFov()) * vRes();

    return Size(im_width,im_height);
  }
  
  std::string to_string( const Camera & camera )
  {
    return safe_printf(" [Camera hFov = % vFov = % hRes = % vRes = % w@d50 = % h@d50 = % mc = %] ",
		       camera.hFov(),camera.vFov(),
		       camera.hRes(),camera.vRes(),
		       camera.widthAtDepth(50),camera.heightAtDepth(50),
		       camera.metric_correction());
  }
}
