/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "util_mat.hpp"
#include "vec.hpp"
#include "util_rect.hpp"
#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include "math.h"
#include <boost/concept_check.hpp>
#include <cmath>
#include "util.hpp"
#include "Log.hpp"
#include "Skin.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "Colors.hpp"
#include "Kahan.hpp"

#ifdef WIN32
#define scalar_isnan _isnan
#else
#define scalar_isnan std::isnan
#endif

namespace deformable_depth
{
  using namespace cv;
  
  double angle(const Vec3d& v1, const Vec3d& v2)
  {
    return rad2deg(std::acos(v1.ddot(v2)/(std::sqrt(v1.ddot(v1)*v2.ddot(v2)))));
  }
  
  Vec3d unhomo4(const Mat& m)
  {
    assert(m.type() == DataType<double>::type);
    double x = m.at<double>(0);
    double y = m.at<double>(1);
    double z = m.at<double>(2);
    double w = m.at<double>(3);
    return Vec3d(x/w,y/w,z/w);
  }
  
  bool any(const Mat src)
  {
    for(int rIter = 0; rIter < src.rows; rIter++)
      for(int cIter = 0; cIter < src.cols; cIter++)
      {
	float val = src.at<float>(rIter,cIter);
	if(val != 0)
	  return true;
      }
    
    return false;
  }
  
  void minmax(Mat_<float> image, float&min, float&max)
  {
    float inf = numeric_limits<float>::infinity();
    float ninf = -inf;
    min = numeric_limits<float>::max();
    max = numeric_limits<float>::min();
    for(int xIter = 0; xIter < image.cols; xIter++)
      for(int yIter = 0; yIter < image.rows; yIter++)
      {
	float v = image.at<float>(yIter,xIter);
	if(v != inf && v != ninf && !scalar_isnan(v))
	{
	  if(v > max)
	    max = v;
	  if(v < min)
	    min = v;
	}
      }    
  }  
  
  float min(Mat_<float> image)
  {
    float min, max;
    minmax(image,min,max);    
    return min;
  }
  
  float max(Mat_<float> image)
  {
    float min, max;
    minmax(image,min,max);
    return max;
  }
  
  Mat max(Mat m1, Mat m2)
  {
    assert(m1.type() == DataType<float>::type);
    assert(m2.type() == DataType<float>::type);
    Mat r = m1.clone();
    
    for(int rIter = 0; rIter < r.rows; rIter++)
      for(int cIter = 0; cIter < r.cols; cIter++)
      {
	r.at<float>(rIter,cIter) =  
	  std::max<float>(
	    m1.at<float>(rIter,cIter),
	    m2.at<float>(rIter,cIter)
	  );
      }
    
    return r;
  }
  
  Mat min(Mat m1, Mat m2)
  {
    assert(m1.type() == DataType<float>::type);
    assert(m2.type() == DataType<float>::type);
    Mat r = m1.clone();
    
    for(int rIter = 0; rIter < r.rows; rIter++)
      for(int cIter = 0; cIter < r.cols; cIter++)
      {
	r.at<float>(rIter,cIter) =  
	  std::min<float>(
	    m1.at<float>(rIter,cIter),
	    m2.at<float>(rIter,cIter)
	  );
      }
    
    return r;
  }  
  
  Extrema extrema(const Mat& im)
  {
    Extrema result;
    cv::minMaxLoc(im,
		  &result.min,
		  &result.max,
		  &result.minLoc,
		  &result.maxLoc);
    return result;
  }  
  
  Mat_<float> isnan(const Mat_<float> src)
  {
    Mat_<float> dst = src.clone();
    
    for(int xIter = 0; xIter < dst.cols; xIter++)
      for(int yIter = 0; yIter < dst.rows; yIter++)
	dst.at<float>(yIter,xIter) = ::scalar_isnan(src.at<float>(yIter,xIter));
    
    return dst;
  }
  
  Mat_< float > isBad(const Mat_< float > src)
  {
    Mat_<float> dst = src.clone();
    
    for(int xIter = 0; xIter < dst.cols; xIter++)
      for(int yIter = 0; yIter < dst.rows; yIter++)
	dst.at<float>(yIter,xIter) = !goodNumber(src.at<float>(yIter,xIter));
    
    return dst;
  }
  
  Mat_<float> sqrt(const Mat_<float> src)
  {
    Mat_<float> dst = src.clone();
    
    for(int xIter = 0; xIter < dst.cols; xIter++)
      for(int yIter = 0; yIter < dst.rows; yIter++)
	dst.at<float>(yIter,xIter) = ::sqrt(src.at<float>(yIter,xIter));
    
    return dst;
  }  
  
  Mat_<float> pow(const Mat_<float> src, float p)
  {
    Mat_<float> dst = src.clone();
    
    for(int xIter = 0; xIter < dst.cols; xIter++)
      for(int yIter = 0; yIter < dst.rows; yIter++)
	dst.at<float>(yIter,xIter) = ::pow(src.at<float>(yIter,xIter),p);
    
    return dst;
  }
  
  Mat log(const Mat&in)
  {
    Mat out = in.clone();
    cv::log(out,out);
    return out;
  }
  
  void printSize(Size sz)
  {
    cout << " sz = [" << sz.height << ", " << sz.width << "] ";
  }
  
  void printMatType(int type)
  {
    switch(type)
    {
      case DataType<float>::type:
	printf("MatrixType = float\n");
      case DataType<double>::type:
	printf("MatrixType = double\n");
      case DataType<char>::type:
	printf("MatrixType = char\n");
      default:
	printf("MatrixType = %d\n",type);
	break;
    }
  }
  
  float matScalar(Mat_<float> m)
  {
    assert(m.type() == DataType<float>::type);
    bool unary = (m.rows == 1 && m.cols == 1);
    if(!unary)
    {
      printf("rows = %d cols = %d\n",m.rows,m.cols);
      assert(unary);
    }
    float scalar = m.at<float>(0);
    return scalar;
  }
  
  Mat affine_lr_flip(int width)
  {
    Mat affine(2,3,DataType<float>::type,Scalar::all(0));
    affine.at<float>(0,0) = -1;
    affine.at<float>(0,2) = width;
    affine.at<float>(1,1) = 1;
    return affine;
  }
  
  Mat affine_transform(float tx, float ty, float theta_rad)
  {
    Mat affine(2,3,DataType<float>::type,Scalar::all(0));
    
    affine.at<float>(0,0) = std::cos(theta_rad);
    affine.at<float>(1,1) = std::cos(theta_rad);
    affine.at<float>(0,1) = std::sin(theta_rad);
    affine.at<float>(1,0) = -std::sin(theta_rad);
    affine.at<float>(0,2) = tx;
    affine.at<float>(1,2) = ty;
    
    return affine;
  }
  
  Mat affine_translation(float tx, float ty)
  {
    Mat affine(2,3,DataType<float>::type,Scalar::all(0));
    affine.at<float>(0,0) = 1.0;
    affine.at<float>(1,1) = 1.0;
    affine.at<float>(0,2) = tx;
    affine.at<float>(1,2) = ty;
    return affine;
  }
  
  Mat affine_compose(Mat affine1, Mat affine2)
  {
    affine1.convertTo(affine1,DataType<float>::type);
    affine2.convertTo(affine2,DataType<float>::type);
    assert(affine1.type() == DataType<float>::type);
    assert(affine2.type() == DataType<float>::type);

    Mat last_row(1,3,DataType<float>::type,Scalar::all(0));
    last_row.at<float>(0,2) = 1;
    
    affine1.push_back<float>(last_row);
    affine2.push_back<float>(last_row);
    
    assert(affine1.size() == Size(3,3));
    assert(affine2.size() == Size(3,3));
    
    Mat mult = affine1*affine2;
    Mat result = mult.rowRange(0,2);
    //cout << printfpp("%d %d",result.rows,result.cols) << endl;
    assert(result.size() == Size(3,2));
    return result;
  }
  
  Mat affine_identity()
  {
    Rect r0(Point(0,0),Size(1,1));
    return affine_transform(r0,r0);
  }
  
  bool goodNumber(float n)
  {
    if(n == numeric_limits<float>::infinity())
      return false;
    if(n == -numeric_limits<float>::infinity())
      return false;
    if(::scalar_isnan(n))
      return false;
    return true;
  }  
    
  Mat inv(Mat m)
  {
    Mat r; threshold(m,r,0,255,THRESH_BINARY_INV);
    return r;
  }  
  
  ///
  /// SECTION: ImRGBZ implementation
  /// 
  ImRGBZ::ImRGBZ(const ImRGBZ& copy, bool clone)
  {
    camera = copy.camera;
    affine = copy.affine.clone();
    filename = copy.filename;    
    valid_region = copy.valid_region;
    
    if(!clone)
    {
      Z = copy.Z;
      RGB = copy.RGB;
    }
    else
    {
      Z = copy.Z.clone();
      RGB = copy.RGB.clone();      
    }
  }
  
  ImRGBZ& ImRGBZ::operator=(const ImRGBZ& copy)
  {
    Z = copy.Z.clone();
    RGB = copy.RGB.clone();
    camera = copy.camera;
    affine = copy.affine.clone();
    filename = copy.filename;
    valid_region = copy.valid_region;
    
    //cout << "Z.size() : " << Z.size() << endl;
    //cout << "copy.Z.size() : " << copy.Z.size() << endl;
    assert(Z.size() == copy.Z.size());
    
    return *this;
  }
  
  ImRGBZ ImRGBZ::resize(double sf, bool skip_image_data) const
  {
    return resize(Size(RGB.size().width*sf,RGB.size().height*sf),skip_image_data);
  }
  
  ImRGBZ ImRGBZ::resize(Size sz, bool skip_image_data) const
  {
    double sx = sz.width/(double)RGB.size().width;
    double sy = sz.height/(double)RGB.size().height;
    
    assert(sz.area() > 0);
    Mat RGB_imScale; 
    Mat Z_imScale; 
    if(!skip_image_data)
    {
      cv::resize(RGB,RGB_imScale,sz);
      cv::resize(Z,Z_imScale,sz);
    }
    else
    {//?
    }
    ImRGBZ result = ImRGBZ(RGB_imScale,Z_imScale,filename,camera); 
    Mat scale_transform = affine_transform(Rect(Point(0,0),RGB.size()),
				      Rect(Point(0,0),result.RGB.size()));
    result.affine = affine_compose(scale_transform, affine);
    return result;
  }

  Mat imrotate(const Mat&m, double angle_in_radians, Point2f center)
  {
    // handle default value
    if(center == Point2f())
      center = Point2f(m.cols/2,m.rows/2);

    Mat rotMat = cv::getRotationMatrix2D(center,rad2deg(angle_in_radians),1);    
    Mat rotated;    
    if(m.type() == DataType<float>::type)
    {
      //cv::warpAffine(m,rotated,rotMat,m.size(),params::DEPTH_INTER_STRATEGY,cv::BORDER_REPLICATE);
      cv::warpAffine(m,rotated,rotMat,m.size(),params::DEPTH_INTER_STRATEGY,cv::BORDER_CONSTANT,Scalar::all(qnan));
    }
    else
      cv::warpAffine(m,rotated,rotMat,m.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    return rotated;
  }

  Mat imrotate_tight(const Mat&m, double angle_in_radians)
  {
    // get the bounding rect
    Rect handBB(Point(0,0),m.size());
    RotatedRect rr(rectCenter(handBB),handBB.size(),rad2deg(angle_in_radians));
    Rect bounding_rect = rr.boundingRect();
    Rect pad_rect = bounding_rect | handBB;
    Mat m_pad = impad(m,pad_rect.size());
    Mat m_rot = imrotate(m_pad,angle_in_radians);

    // update bounding rect
    bounding_rect = rectFromCenter(Point(m_rot.cols/2,m_rot.rows/2),bounding_rect.size());
    return m_rot(clamp(m_rot,bounding_rect));
  }
  
  ImRGBZ ImRGBZ::roi_light(const RotatedRect&roi) const
  {
    // rotate
    Mat rgb_rot;
    Mat z_rot;
    Mat rotMat = cv::getRotationMatrix2D(roi.center,roi.angle,1); 
    if(roi.angle != 0)
    {      
      rgb_rot = imrotate(RGB,deg2rad(roi.angle),roi.center);
      z_rot   = imrotate(Z,deg2rad(roi.angle),roi.center);      
    }
    else
    {
      rgb_rot = RGB;
      z_rot = Z;
    }

    // crop
    RotatedRect unrot = roi; unrot.angle = 0;
    Rect rot_roi = unrot.boundingRect();
    Mat rgb_crop = crop_and_copy<Vec3b>(rgb_rot,rot_roi);
    Mat z_crop   = crop_and_copy<float>(z_rot,rot_roi);
    ImRGBZ result = ImRGBZ(rgb_crop,z_crop,filename,camera);
    //result.skin_cache.reset(new Mat());
    //this->skin();
    //*result.skin_cache = (*this->skin_cache)(roi);

    // update the FOV...
    float hScale = ((float)rot_roi.width)/RGB.cols;
    float vScale = ((float)rot_roi.height)/RGB.rows;
    result.camera.scaleFov(hScale,vScale);
    result.affine = affine_compose(rotMat, affine);
    result.affine = affine_compose(affine_transform(Rect(Point(0,0),RGB.size()),rot_roi), affine);
    
    assert(result.RGB.type() == DataType<Vec3b>::type);
    return result;        
  }

  ImRGBZ ImRGBZ::roi_light(const Rect& roi) const
  {
    RotatedRect roi_rot(rectCenter(roi),roi.size(),0);
    return roi_light(roi_rot);
  }
  
  ImRGBZ ImRGBZ::operator()(const Rect& roi) const
  {
    assert(RGB.type() == DataType<Vec3b>::type);
    Mat rgb_crop = crop_and_copy<Vec3b>(RGB,roi);
    Mat z_crop   = crop_and_copy<float>(Z,roi);
    assert(rgb_crop.type() == DataType<Vec3b>::type);
    ImRGBZ result = ImRGBZ(rgb_crop,z_crop,filename,camera);
    // update the FOV...
    float hScale = ((float)roi.width)/RGB.cols;
    float vScale = ((float)roi.height)/RGB.rows;
    result.camera.scaleFov(hScale,vScale);
    result.affine = affine_compose(affine_transform(Rect(Point(0,0),RGB.size()),roi), affine);
    
    assert(result.RGB.type() == DataType<Vec3b>::type);
    return result;
  }
  
  ImRGBZ ImRGBZ::flipLR() const
  {
    ImRGBZ result(*this);
    
    Mat flip_transform = affine_lr_flip(cols());
    warpAffine(result.RGB,result.RGB,flip_transform,result.RGB.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    warpAffine(result.Z,result.Z,flip_transform,result.Z.size(),params::DEPTH_INTER_STRATEGY,cv::BORDER_REPLICATE);
    result.affine = affine_compose(flip_transform, affine);
    result.filename = filename + "flip";
    return result;
  }
  
  int ImRGBZ::cols() const
  {
    return RGB.cols;
  }

  int ImRGBZ::rows() const
  {
    return RGB.rows;
  }
  
  static double integrate(const Mat&integral_image, const Rect&raw_roi)
  {
    Rect roi = clamp(integral_image,raw_roi);
    assert(integral_image.type() == DataType<double>::type);
    double A = (integral_image).at<double>(roi.tl().y,roi.tl().x);
    double B = (integral_image).at<double>(roi.tl().y,roi.br().x);
    double C = (integral_image).at<double>(roi.br().y,roi.br().x);
    double D = (integral_image).at<double>(roi.br().y,roi.tl().x);    
    
    return (C + A - B - D)/roi.area();
  }
  
  double ImRGBZ::mean_depth(Rect roi) const
  {
    if(!depth_integral)
    {
      static mutex m; lock_guard<mutex> l(m);
      if(!depth_integral)
      {
	Mat int_im; cv::integral(Z,int_im, CV_64F);
	depth_integral.reset(new Mat(int_im));
      }
    }
    
    return integrate(*depth_integral,roi);
  }
  
  double ImRGBZ::mean_intensity(Rect roi) const
  {
    if(!intensity_integral)
    {
      static mutex m; lock_guard<mutex> l(m);
      if(!intensity_integral)
      {
	Mat int_im; cv::integral(gray(),int_im, CV_64F);
	intensity_integral.reset(new Mat(int_im));
      }
    }
    
    return integrate(*intensity_integral,roi);
  }
  
  double ImRGBZ::skin_ratio(Rect roi) const
  {
#ifdef DD_CXX11
    if(!skin_integral || !skin_fg_integral || !skin_bg_integral || !skin_dets_integral)
    {
      lock_guard<mutex> l(skin_integral_mutex);
      if(!skin_integral || !skin_fg_integral || !skin_bg_integral || !skin_dets_integral)
      {
	// detect the skin
	Mat FGProbs, BGProbs;
	Mat skin_likelihoods = skin_detect(RGB, FGProbs, BGProbs);
	//skin_likelihoods.convertTo(skin_likelihoods,DataType<double>::type);
	assert(skin_likelihoods.type() == DataType<double>::type);

	Mat skin_dets(skin_likelihoods.rows,skin_likelihoods.cols,DataType<double>::type);
	for(int rIter = 0; rIter < skin_dets.rows; rIter++)
	  for(int cIter = 0; cIter < skin_dets.cols; cIter++)
	  {
	    if((skin_likelihoods.at<double>(rIter,cIter) > .35))
	      skin_dets.at<double>(rIter,cIter) = 1;
	    else
	      skin_dets.at<double>(rIter,cIter) = 0;
	  }
	  
	// create the integral images
	{Mat int_im; cv::integral(skin_likelihoods,int_im, CV_64F);
	skin_integral.reset(new Mat(int_im));}
	{Mat int_im; cv::integral(FGProbs,int_im, CV_64F);
	skin_fg_integral.reset(new Mat(int_im));}
	{Mat int_im; cv::integral(BGProbs,int_im, CV_64F);
	skin_bg_integral.reset(new Mat(int_im));}
	{Mat int_im; cv::integral(skin_dets,int_im, CV_64F);
	skin_dets_integral.reset(new Mat(int_im));}	
	
	log_im("ImRGBZ_skin_ratio",imageeq("",skin_dets,false,false));
      }
    }
        
    //double mean_fg_prob = integrate(*skin_fg_integral,roi);
    //double mean_bg_prob = integrate(*skin_bg_integral,roi);
    //return mean_fg_prob / mean_bg_prob;
    return integrate(*skin_dets_integral,roi);
#else
	  assert(false);
	  return qnan;
#endif
  }
  
  const Mat& ImRGBZ::gray() const
  {
    if(!gray_cache)
    {
      gray_cache.reset(new Mat());
      cv::cvtColor(RGB,*gray_cache,CV_RGB2GRAY);
      assert(RGB.size() == gray_cache->size());
    }
    return *gray_cache;
  }
  
  const Mat& ImRGBZ::skin() const
  {
    if(!skin_cache)
    {
      skin_cache.reset(new Mat());
      Mat pSkin, pBG;
      *skin_cache = skin_detect(RGB,pSkin,pBG);
      skin_cache->convertTo(*skin_cache,DataType<float>::type);
      assert(skin_cache->type() == DataType<float>::type);
    }
    return *skin_cache;
  }
  
  const Mat& ImRGBZ::distInvarientDepths() const
  {
#ifdef DD_CXX11
    if(!distInvDepth_cache)
    {
      distInvDepth_cache.reset(new Mat());
      *distInvDepth_cache = DistInvarientDepth(*this);
      assert(Z.size() == distInvDepth_cache->size());
    }
    return *distInvDepth_cache;
#endif
	assert(false);
	throw std::exception();
  }
  
  shared_ptr< ImRGBZ > ImRGBZ::translate_camera(float tz) const
  {
    shared_ptr<ImRGBZ> translation(new ImRGBZ(*this,true));
    
    // we know from similar triangles that
    // crop/TZ = Res/(2(f+tz))
    // crop = tz*Res/(2(f+tz))
    float cropX1 = tz*cols() / (2 * (camera.focalX()+tz));
    float cropY1 = tz*rows() / (2 * (camera.focalY()+tz));
    float cropX2 = cols() - cropX1;
    float cropY2 = rows() - cropY1;
    
    Rect roi(Point(cropX1,cropY1),Point(cropX2,cropY2));
    translation->Z = translation->Z(roi);
    translation->RGB = translation->RGB(roi);
    cv::resize(translation->Z,translation->Z,Z.size());
    cv::resize(translation->RGB,translation->RGB,RGB.size());
    translation->Z -= tz;
    translation->camera = camera;
    
    return translation;
  }
  
  void write(FileStorage& fs, string& , const shared_ptr< ImRGBZ >& im)
  {
    fs << "{";
    fs << "RGB" << im->RGB;
    fs << "Z" << im->Z;
    fs << "camera" << im->camera;
    fs << "filename" << im->filename;
    fs << "}";
  }

  void read(const FileNode& fn, shared_ptr< ImRGBZ >& im, shared_ptr< ImRGBZ > )
  {    
    Mat RGB; fn["RGB"] >> RGB;
    Mat Z; fn["Z"] >> Z;
    CustomCamera camera; fn["camera"] >> camera;
    string filename; fn["filename"] >> filename;
    
    assert(Z.type() == DataType<float>::type);
    for(int rIter = 0; rIter < Z.rows; rIter++)
      for(int cIter = 0; cIter < Z.cols; cIter++)
	Z.at<float>(rIter,cIter) = clamp<float>(params::MIN_Z(),Z.at<float>(rIter,cIter),params::MAX_Z());

    im.reset(new ImRGBZ(RGB, Z, filename,camera));
  }
    
  //
  // SECTION: Matrix Utility functions
  //
  
  void coordOf(const ImRGBZ& im, int& row, int& col, int idx) 
  {
    row = idx / im.Z.cols;
    col = idx % im.Z.cols;
    assert(idx == indexOf(im,row,col));
  }

  int indexOf(const ImRGBZ& im, int row, int col) 
  {
    return col + row * im.Z.cols;
  }  
  
  void sort(cv::Mat sort, 
			      std::vector< float >& sorted, 
			      std::vector< int >& idxs)
  {
    // initialize the index array
    assert(sort.cols == 1);
    idxs = vector<int>(sort.rows);
    for(int iter = 0; iter < sort.rows; iter++)
      idxs[iter] = iter;
    
    // delegate sorting to the STL.
    std::sort(idxs.begin(),idxs.end(),
      [&sort](int idx0, int idx1) 
      {
	return sort.at<float>(idx0) < sort.at<float>(idx1);
      });
    
    // fill in the values
    sorted = sort.clone();
    for(int iter = 0; iter < idxs.size(); iter++)
      sorted[iter] = sort.at<float>(idxs[iter]);
  }  
  
  ///
  /// SECTION: Matrix concatitination functions
  ///
  
  Mat horizCat(Mat m1, Mat m2, bool divide)
  { 
    if(m1.empty())
      return m2;
    if(m2.empty())
      return m1;
    
    if(!divide)
    {
      assert(m1.type() == m2.type());
      Mat result(std::max(m1.rows,m2.rows),m1.cols+m2.cols,m1.type(),Scalar::all(0));
      Mat roi1 = result(Rect(Point(0,0),m1.size()));
      Mat roi2 = result(Rect(Point(m1.cols,0),m2.size()));
      m1.copyTo(roi1);
      m2.copyTo(roi2);
      
      return result;
    }
    else
    {
      Mat D(std::max(m1.rows,m2.rows),5,m1.type(),toScalar(INVALID_COLOR));
      return horizCat(m1,horizCat(D,m2,false),false);
    }
  }
  
  Mat vertCat(Mat m1, Mat m2, bool divider)
  {
    if(m1.empty())
      return m2;
    if(m2.empty())
      return m1;    
    
    assert(m1.type() == m2.type());
    // add a boarder
    if(divider)
    {
      m1 = m1.clone();
      m1 = vertCat(m1,Mat(5,m1.cols,m1.type(),toScalar(INVALID_COLOR)),false);
    }
    
    // concat
    assert(m1.type() == m2.type());
    Mat result(m1.rows+m2.rows,std::max(m1.cols,m2.cols),m1.type(),Scalar::all(0));
    Mat roi1 = result(Rect(Point(0,0),m1.size()));
    Mat roi2 = result(Rect(Point(0,m1.rows),m2.size()));
    m1.copyTo(roi1);
    m2.copyTo(roi2);
    
    return result;
  }
  
  Mat tileCat(vector< Mat > ms, bool number)
  {
    // compute the geometry
    int n_rows = ceil(std::sqrt((double)ms.size()));
    int n_cols = ceil((double)ms.size()/n_rows);
    
    // build each row independently
    vector<Mat> rows(n_rows);
    for(int iter = 0; iter < ms.size(); iter++)
    {
      // debug
      Mat tile_here = number?vertCat(ms[iter],image_text(printfpp("%d",iter))):ms[iter];
      int row = iter % n_cols;
      rows[row] = horizCat(rows[row],tile_here);
    }
    
    // combine the rows to get the entire image
    Mat tiles;
    for(int row = 0; row < rows.size(); row++)
      tiles = vertCat(tiles,rows[row]);
    
    return tiles;
  }
  
#ifdef DD_CXX11
  // computes the discrete voronoi diagram using OpenCV
  // but returns with a nicer format...
  void voronoi(const Mat&src, Mat&dt, Mat&bp)
  {
    assert(src.type() == DataType<uchar>::type);
    Mat labels;
    cv::distanceTransform(src, dt, labels, CV_DIST_L2, 5, cv::DIST_LABEL_PIXEL );
    
    // map labels to coordinates
    map<int,Vec2i> src_for_label;
    for(int rIter = 0; rIter < src.rows; rIter++)
      for(int cIter = 0; cIter < src.cols; cIter++)
	// if valid pixel, valid value for component
	if(!src.at<uchar>(rIter,cIter))
	{
	  int label = labels.at<int>(rIter,cIter);
	  src_for_label[label] = Vec2i(rIter,cIter);
	  //cout << "adding label: " << label << endl;
	}
	
    // now, for the invalid coords, map coords to labels
    bp = Mat(src.rows, src.cols, DataType<Vec2i>::type);
    Mat xOrigs(src.rows, src.cols, DataType<float>::type);
    Mat yOrigs(src.rows, src.cols, DataType<float>::type);
    for(int rIter = 0; rIter < src.rows; rIter++)
      for(int cIter = 0; cIter < src.cols; cIter++)
      {
	if(src_for_label.empty())
	{
	  bp.at<Vec2i>(rIter,cIter) = Vec2i(rIter,cIter);
	  xOrigs.at<float>(rIter,cIter) = rIter;
	  yOrigs.at<float>(rIter,cIter) = cIter;	    
	}
	else
	{
	  int label = labels.at<int>(rIter,cIter);
	  bp.at<Vec2i>(rIter,cIter) = src_for_label.at(label);
	  xOrigs.at<float>(rIter,cIter) = src_for_label.at(label)[1];
	  yOrigs.at<float>(rIter,cIter) = src_for_label.at(label)[0];
	}
      }
    log_im_decay_freq("origs",horizCat(imagesc("",xOrigs),imagesc("",yOrigs)));
    log_im_decay_freq("dt",imageeq("",dt));
  }  
#endif

  Mat imtake(const Mat&im,const Size&sz,int bordertype,Scalar value)
  {
    int pad_y = std::max(0,sz.height - im.rows);
    int pad_y_top = std::ceil(pad_y/2);
    int bad_y_bot = std::floor(pad_y - pad_y_top);
    
    int pad_x = std::max(0,sz.width - im.cols);
    int pad_x_top = std::ceil(pad_x/2);
    int bad_x_bot = std::floor(pad_x - pad_x_top);
    
    Mat result; cv::copyMakeBorder(im,result,
				   pad_y_top,pad_y-pad_y_top,pad_x_top,pad_x-pad_x_top,
				   bordertype,value);
    result = result(rectFromCenter(Point(result.cols/2,result.rows/2),sz)).clone();
    assert(result.size() == sz);    
    return result;
  }

  Size enclosingSizeDivisibleBy2ToK(const Size&sz,int k)
  {
    int width = sz.width;
    int w = 0;
    while(width > 0)
    {
      w++;
      width /= 2;
    }

    int height = sz.height;
    int h = 0;
    while(height > 0)
    {
      h++;
      height /= 2;
    }

    Size outSz(std::pow(2,w),std::pow(2,h));
    log_once(safe_printf("enclosingSizeDivisibleBy2ToK = %",outSz));
    return outSz;
  }
  
  Mat impad(const Mat&im,const Size&sz)
  {
    if(im.rows >= sz.height)
    {
      cout << "error: " << im.rows << " > " << sz.height << endl;
      assert(false);
    }
    if(im.cols >= sz.width)
    {
      cout << "error: " << im.cols << " > " << sz.width << endl;
      assert(false);
    }
    
    return imtake(im,sz);
  }

  Mat imunclamped(const Mat&m, float z_min, float z_max)
  {
    assert(m.type() == DataType<float>::type);
    Mat r(m.rows,m.cols,DataType<uint8_t>::type,Scalar::all(255));
    for(int yIter = 0; yIter < m.rows; ++yIter)
      for(int xIter = 0; xIter < m.cols; ++xIter)
      {
	float src = m.at<float>(yIter,xIter);
	if(!goodNumber(src) || (clamp<float>(z_min,src,z_max) != src))
	  r.at<uint8_t>(yIter,xIter) = 0;
	else
	  r.at<uint8_t>(yIter,xIter) = 255;
      }

    return r;    
  }
  
  Mat imclamp(const Mat&m, float z_min, float z_max)
  {
    assert(m.type() == DataType<float>::type);
    Mat r(m.rows,m.cols,DataType<float>::type);
    for(int yIter = 0; yIter < m.rows; ++yIter)
      for(int xIter = 0; xIter < m.cols; ++xIter)
      {
	float src = m.at<float>(yIter,xIter);
	float&dst = r.at<float>(yIter,xIter);
	if(goodNumber(src))
	  dst = clamp<float>(z_min,src,z_max);
	else
	  dst = z_max;
      }

    return r;
  }

  Mat imround(const Mat&m, float discretization)
  {
    assert(m.type() == DataType<float>::type);
    Mat r = 1/discretization * m;
    for(int yIter = 0; yIter < r.rows; ++yIter)
      for(int xIter = 0; xIter < r.cols; ++xIter)
	r.at<float>(yIter,xIter) = std::round(r.at<float>(yIter,xIter));
    return discretization * r;
  }
  
  Mat imroi(const Mat&im,Rect roi)
  {
    auto type = im.type();

    int pad_top    = std::max(-roi.y,0);
    int pad_left   = std::max(-roi.x,0);
    int pad_bottom = std::max(roi.br().y - im.rows,0);
    int pad_right  = std::max(roi.br().x - im.cols,0);

    Mat result; cv::copyMakeBorder(im,result,
				   pad_top,pad_bottom,pad_left,pad_right,
				   cv::BORDER_REPLICATE);

    return result(Rect(Point(roi.x + pad_left,roi.y + pad_top),roi.size())).clone();
  }
  
  Mat resize_pad(const Mat& im, const Size& sz)
  {
    assert(im.rows > 0 && im.cols > 0);
    
    double im_aspect = static_cast<double>(im.cols)/static_cast<double>(im.rows);
    double sz_aspect = static_cast<double>(sz.width)/static_cast<double>(sz.height);
    
    if(im_aspect >= sz_aspect)
    {
      // pad vertical
      int interpolation = im.type() == DataType<float>::type?params::DEPTH_INTER_STRATEGY:cv::INTER_LINEAR;
      double sf = sz.width/static_cast<double>(im.size().width);
      Mat rz; cv::resize(im,rz,Size(sz.width,sf*im.size().height),0,0,interpolation);
      int pad_y = std::abs(rz.rows - sz.height);
      int pad_y_top = pad_y/2;
      int bad_y_bot = pad_y - pad_y_top;
      
      // copy rz into the result
      //Mat result(sz.height,sz.width,DataType<float>::type,Scalar::all(inf));
      //rz.copyTo(result(Rect(Point2i(0,pad_y_top),rz.size())));
      Mat result; cv::copyMakeBorder(rz,result,
				     pad_y_top,pad_y-pad_y_top,0,0,
				     cv::BORDER_REPLICATE);
      assert(result.size() == sz);
      return result;
    }
    else
    {
      // pad horizontal
      return resize_pad(im.t(),Size(sz.height,sz.width)).t();
    }
  }

  // erode then dilate
  Mat imopen(const Mat&im,int iterations)
  {
    Mat result;
    for(int iter = 0; iter < iterations; ++iter)
    {
      cv::erode(im,result,Mat());
    }
    for(int iter = 0; iter < iterations; ++iter)
    {
      cv::dilate(result,result,Mat());
    }
    return result;
  }

  // dilate then erode
  Mat imclose(const Mat&im,int iterations)
  {
    Mat result;
    for(int iter = 0; iter < iterations; ++iter)
      cv::dilate(im,result,Mat());
    for(int iter = 0; iter < iterations; ++iter)
      cv::erode(result,result,Mat());
    return result;
  }

  string write_linked(const Mat m)
  {
    // make sure we have a place to put it.
    string new_dir = safe_printf("%/model_linked_images/",params::out_dir());
    log_once("write_linked into : " + new_dir);
    boost::filesystem::create_directory(new_dir);
    assert(boost::filesystem::exists(new_dir));

    string id = yaml_fix_key(uuid());
    string filename = new_dir + id + ".png";
    imwrite(filename,m);
    return id;
  }

  Mat read_linked(const cv::FileNode&node)
  {
    assert(not node.empty());
    string model_file = g_params.require("SAVED_MODEL");
    string save_prefix = boost::regex_replace(model_file,boost::regex("model.yml"),"");

    // find the linked image
    string root_id;
    node >> root_id;
    string file = save_prefix + "model_linked_images/" + root_id + ".png";
    assert(boost::filesystem::exists(file));

    // load the linked image
    Mat mat; 
    mat = cv::imread(file,CV_LOAD_IMAGE_GRAYSCALE);
    return mat;
  }

  std::mt19937 seeded_generator(int seed)
  {
    if(seed == -1)
    {
      static mutex m; lock_guard<mutex> l(m);
      std::random_device rd;
      seed = rd();
    }    
    std::mt19937 gen(seed);
    return gen;
  }

  Mat cartesianToSpherical(const Mat&Z,const Camera&camera)
  {    
    Mat om = Z.clone();

    for(int yIter = 0; yIter < om.rows; yIter++)
      for(int xIter = 0; xIter < om.cols; xIter++)
      {	
	// convert!
	float&z = om.at<float>(yIter,xIter);
	z = camera.depth_Cartesian_to_Spherical(z,Point(xIter,yIter));
      }
    return om;
  }

  Mat sphericalToCartesian(const Mat&Z,const Camera&camera)
  {
    Mat om = Z.clone();

    for(int yIter = 0; yIter < om.rows; yIter++)
      for(int xIter = 0; xIter < om.cols; xIter++)
      {
	// unconvert!
	float&z = om.at<float>(yIter,xIter);
	z = camera.depth_Spherical_to_Cartesian(z,Point(xIter,yIter));
      }
    return om;    
  }

  ATTRIBUTE_NO_SANITIZE_ADDRESS
  static float matchTemplateL1_one(const Mat&Z,const Mat&t)
  {
    assert(Z.size() == t.size());

    return cv::norm(Z,t,NORM_L1);

    // double d = 0;

    // for(int yIter = 0; yIter < Z.rows; ++yIter)
    //   for(int xIter = 0; xIter < Z.cols; ++xIter)
    // 	d += std::abs(Z.at<float>(yIter,xIter) - t.at<float>(yIter,xIter));

    // return d;
  }

  ATTRIBUTE_NO_SANITIZE_ADDRESS
  static Mat matchTemplateL1(const Mat&Z,const Mat&t, double dynamic_range,int stride = 1)
  {
    Mat resp(Z.rows,Z.cols,DataType<float>::type,Scalar::all(inf));
    Rect im_rect(Point(0,0),Z.size());
    
    for(int yIter = stride/2; yIter < Z.rows; yIter += stride)
      for(int xIter = stride/2; xIter < Z.cols; xIter += stride)
      {
        Rect roi = rectFromCenter(Point(xIter,yIter),t.size());
	double d = 0;

	for(int yIter2 = roi.tl().y; yIter2 < roi.br().y; yIter2 += stride)
	  for(int xIter2 = roi.tl().x; xIter2 < roi.br().x; xIter2 += stride)
	  {
	    double h = 0;
	    if(im_rect.contains(Point2i(xIter2,yIter2)))
	    {
	      float z = Z.at<float>(yIter2,xIter2);
	      float tt = t.at<float>(yIter2-roi.tl().y,xIter2-roi.tl().x);
	      if(z == -inf)
		z = 0;
	      else if(z == inf)
		z = dynamic_range;
	      if(tt == -inf)
		tt = 0;
	      else if(tt == inf)
		tt = dynamic_range;
	      h = std::abs(z - tt);
	    }       
	    else
	    {
	      h = dynamic_range - 1;
	    }
	    //int next_x = stride;
	    //int next_y = stride
	    //int stride_weight_x = std::min(stride,roi.br().x - xIter2);
	    //int stride_weight_y = std::min(stride,roi.br().y - yIter2);
	    
	    d += h * stride * stride;
	  }
	resp.at<float>(yIter,xIter) = d;// - 1;
	
	
	// if(rectContains(Z,roi))	
	//   resp.at<float>(yIter,xIter) = matchTemplateL1_one(Z(roi),t);
	// else
	// {
	//   Rect valid_roi = clamp(Z,roi);	  
	//   resp.at<float>(yIter,xIter) = matchTemplateL1_one(Z(valid_roi),t)
	//     + dynamic_range*(roi.size().area() - valid_roi.size().area());
	// }
      }

    return resp;
  }

  
  Mat matchTemplateL1(const Mat&Z,const Mat&t,double tMaxSize, double dynamic_range)
  {
    if(t.size().area() > tMaxSize)
    {
      // FIRST IMPLEMENTATION
      // s*w*s*h = tMaxSize
      // =>
      // s = sqrt(tMaxSize/oldSize)
      //double s = std::sqrt(tMaxSize/static_cast<double>(t.size().area()));
      //Mat Zc; cv::resize(Z,Zc,Size(),s,s,params::DEPTH_INTER_STRATEGY);
      //Mat Tc; cv::resize(t,Tc,Size(),s,s,params::DEPTH_INTER_STRATEGY);
      //Mat r = matchTemplateL1(Zc,Tc,dynamic_range);
      //cv::resize(r,r,Z.size(),cv::INTER_NEAREST);
      ///static_cast<double>(Tc.size().area());

      // pretend like we calculated at full resolution. 
      //double cf =
      //static_cast<double>(t.size().area())/
      //static_cast<double>(Tc.size().area());
      //return cf*r;

      // SECOND IMPLEMENTATION
      int stride = std::max<int>(
	1,
	std::ceil(std::sqrt(static_cast<double>(t.size().area())/tMaxSize)));
      assert(stride >= 2);
      log_once(safe_printf("stride = %",stride));
      return matchTemplateL1(Z,t,dynamic_range,stride);
    }
    else
    {
      ///static_cast<double>(t.size().area());
      return matchTemplateL1(Z,t,dynamic_range);
    }
  }

  Mat kahan_filter2d(const Mat&X,const Mat&T, Point anchor)
  {
    Mat XX; X.convertTo(XX,DataType<double>::type);
    Mat TT; T.convertTo(TT,DataType<double>::type);
    
    Mat resps(X.rows,X.cols,DataType<double>::type,Scalar::all(0));
    TaskBlock filter_block("filter_block");
    for(int yIter = 0; yIter < XX.rows; ++yIter)
      for(int xIter = 0; xIter < XX.cols; ++xIter)
      {
	filter_block.add_callee([&,yIter,xIter]()
				{
				  KahanSummation sum;
				  for(int y = 0; y < TT.rows; ++y)
				    for(int x = 0; x < TT.cols; ++x)
				    {
				      int XX_row = yIter + y - anchor.y;
				      int XX_col = xIter + x - anchor.x;
				      if(XX_row < 0 || XX_row >= XX.rows || XX_col < 0 || XX_col >= XX.cols)
					continue;
				      sum += TT.at<double>(y,x)*XX.at<double>(XX_row,XX_col);
				    }
				  resps.at<double>(yIter,xIter) = sum.current_total();
				});
      }
    filter_block.execute();
    return resps;
  }
}
