/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_UTIL_MAT
#define DD_UTIL_MAT

#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include <boost/concept_check.hpp>
#undef RGB

#include <limits>
#include "util_depth.hpp"
#include "params.hpp"
#include <memory>
#include <functional>
#include <ThreadCompat.hpp>

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  void minmax(Mat_<float> image, float&min, float&max);
  float min(Mat_<float> image);
  float max(Mat_<float> image);
  Mat max(Mat m1, Mat m2);
  Mat min(Mat m1, Mat m2);
  
  struct Extrema
  {
  public:
    double max;
    double min;
    Point2i maxLoc;
    Point2i minLoc;
  };  
  
  Extrema extrema(const Mat& im);
  
  // any non zero elements?
  template<typename T>
  bool any(const Mat src, function<bool(T val)> predicate)
  {
    for(int rIter = 0; rIter < src.rows; rIter++)
      for(int cIter = 0; cIter < src.cols; cIter++)
	if(predicate(src.at<T>(rIter,cIter)))
	  return true;
    return false;
  }
  bool any(const Mat src);
  Mat_<float> isBad(const Mat_<float> src);
  Mat_<float> isnan(const Mat_<float> src);
  Mat_<float> sqrt(const Mat_<float> src);
  Mat_<float> pow(const Mat_<float> src, float p);
  Mat log(const Mat&in);
  
  void printSize(Size sz);
  void printMatType(int type);
  float matScalar(Mat_<float> m);
  bool goodNumber(float n);
  Mat inv(Mat m);
  
  template<typename T>
  Mat crop_and_copy(Mat src, Rect roi)
  {
    // in bounds, just use ROI operator
    if(roi.tl().x >= 0 && roi.tl().y >= 0 &&
       roi.br().y < src.rows && roi.br().x < src.cols)
    {
      assert(src.type() == DataType<T>::type);
      Mat result = src(roi).clone();
      assert(result.type() == DataType<T>::type);
      return result;
    }
    
    // out of bounds
    assert(src.type() == DataType<T>::type);
    Mat result(roi.height,roi.width,src.type());
    for(int dst_y = 0; dst_y < roi.height; dst_y++)
      for(int dst_x = 0; dst_x < roi.width; dst_x++)
      {
	int src_y = dst_y + roi.tl().y;
	int src_x = dst_x + roi.tl().x;
	// BORDER_REFLECT_101 or BORDER_REPLICATE
	src_y = cv::borderInterpolate(src_y,src.rows,BORDER_REPLICATE);
	src_x = cv::borderInterpolate(src_x,src.cols,BORDER_REPLICATE);
	result.at<T>(dst_y,dst_x) = 
	  src.at<T>(src_y,src_x);
      }
      
    assert(result.type() == DataType<T>::type);
    return result;
  }  
  
  // affine transform utility functions
  Mat affine_lr_flip(int width);
  Mat affine_identity();
  Mat affine_translation(float tx, float ty);
  Mat affine_transform(float tx, float ty, float theta_rad);
  Mat affine_compose(Mat affine1,Mat affine2);
 
  struct ImRGBZ
  {
  public:
    ImRGBZ(Mat RGB_, Mat Z_, string filename,const CustomCamera&camera) : 
      RGB(RGB_), Z(Z_), filename(filename), camera(camera)
    {
      this->camera.setRes(RGB_.cols,RGB_.rows);
      assert(RGB.size() == Z.size());
      valid_region = Rect(Point(0,0),RGB.size());
      affine = affine_identity();
    };
    ImRGBZ(const ImRGBZ&copy, bool clone = true);
    ImRGBZ& operator=(const ImRGBZ&other);
    ImRGBZ operator()( const Rect& roi ) const;
    ImRGBZ roi_light(const Rect&roi) const;
    ImRGBZ roi_light(const RotatedRect&roi) const;
    ImRGBZ resize(double sf, bool skip_image_data = false) const;
    ImRGBZ resize(Size sz, bool skip_image_data = false) const;
    ImRGBZ flipLR() const;
    shared_ptr<ImRGBZ> translate_camera(float tz) const;
    int rows() const;
    int cols() const;
    const Mat& gray() const;
    const Mat& distInvarientDepths() const;
    const Mat& skin() const;
    Mat affine; // affine transform from loaded image to this image

    // functions which use integral images to mean depth in window
    double mean_depth(Rect roi) const;
    double skin_ratio(Rect roi) const;
    double mean_intensity(Rect roi) const;
    
  // data
  public:
    Mat RGB, Z;
    CustomCamera camera;
    string filename;
    Rect valid_region;
  protected:
    mutable std::shared_ptr<Mat> gray_cache;
    mutable std::unique_ptr<Mat> skin_cache;
    mutable std::shared_ptr<Mat> distInvDepth_cache;
    mutable std::shared_ptr<Mat> intensity_integral;
    mutable std::shared_ptr<Mat> depth_integral;
    mutable mutex skin_integral_mutex;
    mutable std::shared_ptr<Mat> 
      skin_integral, skin_fg_integral, skin_bg_integral, skin_dets_integral;

    friend void read(const FileNode&, shared_ptr<ImRGBZ>&, shared_ptr<ImRGBZ>);
    friend void write(FileStorage&, string&, const shared_ptr<ImRGBZ>&);
  };
  void read(const FileNode&, shared_ptr<ImRGBZ>&, shared_ptr<ImRGBZ>);
  void write(FileStorage&, string&, const shared_ptr<ImRGBZ>&);
  
  void coordOf(const ImRGBZ& im, int& row, int& col, int idx) ;
  int indexOf(const ImRGBZ& im, int row, int col);

  std::mt19937 seeded_generator(int seed = -1);

  template<typename T>
  vector<T> random_sample_w_replacement(const vector<T>&domain,size_t n, int seed = -1)
  {
    vector<T> result(n);
    assert(domain.size() > 0);
    
    std::mt19937 gen = seeded_generator(seed);
    std::uniform_int_distribution<> dist(0, domain.size()-1);
    
    for(int iter = 0; iter < n; ++iter)
      result[iter] = domain[dist(gen)];
    
    return result;
  }
  
  template<typename T> 
  vector<T> random_sample(vector<T> domain, size_t n, int seed  = -1)
  {
    if(domain.size() <= 0)
      return vector<T>{};
    std::mt19937 gen = seeded_generator(seed);
    n = clamp<size_t>(0,n,domain.size());
    std::shuffle(domain.begin(),domain.end(),gen);
    return vector<T>(domain.begin(),domain.begin()+n);
  }
 
  void sort(Mat sort,vector<float>&sorted,vector<int>&idxs);

  // 1 for round to nearet int, .5 for round to nearest half.
  Mat imround(const Mat&m, float discretization = 1); 
  Mat imclamp(const Mat&m, float z_min, float z_max);
  Mat imunclamped(const Mat&m, float z_min, float z_max);
  Mat imrotate(const Mat&m, double angle_in_radians, Point2f center = Point2f());
  Mat imrotate_tight(const Mat&m, double angle_in_radians);
  Mat horizCat(Mat m1, Mat m2, bool divide = true);
  Mat vertCat(Mat m1, Mat m2, bool divier = true);
  Mat tileCat(vector<Mat> ms, bool number = true);
  void voronoi(const Mat&src, Mat&dt, Mat&bp);
  Vec3d unhomo4(const Mat&m);
  double angle(const Vec3d&v1,const Vec3d&v2);
  // resize to the given size, maxpect style. Pad for proper aspect ratio.
  Mat resize_pad(const Mat&im,const Size&sz);  
  Mat impad(const Mat&im,const Size&sz);
  Size enclosingSizeDivisibleBy2ToK(const Size&sz,int k);
  Mat imtake(const Mat&im,const Size&sz,int bordertype = cv::BORDER_REPLICATE,Scalar value = Scalar(0,0,0));
  Mat imopen(const Mat&im,int iterations = 1);
  Mat imroi(const Mat&im,Rect roi);
  Mat imclose(const Mat&im, int iterations = 1);
  // write a matrix to a file and return the filename.
  string write_linked(const Mat m);
  Mat read_linked(const cv::FileNode&node);

  class at
  {
  protected:
    int row, col;
    const Mat&m;

  public:
    at(const Mat&m, int row,int col)  : m(m)
    {
      this->row = row;
      this->col = col;
    }

    template<typename T>
    at& operator= (const T&v)
    {
      if(m.type() == DataType<T>::type)
	const_cast<Mat&>(m.at<T>(row,col)) = v;
      else
	throw std::logic_error("wrong type");

      return *this;
    }

    operator double() const
    {      
      assert(0 <= row);
      assert(0 <= col);
      assert(row < m.rows);
      assert(col < m.cols);
      
      if(m.type() == DataType<float>::type)
	return m.at<float>(row,col);
      if(m.type() == DataType<double>::type)
	return m.at<double>(row,col);
      if(m.type() == DataType<uint8_t>::type)
	return m.at<uint8_t>(row,col);
      if(m.type() == DataType<int>::type)
	return m.at<int>(row,col);
      if(m.type() == DataType<size_t>::type)
	return m.at<size_t>(row,col);
      throw std::logic_error("Unknown Type");

      return qnan;
    };
  };

  // convert D pixels to Z pixels  
  Mat cartesianToSpherical(const Mat&Z,const Camera&camera);
  Mat sphericalToCartesian(const Mat&Z,const Camera&camera);

  // filtering operations
  Mat matchTemplateL1(const Mat&Z,const Mat&t,double tMaxSize, double dynamic_range);
  vector<Mat> matchTemplatesL1(const Mat&Z,const vector<Mat>&ts,double tMaxSize, double dynamic_range);
  cv::Mat kahan_filter2d(const cv::Mat&X,const cv::Mat&T, cv::Point anchor);
}

#endif
