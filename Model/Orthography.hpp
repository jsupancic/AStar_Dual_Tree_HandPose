/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_ISOMETRY
#define DD_ISOMETRY

#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

#include "Visualization.hpp"
#include "boost/multi_array.hpp"
#include "util_mat.hpp"
#include "MetaData.hpp"

namespace deformable_depth
{
  using std::map;
  using std::string;
  using cv::Rect;
  using std::shared_ptr;
  
  class MetaData_Orthographic : public MetaData_YML_Backed
  {
  public:
    virtual ~MetaData_Orthographic(){};
    MetaData_Orthographic(string filename, bool read_only = false);
    virtual map<string,AnnotationBoundingBox > get_positives();
    virtual void set_HandBB(cv::Rect newHandBB);
    virtual std::shared_ptr<ImRGBZ> load_im();   
    virtual std::shared_ptr<const ImRGBZ> load_im() const;
  protected:
    float fx, fy;
    void map2iso(float persZ, Size persZSize, int persX, int persY, float&isoX, float&isoY);
    void map2iso(float persZ, Size persZSize, int persX, int persY, int&isoX, int&isoY);
    Size isoSize;
  };
  
  Vec3d ortho2cm(Vec3d ortho_pos, Size ortho_res, const Camera&camera);
  Vec2d ortho2cm(Vec2d ortho_dist,Size ortho_res,const Camera&camera);
  Vec2d cm2ortho(Vec2d cm_dist,Size ortho_res,const Camera&camera);
  void mapFromOrtho(const Camera&perspective_camera,Size outRes,
	      float ox, float oy, float z,
	      float&px, float&py);
  Rect_<double> mapFromOrtho(const Camera&perspective_camera,Size outRes,
		 Rect orthoBB, float z);  
  void map2ortho(const Camera&perspective_camera,Size outRes,
	       float px, float py, float z,
	       float&ox, float&oy, bool noclamp = false);
  Rect_<double> map2ortho(const Camera&perspective_camera,Size outRes,
		 Rect persBB, float z);
  map<string,AnnotationBoundingBox> map2ortho(const Camera&perspective_camera,Size outRes,
		 const map<string,AnnotationBoundingBox>&pers_labels, const ImRGBZ&im);
  void map2ortho_cm(const Camera&perspective_camera,
	       float px, float py, float z,
	       float&ox, float&oy);
  Point2d map2ortho_cm(const Camera&perspective_camera,Point p1, double z);
  Rect_<double> map2ortho_cm(MetaData&metadata,const Rect_<double>&in_rect);
  Rect_<double> map2ortho_cm(const Rect_<double>&in_rect, const Camera&camera,double z);
  void orthography(int argc,char**argv);

  cv::Mat paint_orthographic(const Mat&Z,const Camera&camera);
  cv::Mat paint_orthographic(const ImRGBZ&im);

  class OrthographicOccupancyMap
  {
  protected:
    typedef boost::multi_array<bool, 3> OccupancyMap;    
    OccupancyMap om;    

  public:
    OrthographicOccupancyMap(const ImRGBZ&im);
    Mat proj_z() const;
    Mat proj_x() const;
    Mat proj_y() const;
    Visualization vis() const;
    size_t xSize() const;
    size_t ySize() const;
    size_t zSize() const;
  };
  typedef OrthographicOccupancyMap RectLinearOccupancyMap;
}

#endif

