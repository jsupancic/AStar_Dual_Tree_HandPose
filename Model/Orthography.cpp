/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Orthography.hpp"
#include "util.hpp"
#include "PXCSupport.hpp"

#include <exception>
#include <stdexcept>
#include <limits>
#include <boost/concept_check.hpp>
#include <queue>

namespace deformable_depth
{
  using namespace std;
  
  static constexpr float sf = 2;
  
  MetaData_Orthographic::MetaData_Orthographic(string filename, bool read_only): 
    MetaData_YML_Backed(filename, read_only),
    fx(std::abs(.5*params::depth_hRes/tan(params::PI/180.0f*params::H_Z_FOV/2))),
    fy(std::abs(.5*params::depth_vRes/tan(params::PI/180.0f*params::V_Z_FOV/2))),
    isoSize(sf*params::depth_hRes,sf*params::depth_vRes)
  {
  }
  
  // float version...
  void MetaData_Orthographic::map2iso(float persZ, Size persZSize, int persX, int persY, float& isoX, float& isoY)
  {
      isoX = sf*(persX-persZSize.width/2.0f)/fx*persZ + isoSize.width/2.0f;
      isoY = sf*(persY-persZSize.height/2.0f)/fy*persZ + isoSize.height/2.0f;
  }
  
  // this just converts from int to float
  void MetaData_Orthographic::map2iso(float persZ,Size persZSize, int persX, int persY, int& isoX, int& isoY)
  {
      float fisox, fisoy;
      map2iso(persZ,persZSize,persX,persY,fisox,fisoy);
      isoX = clamp<int>(0,fisox,isoSize.width-1);
      isoY = clamp<int>(0,fisoy,isoSize.height-1);
  }
  
  shared_ptr< ImRGBZ > MetaData_Orthographic::load_im()
  {
    Mat RGB, Z;
    loadRGBOnZ(filename + ".gz",RGB,Z);
    //image_safe("RGB - Loading",RGB);
    
    Mat 
      isoZ(isoSize.height,isoSize.width,DataType<float>::type,Scalar::all(numeric_limits<float>::quiet_NaN())), 
      isoRGB(isoSize.height,isoSize.width,DataType<Vec3b>::type,Scalar::all(0));
    //cout << "isoZ dims " << isoZ.rows << ", " << isoZ.cols << endl;
      
    // map
    Mat invalid = Mat(isoZ.rows,isoZ.cols,DataType<uchar>::type,Scalar::all(1));
    for(int inY = 0; inY < Z.rows; inY++)
      for(int inX = 0; inX < Z.cols; inX++)
      {
	// in{X,Y} each correspond to an angle.
	int outYi, outXi;
	float z = Z.at<float>(inY,inX);
	map2iso(z,Size(Z.cols,Z.rows),inX,inY,outXi,outYi);
	
	// update the iso maps
	float&outz = isoZ.at<float>(outYi,outXi);
	//if(z < params::MAX_Z() && (::isnan(outz) || z < outz))
	if(::isnan(outz) || z < outz)
	{
	  outz = z;
	  isoRGB.at<Vec3b>(outYi,outXi) = RGB.at<Vec3b>(inY,inX);
	  invalid.at<uchar>(outYi,outXi) = 0;
	}
      }
      
    // clean up.
    Mat dist2Valid; distanceTransform(invalid,dist2Valid,CV_DIST_L2,CV_DIST_MASK_PRECISE);
    //float valid_dist = 2*::sqrt(isoZ.size().area()/(float)Z.size().area()) + 1;
    float valid_dist = 2*::sqrt(isoZ.size().area()/(float)Z.size().area()) + 1;
    for(int yIter = 0; yIter < isoZ.rows; yIter++)
      for(int xIter = 0; xIter < isoZ.cols; xIter++)
	if(dist2Valid.at<float>(yIter,xIter) >= valid_dist)
	  isoZ.at<float>(yIter,xIter) = params::MAX_Z();
    isoZ = fillDepthHoles(isoZ);
    
    float qnan = numeric_limits<float>::quiet_NaN();
    CustomCamera camera(qnan,qnan, qnan,qnan);
    return shared_ptr<ImRGBZ>(new ImRGBZ(isoRGB,isoZ,filename,camera));
  }
  
  std::shared_ptr< const ImRGBZ > MetaData_Orthographic::load_im() const
  {
    assert(false);
    // TODO: Unimplemented!
  }

  map< string, AnnotationBoundingBox > MetaData_Orthographic::get_positives()
  {
    // we need perpsecitive Z map to perform this mapping operation.
    Mat persRGB, persZ;
    loadRGBOnZ(filename + ".gz",persRGB,persZ);
    
    map<string,AnnotationBoundingBox> result;
    map<string,AnnotationBoundingBox> input_labels;
    input_labels["HandBB"] = AnnotationBoundingBox(HandBB_ZCoords,true);
    float qnan = numeric_limits<float>::quiet_NaN();
    
    // map all the results from the perspective to the isometric camera.
    for(auto &label : input_labels)
    {
      // find median depth in hand
      AnnotationBoundingBox bb = label.second;
      vector<float> zs;
      for(int yIter = bb.tl().y; yIter < bb.br().y; yIter++)
	for(int xIter = bb.tl().x; xIter < bb.br().x; xIter++)      
	  zs.push_back(persZ.at<float>(yIter,xIter));
      sort(zs.begin(),zs.end());

      // map
      float z = zs[zs.size()/4];
      Point p1, p2;
      map2iso(z,Size(persZ.cols,persZ.rows),bb.tl().x, bb.tl().y, p1.x, p1.y);
      map2iso(z,Size(persZ.cols,persZ.rows),bb.br().x, bb.br().y, p2.x, p2.y);
      Rect newBB(p1,p2);
      
      //result[label.first + "orig"] = bb;
      result[label.first] = AnnotationBoundingBox(newBB,true);//updateBB(bb,mappings,false,true);
    }
    
    return result;
  }

  void MetaData_Orthographic::set_HandBB(Rect newHandBB)
  {
    throw runtime_error("unsupported");
  }
  
  map< string, AnnotationBoundingBox > 
    map2ortho(const Camera& perspective_camera, Size outRes, 
	      const map< string, AnnotationBoundingBox >& pers_labels, const ImRGBZ& im)
  {
    map< string, AnnotationBoundingBox > ortho_labels;
    
    for(auto&pers_label : pers_labels)
    {
      auto center = rectCenter(pers_label.second);
      if(center.x >= im.Z.cols || center.y >= im.Z.rows ||
	 center.x <  0         || center.y <  0)
      {
	log_file << "warning: interpolating" << pers_label.first << " : " << center;
	center.x = cv::borderInterpolate(center.x,im.Z.cols, BORDER_REPLICATE);
	center.y = cv::borderInterpolate(center.y,im.Z.rows, BORDER_REPLICATE);
	log_file << " => " << center << endl;
      }
      double z = im.Z.at<float>(center.y,center.x);
      ortho_labels[pers_label.first].write(map2ortho(perspective_camera,outRes,pers_label.second,z));
    }
    
    return ortho_labels;
  }
  
  Rect_<double> map2ortho(const Camera& perspective_camera, Size outRes, Rect persBB, float z)
  {
    // find z as best we can..
    float x1, y1, x2, y2;
    map2ortho(perspective_camera,outRes,persBB.tl().x,persBB.tl().y,z,x1,y1);
    map2ortho(perspective_camera,outRes,persBB.br().x,persBB.br().y,z,x2,y2);
    return Rect(Point(x1,y1),Point(x2,y2));
  }
  
  void map2ortho(const Camera& perspective_camera, 
		 Size outRes, 
		 float px, float py, float z, 
		 float& ox, float& oy, bool noclamp)
  {
    // a high level eq. 
    // ox = (outWidth/max_ox) * z * (px - hRes/2)/fx + outWidth/2
    
    // focal lengths
    float fx = perspective_camera.focalX();
    float fy = perspective_camera.focalY();    
    // center
    px = px - perspective_camera.hRes()/2;
    py = py - perspective_camera.vRes()/2;
    // to ortho
    ox = px/fx*z;
    oy = py/fy*z;
    //
    float max_ox = (perspective_camera.hRes()/fx*params::MAX_Z());
    float max_oy = (perspective_camera.vRes()/fy*params::MAX_Z());
    // compute unit conversion to outDim
    float sx = outRes.width/max_ox;
    float sy = outRes.height/max_oy;
    ox *= sx;
    oy *= sy;
    // fix coordinate frame
    ox = outRes.width/2+ox;
    oy = outRes.height/2+oy;
    // return
    if(noclamp)
      return;
    ox = clamp<float>(0,ox,outRes.width-1);
    oy = clamp<float>(0,oy,outRes.height-1);
  }
  
  Rect_< double > mapFromOrtho(const Camera& perspective_camera, Size outRes, Rect orthoBB, float z)
  {
    float x1, y1, x2, y2;
    mapFromOrtho(perspective_camera,outRes,orthoBB.tl().x,orthoBB.tl().y,z,x1,y1);
    mapFromOrtho(perspective_camera,outRes,orthoBB.br().x,orthoBB.br().y,z,x2,y2);
    return Rect(Point(x1,y1),Point(x2,y2));
  }
  
  void mapFromOrtho(
    const Camera& perspective_camera, 
    Size outRes, 
    float ox, float oy, float z, 
    float& px, float& py)
  {
    // we want to invert the following (solve for px): 
    // ox = (outWidth/max_ox) * z * (px - hRes/2)/fx + outWidth/2
    // ox - outWidth/2 = (outWidth/max_ox) * z * (px - hRes/2)/fx 
    // (ox - outWidth/2)*(fx/z) = (outWidth/max_ox) * (px - hRes/2) 
    // (ox - outWidth/2)*(fx/z)*(max_ox/outWidth) = px - hRes/2
    // px = (ox - outWidth/2)*(fx/z)*(max_ox/outWidth) + hRes/2
    
    // compute focal lengths and orthographic limits
    float fx = perspective_camera.focalX();
    float fy = perspective_camera.focalY(); 
    float max_ox = (perspective_camera.hRes()/fx*params::MAX_Z());
    float max_oy = (perspective_camera.vRes()/fy*params::MAX_Z());
    
    // map the coordinates from orthographic inputs to perspective outputs
    px = (ox - outRes.width/2)*(fx/z)*(max_ox/outRes.width) + perspective_camera.hRes()/2;
    py = (oy - outRes.height/2)*(fy/z)*(max_oy/outRes.height) + perspective_camera.vRes()/2;
  }
  
  Vec2d ortho2cm(Vec2d ortho_dist, Size ortho_res, const Camera& camera)
  {
    ortho_dist[0] *= camera.widthAtDepth(params::MAX_Z())/ortho_res.width;
    ortho_dist[1] *= camera.heightAtDepth(params::MAX_Z())/ortho_res.height;
    return ortho_dist;
  }
  
  Vec3d ortho2cm(Vec3d ortho_pos, Size ortho_res, const Camera& camera)
  {
    Vec2d xy = ortho2cm(Vec2d(ortho_pos[0],ortho_pos[1]),ortho_res,camera);
    return Vec3d(xy[0],xy[1],ortho_pos[2]);
  }
  
  Vec2d cm2ortho(Vec2d cm_dist, Size ortho_res, const  Camera& camera)
  {
    cm_dist[0] *= ortho_res.width/camera.widthAtDepth(params::MAX_Z());
    cm_dist[1] *= ortho_res.height/camera.heightAtDepth(params::MAX_Z());
    return cm_dist;    
  }
  
  void map2ortho_cm(const Camera& perspective_camera, 
	       float px, float py, float z, 
	       float&ox, float&oy)
  {
    const bool VERBOSE = false;
    if(VERBOSE)
    {
      cout << "======================" << endl;
      cout << "z: " << z << endl;
    }
    
    // focal lengths
    float fx = perspective_camera.focalX();
    float fy = perspective_camera.focalY();    
    // center
    px = px - perspective_camera.hRes()/2;
    py = py - perspective_camera.vRes()/2;
    // to ortho
    ox = px/fx*z;
    oy = py/fy*z;
    // to cm units
    ox = ox*perspective_camera.widthAtDepth(fx)/perspective_camera.hRes();
    oy = oy*perspective_camera.heightAtDepth(fy)/perspective_camera.vRes();
    
    if(VERBOSE)
    {
      cout << "ox: " << ox << endl;
      cout << "oy: " << oy << endl;
    }
  }
  
  Point2d map2ortho_cm(const Camera& perspective_camera, Point p1, double z)
  {
    float ox, oy;
    map2ortho_cm(perspective_camera,p1.x,p1.y,z,ox,oy);
    return Point(ox,oy);
  }
  
  Rect_< double > map2ortho_cm(const Rect_< double >& in_rect, const Camera&camera,double z)
  {
    // using the same Z map TL and BR
    float tl_ox, tl_oy;
    map2ortho_cm(camera, 
	       in_rect.tl().x, in_rect.tl().y, z, 
	       tl_ox, tl_oy);
    float br_ox, br_oy;
    map2ortho_cm(camera, 
	       in_rect.br().x, in_rect.br().y, z, 
 	       br_ox, br_oy);
    
    // return the result
    Rect_<double> result(Point2d(tl_ox,tl_oy),Point2d(br_ox,br_oy));
    //assert(result != Rect_<double>());
    return result;
  }
  
  Rect_< double > map2ortho_cm(MetaData& metadata, const Rect_< double >& in_rect)
  {
    // establish the Z
    shared_ptr<ImRGBZ> im = metadata.load_im();
    double z = extrema(im->Z(in_rect)).min;
    log_once(printfpp("map2ortho_cm %f",z));
    
    return map2ortho_cm(in_rect,im->camera,z);
  }
  
  void orthography(int argc, char** argv)
  {
    assert(argc >= 3);
    string filename(argv[2]);
    cout << "orthography demo running on " << filename << endl;
    bool vis;
    
    shared_ptr<MetaData> metadata = metadata_build(filename);
    shared_ptr<ImRGBZ> im = metadata->load_im();
    string winName = "Orthography Demo";
    imageeq(winName.c_str(),im->Z,true,false);
    
    while(true)
    {
      cout << "give point 1" << endl;
      Point2i p1 = getPt(winName,&vis);
      cout << "give point 2" << endl;
      Point2i p2 = getPt(winName,&vis);
      
      Vec3f orth_p1;
      orth_p1[2] = im->Z.at<float>(p1.y,p1.x);
      map2ortho_cm(im->camera,
	       p1.x, p1.y, orth_p1[2],
	       orth_p1[0], orth_p1[1]);
      
      Vec3f orth_p2;
      orth_p2[2] = im->Z.at<float>(p2.y,p2.x);
      map2ortho_cm(im->camera,
	       p2.x, p2.y, orth_p2[2],
	       orth_p2[0], orth_p2[1]);      
      
      Vec3f diff = (orth_p1-orth_p2);
      float dist = std::sqrt(diff.ddot(diff));
      cout << "distance is " << dist << endl;
    }
  }

  cv::Mat paint_orthographic(const Mat&Z,const Camera&camera)
  {
    Mat RGB = imageeq("",Z,false,false);
    CustomCamera cc(camera.hFov(),camera.vFov(),camera.hRes(),camera.vRes(),camera.metric_correction());
    ImRGBZ im(RGB,Z,uuid(),cc);

    return OrthographicOccupancyMap(im).proj_z();
  }

  cv::Mat paint_orthographic(const ImRGBZ&im)
  {
    return OrthographicOccupancyMap(im).proj_z();
  }

  ///
  /// SECTION: class OrthographicOccupancyMap
  ///
  OrthographicOccupancyMap::OrthographicOccupancyMap(const ImRGBZ&im) : 
    om(boost::extents
       [params::MAX_X() - params::MIN_X()]
       [params::MAX_Y() - params::MIN_Y()]
       [params::MAX_Z() - params::MIN_Z()])
  {
    // clear the image.
    for(int xIter = 0; xIter < xSize(); ++xIter)
      for(int yIter = 0; yIter < ySize(); ++yIter)
	for(int zIter = 0; zIter < zSize(); ++zIter)
	  om[xIter][yIter][zIter] = 0;

    // store all the metric voxes in a priority queue
    // for the painter's algorithm.
    Rect_<double> ortho_bounding_box; // box bounding the orthographic pixels
    struct Pixel
    {
      float z;
      Rect_<double> ortho_rect;
      Rect_<double> far_rect;
      // from small to large ordering, 
      // std::pq will go from back to front (eg painter's algorithm)
      bool operator< (const Pixel&pixel) const
      {
	return z < pixel.z;
      };
    };
    std::priority_queue<Pixel> q;
    for(int yIter = 0; yIter < im.rows(); yIter++)
      for(int xIter = 0; xIter < im.cols(); xIter++)
      {
	// figure out where the pixel is in depth
	float z = im.Z.at<float>(yIter,xIter);
	if(goodNumber(z))
	{
	  Rect_<double> persp_pixel(Point2d(xIter,yIter),Size(1,1));
	  Rect_<double> ortho_pixel = map2ortho_cm(persp_pixel,im.camera,z);
	  Rect_<double> far_pixel   = map2ortho_cm(persp_pixel,im.camera,params::MAX_Z());
	  if(ortho_bounding_box == Rect_<double>())
	    ortho_bounding_box = ortho_pixel;
	  else
	    ortho_bounding_box |= far_pixel;
	  q.push(Pixel{z,ortho_pixel,far_pixel});
	}
      }

    // allocate the orthographic matrices
    //log_once(safe_printf("AutoAlignedTemplate::AutoAlignedTemplate res = % X %",x_range,y_range));
    Rect out_window(Point(0,0),Point(xSize(),ySize()));
    Mat at = affine_transform(ortho_bounding_box,out_window);
    //cout << safe_printf("at from % to %",ortho_bounding_box,out_window) << endl;
    // paint the matrices
    while(!q.empty())
    {
      // get the next pixel
      Pixel p = q.top();
      q.pop();
      // draw it
      for(int zIter = p.z; zIter < params::MAX_Z(); zIter++)
      {
	double x1 = interpolate_linear_prop(zIter,p.z,params::MAX_Z(),p.ortho_rect.tl().x,p.far_rect.tl().x);
	double y1 = interpolate_linear_prop(zIter,p.z,params::MAX_Z(),p.ortho_rect.tl().y,p.far_rect.tl().y);
	double x2 = interpolate_linear_prop(zIter,p.z,params::MAX_Z(),p.ortho_rect.br().x,p.far_rect.br().x);
	double y2 = interpolate_linear_prop(zIter,p.z,params::MAX_Z(),p.ortho_rect.br().y,p.far_rect.br().y);
	Rect rectAtDepth(Point(x1,y1),Point(x2,y2));

	Point2d tl = point_affine(rectAtDepth.tl(),at);
	Point2d br = point_affine(rectAtDepth.br(),at);
	//cout << safe_printf("paint_orthographic tl = % br = %",tl,br) << endl;
	int yMin = clamp<int>(0,std::floor(tl.y),ySize()-1);
	int yMax = clamp<int>(0,std::ceil(br.y),ySize()-1);
	int xMin = clamp<int>(0,std::floor(tl.x),xSize()-1);
	int xMax = clamp<int>(0,std::ceil(br.x),xSize()-1);
	for(int yIter = yMin; yIter < yMax; ++yIter)
	  for(int xIter = xMin; xIter < xMax; ++xIter)
	  {
	    om[xIter][yIter][zIter] = true;
	  }
      }
    }
  }

  Visualization OrthographicOccupancyMap::vis() const
  {
    return Visualization(proj_z(),"ProjectZ");
  }

  Mat OrthographicOccupancyMap::proj_z() const
  {
    Mat viz(ySize(),xSize(),DataType<float>::type,Scalar::all(inf));
    
    for(int yIter = 0; yIter < ySize(); ++yIter)
      for(int xIter = 0; xIter < xSize(); ++xIter)
	for(int zIter = 0; zIter < zSize(); ++zIter)
	  if(om[xIter][yIter][zIter])
	  {
	    float & here = viz.at<float>(yIter,xIter);
	    here = std::min<float>(here,zIter);
	  }

    return imageeq("",viz,false,false);
  }

  Mat OrthographicOccupancyMap::proj_y() const
  {
    Mat viz(ySize(),xSize(),DataType<float>::type,Scalar::all(inf));
    
    for(int yIter = 0; yIter < ySize(); ++yIter)
      for(int xIter = 0; xIter < xSize(); ++xIter)
	for(int zIter = 0; zIter < zSize(); ++zIter)
	  if(om[xIter][yIter][zIter])
	  {
	    float & here = viz.at<float>(xIter,zIter);
	    here = std::min<float>(here,yIter);
	  }

    return imageeq("",viz,false,false);
  }

  size_t OrthographicOccupancyMap::xSize() const
  {
    return om.shape()[0];
  }

  size_t OrthographicOccupancyMap::ySize() const
  {
    return om.shape()[1];
  }

  size_t OrthographicOccupancyMap::zSize() const
  {
    return om.shape()[2];
  }
}

