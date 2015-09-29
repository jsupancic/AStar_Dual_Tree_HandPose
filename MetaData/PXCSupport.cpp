/**
 * Copyright 2014: James Steven Supancic III
 **/
#include <stdio.h>
#include <opencv2/core/core.hpp>

#include "PXCSupport.hpp"
#include "util_real.hpp"
#include "util_file.hpp"
#include "util_depth.hpp"
#include "params.hpp"
#include "util.hpp"
#include "util_file.hpp"

namespace deformable_depth
{
  using namespace cv;
  
  static Size IM_SIZE(640,480);
  	
  void loadPCX_RGB(const string&filename,Mat&RGB)
  {
    // load the data
    FileStorage pxcFile(filename,FileStorage::READ);
    if(!pxcFile.isOpened())
    {
      printf("loadPCX_RGB: Failed to Open %s\n",filename.c_str());
      assert(pxcFile.isOpened());
    }
    pxcFile["RGB"] >> RGB;
    pxcFile.release();
  }

  PCX_Props loadPCX_RawZ(const string& filename, Mat& ZMap, Mat& ZConf, Mat& ZUV)
  {
    PCX_Props props;
    
    FileStorage pxcFile(filename,FileStorage::READ);
    if(!pxcFile.isOpened())
    {
      printf("Failed to open \"%s\"\n",filename.c_str());
      assert(pxcFile.isOpened());
    }
    pxcFile["ZMap"] >> ZMap;
    pxcFile["ZConf"] >> ZConf; // higher is better
    pxcFile["UV"] >> ZUV;
    pxcFile["PROPERTYDEPTHLOWCONFIDENCEVALUE"] >> props.lowConf; 
    pxcFile["PROPERTY_DEPTH_SATURATION_VALUE"] >> props.saturated;
    pxcFile.release(); 
    
    return props;
  }

  Mat registerRGB2Depth(const Mat&raw_RGB,const Mat&ZMap,const Mat&UV)
  {
    return registerRGB2Depth_adv(raw_RGB,ZMap,UV).registration;
  }

  PXCRegistration registerRGB2Depth_adv(const Mat&rawRGB,const Mat&ZMap,const Mat&UV)
  {
    // check preconditions
    assert(UV.size() == ZMap.size());

    // create the registered RGB image.
    Mat RGB = Mat(ZMap.rows, ZMap.cols, DataType<Vec3b>::type,Scalar::all(0));
    Mat invalid = Mat(ZMap.rows,ZMap.cols,DataType<uchar>::type,Scalar::all(1));
    assert(RGB.size().area() > 0);
    
    // use the UV map
    Point tl(UV.cols-1,UV.rows-1), br(0,0);
    for(int Yto = 0; Yto < UV.rows; Yto++)
      for(int Xto = 0; Xto < UV.cols; Xto++)
      {
	Vec2f uv = UV.at<Vec2f>(Yto,Xto);
	int XrawFrom = uv[0]*UV.cols + .5;
	int YrawFrom = uv[1]*UV.rows + .5;
	int Xfrom = clamp<int>(0,XrawFrom,ZMap.cols - 1);
	int Yfrom = clamp<int>(0,YrawFrom,ZMap.rows - 1);
	if(Xfrom == XrawFrom and Yfrom == YrawFrom)
	{
	  tl.x = std::min(tl.x,Xfrom);
	  tl.y = std::min(tl.y,Yfrom);
	  br.x = std::max(br.x,Xfrom);
	  br.y = std::max(br.y,Yfrom);
	}

	RGB.at<Vec3b>(Yto,Xto) = rawRGB.at<Vec3b>(Yfrom,Xfrom);
	invalid.at<uchar>(Yto,Xto) = 0;
	//printf("Mapped %d %d to %d %d\n",Xfrom,Yfrom,Xto,Yto);
      }
    assert(RGB.size().area() > 0);
    RGB = fillHoles<Vec3b>(RGB,invalid); 
    assert(RGB.size().area() > 0);

    return {RGB,tl,br};
  }
    
  void loadRGBOnZ(std::string filename, cv::Mat&RGB, cv::Mat&Z)
  {
    /// prepare the RGB
    // load the raw data
    Mat ZMap, ZConf, UV, rawRGB;
    PCX_Props props = loadPCX_RawZ(filename, ZMap, ZConf, UV);
    loadPCX_RGB(filename,rawRGB); 
    resize(rawRGB,rawRGB,ZMap.size());
    float lowConf = params::depth_low_conf, saturated = props.saturated;

    // register the RGB
    RGB = registerRGB2Depth(rawRGB,ZMap,UV);
    
    /// prepare the Z
    Z = Mat(ZMap.rows, ZMap.cols, DataType<float>::type);
    // mm -> cm by /10, also, handle range thresholds
    for(int rIter = 0; rIter < ZMap.rows; rIter++)
      for(int cIter = 0; cIter < ZMap.cols; cIter++)
      {
	float z = ZMap.at<unsigned short>(rIter,cIter);
	Z.at<float>(rIter,cIter) = clamp<float>(0,z/10,params::MAX_Z());
      }
  }
  
  void loadPCX_Z(const string&filename,Mat&Z)
  {
    //printf("loadPCX_Z %s\n",filename.c_str());
    
    // load the raw depth data
    Mat ZMap, ZConf, UV, RGB;
    PCX_Props props = loadPCX_RawZ(filename, ZMap, ZConf, UV);
    loadPCX_RGB(filename,RGB);
    float lowConf = params::depth_low_conf, saturated = props.saturated;
    
    // register the depth data into Z...
    float nan = numeric_limits<float>::quiet_NaN();
    Z = Mat(ZMap.rows,ZMap.cols,DataType<float>::type,nan); 
    Mat invalid = Mat(Z.rows,Z.cols,DataType<uchar>::type,Scalar::all(1));
    Mat conf = Mat(Z.rows,Z.cols,DataType<uchar>::type,Scalar::all(0));
    double conf_ct = 0;
    for(int rIter = 0; rIter < ZMap.rows; rIter++)
      for(int cIter = 0; cIter < ZMap.cols; cIter++)
      {
	bool is_far  = ZMap.at<unsigned short>(rIter,cIter) > 2500;
	bool is_sat  = ZMap.at<unsigned short>(rIter,cIter) == saturated;
	bool is_conf = ZConf.at<unsigned short>(rIter,cIter) > lowConf;
	
	// update Z
	Vec2f uv = UV.at<Vec2f>(rIter,cIter);
	int im_col = clamp<int>(0,uv[0]*Z.cols + .5,Z.cols-1);
	int im_row = clamp<int>(0,uv[1]*Z.rows + .5,Z.rows-1);
	
	// apply translation from Z viewport to RGB viewport
	
	//printf("(%d, %d) => (%d, %d)\n",rIter,cIter,im_row,im_col);
	// mm -> cm by /10
	float z = ((float)ZMap.at<unsigned short>(rIter,cIter))/10;
	Z.at<float>(im_row,im_col) = clamp<float>(0,z,params::MAX_Z());
	
	// update flag arrays
	//invalid.at<uchar>(im_row,im_col) = !(is_conf | is_far);
	//conf.at<uchar>(im_row,im_col) = is_conf;	
	
	// don't use confidence to update arrays
	invalid.at<uchar>(im_row,im_col) = 0;
	conf.at<uchar>(im_row,im_col) = 1;	
	
	// use confidnece alone to update flag arrays
	//invalid.at<uchar>(im_row,im_col) = !is_conf;
	//conf.at<uchar>(im_row,im_col) = is_conf;	
	conf_ct += is_conf;
      }
    Mat registeredZ = Z.clone();
    
    // finally, fill in depth holes
    //printf("%f%% confidence\n",(float)100*conf_ct/ZMap.size().area());
    assert(Z.size().area() > 0 && invalid.size().area() > 0);
    Z = fillDepthHoles(Z,invalid,conf);
    
    // apply the median blur?
    medianBlur(Z,Z,5);
    
    resize(Z,Z,RGB.size());
    //resize(ZMap,Z,RGB.size());
    
    // DEBUG: DISPLAY
    //imageeq("DEBUG: ",registeredZ);
    //imageeq("DEBUG: RawZ",ZMap);
    //imageeq("DEBUG: Loaded Z",Z);
    //image_safe("DEBUG: Loaded RGB",RGB);
  }
  
  Rect loadBB_PCX(const string& filename, string bb_name)
  {
    FileStorage labels(filename,FileStorage::READ);
    if(!labels.isOpened())
    {
      printf("failed to open %s\n",filename.c_str());
      assert(labels.isOpened());
    }

	// try loading the RGB BB directly
    Rect BB = loadRect(labels,bb_name);

    labels.release();
    return BB;
  }
  
  Rect loadRandomExamplePCX(string&directory,Mat&RGB,Mat&Z)
  {
    string stem = directory + "/" + randomStem(directory,".gz");
    printf("loadRandomExamplePCX from %s\n",stem.c_str());
    loadPCX_RGB(stem + ".gz",RGB);
    loadPCX_Z(stem + ".gz",Z);
    FileStorage labels(stem + ".labels.yml",FileStorage::READ);
    Rect BB = loadRect(labels,"HandBB");
    labels.release();
    return BB;
  }
  
  bool PXCFile_PXCFired(const string&filename)
  {
    FileStorage labels(filename,FileStorage::READ);
    assert(labels.isOpened());
    assert(!labels["PXCFired"].empty());
    bool pxcFired; labels["PXCFired"] >> pxcFired;
    labels.release();
    return pxcFired;
  }

	Rect rectZtoRGB(Rect zpos, Mat&UV, Size&RGB)
	{
		// just transluate the entire rect using the center point?
		// update center (works)
		int old_cen_y = zpos.y+zpos.height/2, old_cen_x = zpos.x+zpos.width/2;
		printf("old center = (%d, %d)\n",old_cen_y,old_cen_x);
		Vec2f cen_uv = UV.at<Vec2f>(old_cen_y,old_cen_x);
		int cen_x = clamp<int>(0,(cen_uv[0]*RGB.width +  .5),RGB.width-1);
		int cen_y = clamp<int>(0,(cen_uv[1]*RGB.height + .5),RGB.height-1);
		printf("new center = (%d, %d)\n",cen_y, cen_x);
		// update size?
		Size newSize(
			2*zpos.width*params::H_Z_FOV/params::H_RGB_FOV,
			2*zpos.height*params::V_Z_FOV/params::V_RGB_FOV);

		// return
		return rectFromCenter(Point(cen_x,cen_y),newSize);

		// translate the corners indepndently
		//return Rect(
		//	pointZtoRGB(zpos.tl(),UV,RGB),
		//	pointZtoRGB(zpos.br(),UV,RGB));	
	}
	
  Rect rectRGBtoZ(Rect rgbPos, Mat&UV)
  {
    assert(UV.type() == DataType<Vec2f>::type);
    // UV[Zx,Zy] = (RGBx,RGBy)
    // try to compute a translation
    Vec2d t(0,0);
    double n = 0;
    // loop over Z positions
    for(int yIter = 0; yIter < UV.rows; yIter++)
      for(int xIter = 0; xIter < UV.cols; xIter++)
      {
	Vec2f uv = UV.at<Vec2f>(yIter,xIter);
	uv[0] = uv[0] * UV.cols + .5;
	uv[1] = uv[1] * UV.rows + .5;	
	Point rgb_pt(uv[0],uv[1]);

	if(rgbPos.contains(rgb_pt))
	{
	  t += Vec2d(uv[0] - xIter,uv[1] - yIter);
	  n ++;
	}
      }
    t /= n;
    
    rgbPos.x -= t[0];
    rgbPos.y -= t[1];
    log_once(safe_printf("translation % % to % %",t[0],t[1],rgbPos.x,rgbPos.y));

    return rgbPos;
  }

#ifndef WIN32
  DetectorResult convert(const PXCDetection& det)
  {
    DetectorResult result(new Detection());
    
    result->blob = det.blob;
    result->BB = det.bb;
    result->src_filename = det.filename;
    for(auto&sub_det : det.parts)
      result->emplace_part(sub_det.first,*convert(sub_det.second));
    
    return result;
  }
#endif
}
