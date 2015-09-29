/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifdef DD_ENABLE_OPENNI
#include "ONI_Video.hpp"
#include "Log.hpp"
#include "util.hpp"
#include <boost/filesystem.hpp>
#include <boost/regex.h>
#include "PS1080.h"
#include "PSLink.h"

namespace deformable_depth
{
  static mutex ONI_exclusion;
  
  static float lookup_camera_translation(string video_name)
  {
    if(boost::regex_match(video_name,boost::regex(".*library.*")))
      return 50;
    else if(boost::regex_match(video_name,boost::regex(".*parking.*")))
      return 100;
    else
      return 0;
  }  
  
  static float lookup_metric_correction(string video_name)
  {
    if(boost::regex_match(video_name,boost::regex(".*sequence.*")))
      return 1;
    else
      return 1;
  }    

  static bool flip_tb(string video_name)
  {
    if(boost::regex_match(video_name,boost::regex(".*(wheel|road|driving).*")))
      return true;
    else
      return false;
  }
  
  static int frame_skip(string video_name)
  {
    if(boost::regex_match(video_name,boost::regex(".*parking.*")))
      return 100;
    else
      return 0;
  }
  
  void ONI_Video::cvt_oni_to_cv(
    openni::VideoFrameRef&colorFrame, 
    openni::VideoFrameRef&depthFrame,
    cv::Mat&cv_depth,
    cv::Mat&cv_rgb, CustomCamera& camera)
  {    
    // retrieve the frame
    const uint16_t*p_depth = static_cast<const uint16_t*>(depthFrame.getData());
    const uint8_t *p_color = static_cast<const uint8_t* >(colorFrame.getData());
    cv_depth = cv::Mat(depthFrame.getHeight(),depthFrame.getWidth(),CV_16UC1,
		      (void*)p_depth,  sizeof(uint16_t)*depthFrame.getWidth());
    cv_rgb = cv::Mat(colorFrame.getHeight(),colorFrame.getWidth(),CV_8UC3 ,
		      (void*)p_color,3*sizeof(uint8_t )*colorFrame.getWidth());
    cv::resize(cv_depth,cv_depth,Size(320,240),params::DEPTH_INTER_STRATEGY);
    cv::resize(cv_rgb,cv_rgb,Size(320,240));
    // OpenNI uses RGB, while OpenCV likes BGR
    cv::cvtColor(cv_rgb,cv_rgb,CV_RGB2BGR);
    // OpenNI uses uint16 depth (mm), while DD uses float depths (cm)
    cv_depth.convertTo(cv_depth,CV_32F);
#ifdef DD_CXX11
    cout << "median depth: " << medianApx(cv_depth,Rect(Point(0,0),cv_depth.size())) << endl;
#endif
    cv_depth /= 10;
#ifdef DD_CXX11
    cout << "median depth: " << medianApx(cv_depth,Rect(Point(0,0),cv_depth.size())) << endl;
#endif
    
    if(flip_tb(get_name()))
    {
      cv::flip(cv_depth,cv_depth,0);
      cv::flip(cv_rgb,cv_rgb,0);
    }

    float tz = lookup_camera_translation(get_name());
    
    // now, replace all zero values with NaN because they are invalid.
    if(!g_params.has_key("ONI_ZERO_VALID"))
      for(int rIter = 0; rIter < cv_depth.rows; rIter++)
	for(int cIter = 0; cIter < cv_depth.cols; cIter++)
	{
	  float&value = cv_depth.at<float>(rIter,cIter);
	  
	  // apply translation
	  //cout << value << " => ";
	  if(goodNumber(value))
	    value -= tz;
	  
	  // mark all forms of invalid pixels
	  if(!goodNumber(value) || value <= params::MIN_Z())
	    value = qnan;
	  
	  // clamp inf pixels
	  if(value == inf || (goodNumber(value) && value > params::MAX_Z()))
	    value = params::MAX_Z();
	  //cout << value << endl;
	}
    //imageeq("ONI depth with holes",cv_depth,true,false);
    if(!g_params.has_key("DISP_DEPTH"))
      cv_depth = fillDepthHoles(cv_depth,5);
    //imageeq("ONI depth without holes",cv_depth,true,false);
#ifdef DD_CXX11
    cout << "median depth: " << medianApx(cv_depth,Rect(Point(0,0),cv_depth.size())) << endl;
#endif    
    
    // set the correction factor for ONI videos
    camera.setMetricCorrection(lookup_metric_correction(get_name()));
    
    // log focal lengths?
    //float zpx, zpy;
    //depth_video.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE,&zpx);
    //depth_video.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE,&zpy);
    //log_once(printfpp("zpx = %f zpy = %f name = %s",zpx,zpy,get_name().c_str()));
    
    // translate the camera
    if(tz > 0)
    {
      float fx = camera.focalX(), fy = camera.focalY();
      //camera.focalX()
      
      // compute the crop resolution
      float cropWidth  = cv_rgb.cols/fx * (fx - tz);
      float cropHeight = cv_rgb.rows/fy * (fy - tz);
      
      // TODO: show divide by 2?
      float cropX1 = (cv_rgb.cols - cropWidth)/2;//tz*cv_rgb.cols / (2 * (fx+tz));
      float cropY1 = (cv_rgb.rows - cropHeight)/2;//tz*cv_rgb.rows / (2 * (fy+tz));
      float cropX2 = cv_rgb.cols - cropX1;
      float cropY2 = cv_rgb.rows - cropY1;
      Rect roi(Point(cropX1,cropY1),Point(cropX2,cropY2));
      cout << "ROI: " << roi << endl;
      cv_depth = cv_depth(roi).clone();
      cv_rgb   = cv_rgb(roi).clone();
      cv::resize(cv_depth,cv_depth,Size(320,240),params::DEPTH_INTER_STRATEGY);
      cv::resize(cv_rgb,cv_rgb,Size(320,240));
    }
        
    //imagesc("cv_depth",cv_depth); 
    //image_safe("cv_rgb",cv_rgb);
    waitKey_safe(10);    
  }
    
  shared_ptr<MetaData_YML_Backed> ONI_Video::getFrame(int iter,bool read_only)
  {
    iter += frame_skip(get_name());
    iter = clamp<int>(frame_skip(get_name()),iter,control->getNumberOfFrames(depth_video)-1);
    
    unique_lock<mutex> critical_section(ONI_exclusion);
    unique_lock<mutex> l(monitor);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // create left & right then assemble.
    auto metadata_left  = getFrame_simple(iter,read_only,"left");
    auto metadata_right = getFrame_simple(iter,read_only,"right");
    map<string,shared_ptr<MetaData_YML_Backed> > comps{{"left_hand",metadata_left},{"right_hand",metadata_right}};
    auto metadata = make_shared<MetaDataAggregate>(comps);

    std::atomic_thread_fence(std::memory_order_seq_cst);    
    return metadata;
  }

  shared_ptr< MetaData_YML_Backed > ONI_Video::getFrame_simple(int iter,bool read_only,string lr_id)
  {
    // filename  
    string raw_md_filename = safe_printf("%.frame%",filename.c_str(),iter);    
    string metadata_filename = 
      safe_printf("%.frame%_%",filename.c_str(),lr_id,iter);    

    // select the frame from video
    if(control)
    {
      control->setRepeatEnabled(true);
      control->seek(depth_video,iter);
      //control->seek(color_video,iter);
    }
    openni::VideoFrameRef colorFrame, depthFrame;
    color_video.readFrame(&colorFrame);
    depth_video.readFrame(&depthFrame);
    ensure(colorFrame.isValid() && depthFrame.isValid());
    
    // ONI camera paramters
    double deg_per_rad = 57.2957795;
    double hfov = deg_per_rad*depth_video.getHorizontalFieldOfView();
    double vfov = deg_per_rad*depth_video.getVerticalFieldOfView();
    double rgb_hfov = deg_per_rad*color_video.getHorizontalFieldOfView();
    double rgb_vfov = deg_per_rad*color_video.getVerticalFieldOfView();
    log_once(printfpp("ONI Camera(Depth) hfov = %f vfov = %f" ,hfov,vfov));
    log_once(safe_printf("ONI Camera(Depth) hres = % vres = %" ,depthFrame.getHeight(),depthFrame.getWidth()));
    log_once(printfpp("ONI Camera(Color) hfov = %f vfov = %f" ,rgb_hfov,rgb_vfov));
    log_once(safe_printf("ONI Camera(Color) hres = % vres = %" ,colorFrame.getHeight(),colorFrame.getWidth()));
    if(depth_video.isPropertySupported(LINK_PROP_ZERO_PLANE_DISTANCE))
    {
      double zpd; depth_video.getProperty<double>(LINK_PROP_ZERO_PLANE_DISTANCE,&zpd);
      log_once(safe_printf("ONI Camera(Depth) zpd = %",zpd));
    }
    CustomCamera camera(hfov,vfov, 320,240);
    
    // now, convert from ONI frames to OpenCV frames.
    cv::Mat cv_rgb,cv_depth;
    cvt_oni_to_cv(colorFrame, depthFrame,cv_depth,cv_rgb,camera);
      
    // convert the frame to an Deformable_depth image.
    ImRGBZ im(cv_rgb,cv_depth,metadata_filename+".im.gz",camera);
    
    // now, convert the frame to deformable_depth metatadata
    //image_safe("ONI Frame Final",horizCat(imageeq("",im.Z,true,false),im.RGB));
    shared_ptr<Metadata_Simple> metadata;
    shared_ptr<Metadata_Simple> old_metadata(new Metadata_Simple(raw_md_filename,true,true,false));
    old_metadata->setIm(im);
    if(lr_id == "left" and old_metadata->leftP())
    {
      metadata = old_metadata;
    }
    else if(lr_id == "right" and !old_metadata->leftP())
    {
      metadata = old_metadata;
    }
    else
    {
      //cout << old_metadata->get_positives().size() << endl;
      metadata = shared_ptr<Metadata_Simple>(new Metadata_Simple(metadata_filename,read_only,true,false));
      metadata->setIm(im);
    }
    
    if(read_only)
      boost::filesystem::remove(metadata_filename);

    return metadata;
  }

  ONI_Video::ONI_Video(string filename) : filename(filename)
  {
    unique_lock<mutex> critical_section(ONI_exclusion);
    // configure OpenNI2
    ensure(openni::OpenNI::initialize() == openni::STATUS_OK);
    ensure(device.open(filename.c_str()) == openni::STATUS_OK);
    ensure(depth_video.create(device,openni::SENSOR_DEPTH) == openni::STATUS_OK);
    ensure(color_video.create(device,openni::SENSOR_COLOR) == openni::STATUS_OK);
    ensure(depth_video.start() == openni::STATUS_OK);
    ensure(color_video.start() == openni::STATUS_OK);
    ensure(depth_video.isValid() && color_video.isValid());
    control = device.getPlaybackControl();
    if(control)
    {
      control->setSpeed(-1); // pause, requrie seek to change frame
    }
  }

  ONI_Video::~ONI_Video()
  {
  }
  
  int ONI_Video::getNumberOfFrames()
  {
    unique_lock<mutex> critical_section(ONI_exclusion);
    return control->getNumberOfFrames(depth_video) - frame_skip(get_name());
  }
  
  string ONI_Video::get_name()
  {
    boost::filesystem::path v_path(filename);
    return v_path.leaf().string();
  }
}

#endif
