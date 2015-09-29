/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ManualPose.hpp"  
#include "LibHandRenderer.hpp"

namespace deformable_depth
{
  void pose_hand()
  {
#ifdef DD_ENABLE_HAND_SYNTH
    // prepare for rendering.
    renderers::with_arm();
    LibHandRenderer * renderer = renderers::segmentation();
    renderer->get_hand_pose() = libhand::FullHandPose(renderer->get_scene_spec().num_bones());
    renderer->render();
    
    string win_name = "Interact";
    int key_code;
    cv::namedWindow(win_name);
    int active_joint = 0;
    int fliplr = 0;
    int old_joint = 0;
    int bend = 0, side = 0, twist = 0;
    cv::createTrackbar("fliplr",win_name,&fliplr,1);	  
    cv::createTrackbar("joint",win_name,&active_joint,renderer->get_hand_pose().num_joints() - 1);	  
    cv::createTrackbar("bend",win_name,&bend,360);	  
    cv::createTrackbar("side",win_name,&side,360);	  
    cv::createTrackbar("twist",win_name,&twist,360);	  
    do
    {
      key_code = waitKey(30);  
      if(old_joint != active_joint)
      {
	cv::setTrackbarPos("bend",win_name.c_str(),rad2deg(renderer->get_hand_pose().bend(active_joint)));
	cv::setTrackbarPos("side",win_name.c_str(),rad2deg(renderer->get_hand_pose().side(active_joint)));
	cv::setTrackbarPos("twist",win_name.c_str(),rad2deg(renderer->get_hand_pose().twist(active_joint)));
	old_joint = active_joint;
      }      
      else
      {
	renderer->get_hand_pose().bend(active_joint) = deg2rad(bend);
	renderer->get_hand_pose().side(active_joint) = deg2rad(side);
	renderer->get_hand_pose().twist(active_joint) = deg2rad(twist);
      }
      renderer->render();
      Mat Z = renderer->getDepth();
      Mat vis_Z = imageeq(win_name.c_str(),Z,false,false);   
      Mat seg = renderer->getRGB();
      if(fliplr)
      {
	cv::flip(vis_Z,vis_Z,+1);
	cv::flip(seg,seg,+1);
      }
      image_safe(win_name,horizCat(imVGA(vis_Z),imVGA(seg)));
    } while(key_code != 'q');
#else
    throw std::logic_error("unsupported");
#endif    
  }
}

