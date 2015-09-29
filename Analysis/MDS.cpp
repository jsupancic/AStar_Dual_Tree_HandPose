/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "MDS.hpp"
#include "util.hpp"
#include "Video.hpp"
#include "TestModel.hpp"
#include "InverseKinematics.hpp"
#include "BaselineDetection.hpp"

namespace deformable_depth
{
#ifdef DD_ENABLE_HAND_SYNTH      
  void write_ps(ofstream&ofs,map<string,AnnotationBoundingBox>&poss)
  {    
    Rect handBB = poss["HandBB"];
    Point2d hand_center = rectCenter(handBB);

    for(int iter = 1; iter <= 5; ++iter)
    {
      // get the part
      string part_name = safe_printf("dist_phalan_%",iter);
      Rect part = handBB;
      if(poss.find(part_name) != poss.end())
	part = poss.at(part_name);
      Point2d part_center = rectCenter(part);
      
      // normalize the part
      part_center.x -= hand_center.x;
      part_center.y -= hand_center.y;
      double nf = std::sqrt(handBB.size().area());
      part_center.x /= nf;
      part_center.y /= nf;
      
      // write
      ofs << part_center.x << ",";
      ofs << part_center.y << ",";
    }    
  }

  void write_thetas(ofstream&ofs,PoseRegressionPoint&ik_res)
  {
    for(float theta : ik_res.hand_pose)
      ofs << theta << ",";
  }

  void write_ranges(ofstream&ofs, ImRGBZ&im,map<string,AnnotationBoundingBox>&poss,PoseRegressionPoint&ik_res)
  {
    Rect handBB = poss["HandBB"];
    ofs << ik_res.cam_spec.theta << ",";
    ofs << ik_res.cam_spec.phi << ",";
    ofs << ik_res.cam_spec.tilt << ",";
 
    handBB = clamp(im.Z,handBB);
    if(handBB.size().area() > 0)
    {
      Extrema hand_range = extrema(im.Z(handBB));
      ofs << hand_range.min << ",";
      ofs << hand_range.max << ",";
    }
    else
    {
      ofs << qnan << ",";
      ofs << qnan << ",";
    }
    Extrema im_range = extrema(im.Z);
    ofs << im_range.min << ",";
    ofs << im_range.max << ",";
  }

  void export_mds_data()
  {
    ofstream ofs(params::out_dir() + "/poses_by_dataset.csv");
    ofs << "dataset,typeID,";
    for(int iter = 1; iter <= 5; ++iter)
    {
      ofs << safe_printf("dist_phalan_%_x,",iter);
      ofs << safe_printf("dist_phalan_%_y,",iter);
    }
    ofs << endl;    

    vector<string> videos = test_video_filenames();
    cout << "got videos: " << videos.size() << endl;
    TaskBlock do_mds("do_mds");
    for(auto video_filename : videos)
    {
      auto vid = load_video(video_filename);
      for(int iter = 0; iter < vid->getNumberOfFrames(); ++iter)
      {
	if(vid->is_frame_annotated(iter))
	{	  
	  do_mds.add_callee([&,iter,vid]{
	      // show the frame
	      auto frame = vid->getFrame(iter,true);
	      auto poss = frame->get_positives();
	      auto im = frame->load_im();
	      Rect handBB = poss["HandBB"];
	      Point2d hand_center = rectCenter(handBB);
	      if(handBB == Rect())
		return;
	      auto vis = image_datum(*frame,false);	  	  
	      log_im(safe_printf("%_%_mds_frame_vis",iter,vid->get_name()),vis);

	      // regress the joint angles with IK
	      DetectorResult det = fromMetadata(*frame).front();
	      BaselineDetection base(*det);
	      PoseRegressionPoint ik_res = ik_regress(base);
	      
	      // get the positives and write them
	      static mutex m; lock_guard<mutex> l(m);
	      ofs << vid->get_name() << "," << vid_type_id(vid->get_name()) << ",";
	      write_ps(ofs,poss);
	      write_thetas(ofs,ik_res);
	      write_ranges(ofs,*im,poss,ik_res);
	      ofs << endl;
	    }); 	    
	}
      }
    }
    do_mds.execute();
  }
#else
  void export_mds_data(){throw std::logic_error("unsupported");}
#endif
}
