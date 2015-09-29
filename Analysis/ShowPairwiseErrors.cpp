/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ShowPairwiseErrors.hpp"
#include "LibHandRenderer.hpp"
#include "InverseKinematics.hpp"
#include "Colors.hpp"

namespace deformable_depth
{
#ifdef DD_ENABLE_HAND_SYNTH
  struct Centering
  {
    map< string, Vec3d > xs;
    Vec3d offset;
  };

  static Centering center(vector<string>&part_names,map< string, Vec3d >&xs)
  {
    Vec3d center(0,0,0);
    for(auto && part_name : part_names)
    {
      center += xs.at(part_name);
    }
    center[0] /= part_names.size();
    center[1] /= part_names.size();
    center[2] /= part_names.size();

    auto xs_centered = xs;
    for(auto && pair : xs_centered)
      pair.second -= center;
    return Centering{xs_centered,center};
  }

  struct AlignedDist
  {
    Vec2d shift_dst;
    double dist;
  };

  static AlignedDist dist_unaligned(double target, IKBinaryDatabase::XYZ_TYR_Pair&src, IKBinaryDatabase::XYZ_TYR_Pair&dst)
  {
    vector<string> part_names{"Z_J11","Z_J21","Z_J31","Z_J41","Z_J51","Z_P0","Z_P0","Z_P1"};
    for(auto && dd_part_name : part_names)
    {
      dd_part_name = dd2libhand(dd_part_name);
    }

    double mean_dist = 0;
    double max_dist = -inf;
    for(auto && part_name : part_names)
    {      
      Vec3d vec_s = src.joint_position_map.at(part_name);
      Vec3d vec_d = dst.joint_position_map.at(part_name);
      double dist = 
	(vec_s[0] - vec_d[0])*(vec_s[0] - vec_d[0]) + 
	(vec_s[1] - vec_d[1])*(vec_s[1] - vec_d[1]) + 
	(vec_s[2] - vec_d[2])*(vec_s[2] - vec_d[2]);
      dist = std::sqrt(dist);
      mean_dist += dist;
      max_dist = std::max<double>(max_dist,dist);
    }
    mean_dist /= part_names.size();
    if(mean_dist == 0)
      mean_dist = inf;
    if(max_dist == 0)
      max_dist = inf;

    Vec2d offset(0,0);
    return AlignedDist{offset,std::abs(target - mean_dist)};
  }

  static AlignedDist dist_aligned(double target, IKBinaryDatabase::XYZ_TYR_Pair&src, IKBinaryDatabase::XYZ_TYR_Pair&dst)
  {    
    vector<string> part_names{"Z_J11","Z_J21","Z_J31","Z_J41","Z_J51","Z_P0","Z_P0","Z_P1"};
    for(auto && dd_part_name : part_names)
    {
      dd_part_name = dd2libhand(dd_part_name);
    }
    auto src_center = center(part_names,src.joint_position_map);
    auto dst_center = center(part_names,dst.joint_position_map);

    double mean_dist = 0;
    double max_dist = -inf;
    for(auto && part_name : part_names)
    {      
      Vec3d vec_s = src_center.xs.at(part_name);
      Vec3d vec_d = dst_center.xs.at(part_name);
      double dist = 
	(vec_s[0] - vec_d[0])*(vec_s[0] - vec_d[0]) + 
	(vec_s[1] - vec_d[1])*(vec_s[1] - vec_d[1]);
      dist = std::sqrt(dist);
      mean_dist += dist;
      max_dist = std::max<double>(max_dist,dist);
    }
    mean_dist /= part_names.size();
    if(mean_dist == 0)
      mean_dist = inf;
    if(max_dist == 0)
      max_dist = inf;

    Vec2d offset(src_center.offset[0] - dst_center.offset[0],src_center.offset[1] - dst_center.offset[1]);
    return AlignedDist{offset,std::abs(target - mean_dist)};
  }

  static AlignedDist dist(double target, IKBinaryDatabase::XYZ_TYR_Pair&src, IKBinaryDatabase::XYZ_TYR_Pair&dst)
  {
    return dist_unaligned(target,src,dst);
  }

  void show_matched_pair(Mat&srcZ,Mat&dstZ,string text)
  {
    // Now render the source and its match.
    Mat vizSrc = imageeq("",srcZ,false,false);
    //
    Mat vizDst = imageeq("",dstZ,false,false);
    Mat viz_cat = horizCat(vizSrc,vizDst);

    // merge?
    Mat viz_both(srcZ.rows,srcZ.cols,DataType<Vec3b>::type,toScalar(INVALID_COLOR));
    for(int yIter = 0; yIter < srcZ.rows; yIter++)
      for(int xIter = 0; xIter < srcZ.cols; xIter++)
      {
	float z_src = srcZ.at<float>(yIter,xIter);
	double src_weight = vizSrc.at<Vec3b>(yIter,xIter)[0] / 255.0;
	float z_dst = dstZ.at<float>(yIter,xIter);
	double dst_weight = vizDst.at<Vec3b>(yIter,xIter)[0] / 255.0;
	if(goodNumber(z_src) and goodNumber(z_dst))
	{
	  viz_both.at<Vec3b>(yIter,xIter) = src_weight*RED + dst_weight*BLUE;

	  // if(z_src < z_dst)
	  // {
	  //   viz_both.at<Vec3b>(yIter,xIter) = src_weight*RED;
	  // }
	  // else
	  // {
	  //   viz_both.at<Vec3b>(yIter,xIter) = 
	  // }
	}
	else if(goodNumber(z_src))
	{
	  viz_both.at<Vec3b>(yIter,xIter) = src_weight*RED;
	}
	else if(goodNumber(z_dst))
	{
	  viz_both.at<Vec3b>(yIter,xIter) = dst_weight*BLUE;
	}
      }
    Mat viz = vertCat(viz_both,viz_cat);    
    log_im("viz",vertCat(viz,image_text(text)));    
  }

  void show_pairwise_errors(LibHandRenderer * renderer,int src_index)
  {
    // 
    vector< string > databases = find_files("data/bin_ik/", boost::regex(".*bin"));
    IKBinaryDatabase ik_db_src(*renderer,databases.front());
    for(int iter = 0; iter < src_index; ++iter)
      ik_db_src.next_record();
    IKBinaryDatabase::XYZ_TYR_Pair src = ik_db_src.next_record();

    // find something...
    const double target_dist = fromString<double>(g_params.require("FINGER_DIST_THRESH"));
    IKBinaryDatabase::XYZ_TYR_Pair best_match;
    Vec2d offset;
    double best_match_dist = inf;
    TaskBlock search_database("search_database");
    for(string database_filename : databases)
    {
      search_database.add_callee([&,database_filename]()
				 {
				   IKBinaryDatabase ik_db_dst(*renderer,database_filename);
				   int CHECK_COUNT = 500000;
				   for(int iter = 0; ik_db_dst.hasNext() and iter < CHECK_COUNT; ++iter)
				   {
				     IKBinaryDatabase::XYZ_TYR_Pair dst = ik_db_dst.next_record();
				     if(not dst.valid)
				       continue;
				     auto aligned_dist = dist(target_dist,src,dst);
				     double cur_dist = aligned_dist.dist;
				     static mutex m; lock_guard<decltype(m)> l(m);
				     if(cur_dist < best_match_dist)
				     {
				       best_match_dist = cur_dist;
				       best_match = dst;
				       offset = aligned_dist.shift_dst;
				     }
				     progressBars->set_progress("show_pairwise_errors",iter,CHECK_COUNT);
				   }      
				 });
    }
    search_database.execute();
    progressBars->set_progress("show_pairwise_errors",0,0);
    string message = safe_printf("INFO: off target of % by %",target_dist,best_match_dist);
    log_once(message);

    // 
    Mat translationMat = getRotationMatrix2D(Point2d(0,0),0,1);
    translationMat.at<double>(0,2) = -offset[0];
    translationMat.at<double>(1,2) = -offset[1];
    log_file << "translationMat: "  << translationMat << endl;

    // Now render the source and its match.
    renderer->get_cam_spec() = src.cam_spec;
    renderer->get_hand_pose() = src.hand_pose;
    renderer->render();
    Mat srcZ = renderer->getDepth();
    cv::warpAffine(srcZ,srcZ,translationMat,srcZ.size());
    //
    renderer->get_cam_spec() = best_match.cam_spec;
    renderer->get_hand_pose() = best_match.hand_pose;
    renderer->render();
    Mat dstZ = renderer->getDepth();
    // 
    show_matched_pair(srcZ,dstZ,message);
  }

  void show_pairwise_errors()
  {
    // 
    LibHandRenderer * renderer = renderers::no_arm();
    renderer->get_hand_pose() = libhand::FullHandPose(renderer->get_scene_spec().num_bones());
    renderer->render();

    for(int iter = 0; iter < 500; iter += 10)
      show_pairwise_errors(renderer,iter);
  }
#else
  void show_pairwise_errors(){throw std::logic_error("unsupported");}
#endif
}
