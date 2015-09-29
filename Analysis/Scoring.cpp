/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "Scoring.hpp"
#include "opencv2/opencv.hpp"
#include "BaselineDetection.hpp"
#include "Orthography.hpp"
#include "Plot.hpp"
#include "PXCSupport.hpp"
#include "Eval.hpp"
#include "Video.hpp"
#include "ConvexityDefects.hpp"
#include "HL_IK.hpp"
#include "Colors.hpp"
#include "RegEx.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
    
  ///
  /// SECTION: Analyze
  ///
  double analyze_cm_displacement_3D_l2(
    BaselineDetection&det,MetaData&gt,
    Rect_<double> finger_det_bb,
    Rect_<double> gt_bb,
    shared_ptr<ImRGBZ>&im,
    RectAtDepth&ortho_gt,RectAtDepth&ortho_dt)
  {
    // get the depth of the ground truth.
    Point2d gt_handCenter = rectCenter(gt.get_positives()["HandBB"]);
    double gt_z = im->Z.at<float>(gt_handCenter.y,gt_handCenter.x);
    //double gt_z = extrema(im->Z(gt.get_positives()["HandBB"])).min;
    Point2d dt_handCenter = rectCenter(det.bb);
    //Point2d dt_fingerCenter = rectCenter(finger_det_bb);

    // derive the detection's Z
    double dt_z = im->Z.at<float>(dt_handCenter.y,dt_handCenter.x);    
    if(std::abs(dt_z - gt_z) > 10)
    {
      dt_z = gt_z + thread_rand()%60 - 60.0/2;
      //dt_z = gt_z;
    }
    if(g_params.option_is_set("ANALYZE_METRIC_CM_WITH_GT_Z"))
    {
      dt_z = gt_z;
    }
    //if(!rectContains(im->Z,det.bb))
      //log_file << det.bb << " not in " << im->Z.size() << endl;
    //double dt_z = extrema(im->Z(det.bb)).min;
    log_once(printfpp("gt_z = %f dt_z = %f",gt_z,dt_z));
    
    // transfer the locations into world coordinates (cms)
    Rect_<double>ortho_gt_bb = map2ortho_cm(gt_bb,im->camera,gt_z);
    Rect_<double>ortho_dt_bb = map2ortho_cm(finger_det_bb,im->camera,dt_z);
    log_file << "ortho_gt_bb = " << ortho_gt_bb << endl;
    log_file << "ortho_dt_bb = " << ortho_dt_bb << endl;
    ortho_gt = RectAtDepth(ortho_gt_bb);
    ortho_dt = RectAtDepth(ortho_dt_bb);
    bool bad_dt_bb = ((Rect_<double>)ortho_dt) == Rect_<double>();
    if(bad_dt_bb)
    {
      log_file << "warning: ortho_dt_bb: " << det.bb << ortho_dt_bb << finger_det_bb << " " << dt_z << endl;
    }
    Point2d gt_pos_cm = rectCenter(ortho_gt);
    Point2d dt_pos_cm = rectCenter(ortho_dt);
    
    // get the Z offset
    if(g_params.has_key("SCORE_FINGER_2D_ORTHO"))
    {
      ortho_gt.depth() = 100;
      ortho_dt.depth() = 100;
    }
    else
    {      
      //ortho_gt.depth() = im->Z.at<float>(rectCenter(gt_bb).y,rectCenter(gt_bb).x);
      //ortho_dt.depth() = im->Z.at<float>(rectCenter(finger_det_bb).y,rectCenter(finger_det_bb).x);
      ortho_gt.depth() = gt_z;
      ortho_dt.depth() = dt_z;
    }
    
    // compute the distance in world coords
    return ortho_gt.dist(ortho_dt);    
  }
  
  double analyze_cm_displacement(
    BaselineDetection&det,MetaData&gt,
    BaselineDetection&finger_det,
    Rect_<double> gt_bb,
    shared_ptr<ImRGBZ>&im,
    RectAtDepth&ortho_gt,RectAtDepth&ortho_dt)
  {    
    // interpolate the finger if required
    Rect_<double> finger_det_bb = finger_det.bb;
    if(finger_det_bb == Rect_<double>())
      finger_det_bb = det.bb; // default to center of hand detection if no finger detection
    finger_det_bb = clamp(im->Z, finger_det_bb);
    if(finger_det_bb == Rect_<double>())
    {
      Rect_<double> im_rect(Point2d(im->cols()/2,im->rows()/2),im->Z.size()); 
      cout << "im_rect: " << im_rect << endl;
      finger_det_bb = im_rect;        // default to center of image if no hand detection
      log_file << "warning: defaulting to center of hand" << endl;
    }
    assert(finger_det_bb != Rect_<double>());
    gt_bb = clamp(im->Z, gt_bb);
    assert(im->Z.rows > 5 && im->Z.cols > 5);
    finger_det_bb = clamp(im->Z, finger_det_bb);
    assert(finger_det_bb != Rect_<double>());
    
    // compute the distance
    if(g_params.has_key("ANALYZE_FINGER_DIST_PIXEL"))
    {
      // hack
      analyze_cm_displacement_3D_l2(det,gt,finger_det_bb,gt_bb,im,ortho_gt,ortho_dt);
      
      // compute pixel dist
      Point2d dt_center = rectCenter(finger_det_bb);
      Point2d gt_center = rectCenter(gt_bb);
      double x_diff = dt_center.x - gt_center.x;
      double y_diff = dt_center.y - gt_center.y;
      return std::sqrt(x_diff * x_diff + y_diff * y_diff);
    }
    else
      return analyze_cm_displacement_3D_l2(det,gt,finger_det_bb,gt_bb,im,ortho_gt,ortho_dt);
  }
  
  void analyze_score_one_finger(BaselineDetection det,MetaData&gt,
				Mat Z, DetectorScores&pcp_score,
				DetectorScores&agnostic_finger_scores,
				vector<double>&cm_displs,
				Mat&total_vis,
				vector<Mat>&per_finger_vis, int iter)
  {
    // 
    Mat finger_vis = imageeq("",Z.clone(),false,false);
    
    // get the part
    string partname = printfpp("dist_phalan_%d",iter); // 6 - iter?
    BaselineDetection finger_det;
    finger_det = (det.parts)[partname];
    
    // get the GT
    Rect_<double> gt_bb = gt.get_positives()[partname];
        
    // project the and score
    shared_ptr<ImRGBZ> im = gt.load_im();
    // find the finger BB
    RectAtDepth ortho_gt, ortho_dt;
    double dist = analyze_cm_displacement(det,gt,finger_det,gt_bb,im,ortho_gt, ortho_dt); ;
    if(gt_bb == Rect_<double>() or gt_bb.tl() == Point2d() or gt_bb.size().area() == 0)
    {
      // score missing gts as correct.
      dist = 0;
    }
    bool correct = ortho_gt.dist(ortho_dt) < params::finger_correct_distance_threshold();
    cm_displs.push_back(dist);
    double min_value = -std::numeric_limits<float>::max();
    double resp = qnan;
    if(finger_det.bb != Rect() and finger_det.bb.area() > 0 and ortho_dt.area() > 0)
    {
      if(not goodNumber(finger_det.resp) or finger_det.resp <= min_value)
	finger_det.resp = det.resp;
      resp = std::max(min_value,std::min(det.resp,finger_det.resp));
      //if(finger_det.remapped)
      //resp /= 2;
      cout << "finger resp = " << resp << endl;
      bool write_worked = (agnostic_finger_scores.put_detection(
	gt.get_filename(),ortho_dt,resp));
      pcp_score.put_detection(gt.get_filename() + partname, ortho_dt, resp);
      if(!write_worked)
      {
	cout << resp << endl;
	cout << (Rect_<double>)ortho_dt << endl;
	assert(false);
      }
    }
    agnostic_finger_scores.put_ground_truth(gt.get_filename(),ortho_gt);
    pcp_score.put_ground_truth(gt.get_filename() + partname, ortho_gt);
       
    // show the gt
    cout << printfpp("gt_bb = %d %d %d %d",(int)gt_bb.x,(int)gt_bb.y,(int)gt_bb.width,(int)gt_bb.height) << endl;
    rectangle(finger_vis,gt_bb.tl(),gt_bb.br(),Scalar(255,255,0));    
    rectangle(total_vis,gt_bb.tl(),gt_bb.br(),Scalar(255,255,0));
    
    // show the detection
    Scalar det_color = correct?toScalar(GREEN):toScalar(RED);
    rectangle(finger_vis,finger_det.bb.tl(),finger_det.bb.br(),det_color);      
    rectangle(total_vis,finger_det.bb.tl(),finger_det.bb.br(),det_color);   
    Mat line1 = image_text(safe_printf("finger = % dist = % resp = %",partname,dist,resp));
    Mat line2 = image_text(safe_printf("gt_bb = % det_bb = %",gt_bb,finger_det.bb));
    finger_vis = vertCat(finger_vis,vertCat(line1,line2));
    per_finger_vis.push_back(finger_vis);
    assert(finger_vis.type() == DataType<Vec3b>::type); 

    // draw a line from the gt to the detection
    line(finger_vis,rectCenter(gt_bb),rectCenter(finger_det.bb),toScalar(BLUE),det_color);    
    line(total_vis,rectCenter(gt_bb),rectCenter(finger_det.bb),toScalar(BLUE),det_color);    
  }
  
  bool analize_score_one_frame_hand(
    string method_name, Mat Z, BaselineDetection det, MetaData&gt, 
    PerformanceRecord&record, Mat&total_vis)  
  {
    record.getGtCount() = 1;
    
    // Z needs to be float
    assert(Z.type() == DataType<float>::type);
    
    // draw the BB
    cout << "analize_score_one_frame: " << gt.get_filename() << endl;
    cout << "analize_score_one_frame: resp = " << det.resp << endl;
    rectangle(Z,det.bb.tl(),det.bb.br(),Scalar(0,255,0));
    total_vis = imageeq("",Z.clone(),false,false);
    
    // score the root
    auto poss = gt.get_positives();
    Rect handBB = poss["HandBB"];
    if(handBB == Rect())
    {
      return false;
    }
    else
    {
      record.getHandScores().put_detection(gt.get_filename(),RectAtDepth(det.bb),det.resp);
      record.getHandScores().put_ground_truth(gt.get_filename(),RectAtDepth(handBB));    
      return true;
    }
  }
  
  void analize_score_one_frame_fingers(
    string method_name, Mat Z, BaselineDetection det, MetaData&gt, 
    PerformanceRecord&record, Mat&total_vis)  
  {
    vector<Mat> per_finger_vis;
    // draw the fingers
    for(int iter = 1; iter <= 5; iter++)
    {
      analyze_score_one_finger(det,gt,Z, 
			       record.getFingerScores(),
			       record.getAgnosticFingerScores(),
			       record.getFingerErrorsCM(),total_vis,per_finger_vis,iter);
    }
    per_finger_vis.push_back(total_vis);    
    per_finger_vis.push_back(imageeq("",Z,false,false));
    //cout << "prec = " << record.getFingerScores().p() << endl;
    
    // save the result
    Mat message = image_text(safe_printf("total p = % title = %",
					 (double)record.getFingerScores().p(),
					 gt.get_filename()));    
    string filename = log_im(printfpp("%s_finger_dets",method_name.c_str()),
			     vertCat(tileCat(per_finger_vis),message));    
    ostringstream oss;
    IKSyntherError synther_errs;// = synther_error(gt);  
    oss << "info(max finger err): " << method_name << "\t" << gt.get_filename() << "\t" 
	<< max(record.getFingerErrorsCM()) << "\t" << mean(record.getFingerErrorsCM()) 
	<< "\t" << filename << "\t" 
	<< synther_errs.pose_error << "\t" << synther_errs.template_error << endl;
    log_locked(oss.str());

    // compute the score for the joint pose
    bool fingers_correct = 0;
    for(double finger_dist_cm : record.getFingerErrorsCM())
    {
      cout << "finger_dist_cm: " << finger_dist_cm << endl;
      double dist_thresh = params::finger_correct_distance_threshold();
      if(finger_dist_cm <= dist_thresh /*cm*/)
	fingers_correct ++;
    }
    log_file << printfpp("fingers correct = %d",fingers_correct) << endl;
    // three possibilities : FP, FN, TP
    assert(record.getFingerErrorsCM().size() == 5);
    if(det.bb == Rect())
    {
      log_file << "finger joint score: FN++" << endl;
      record.getJointPoseScores().score(Scores::Type::FN,det.resp);
    }
    else if(fingers_correct < 3)
    {
      log_file << "finger joint score: FP++" << endl;
      record.getJointPoseScores().score(Scores::Type::FP,det.resp);
    }
    else
    {
      log_file << "finger joint score: TP++" << endl;
      record.getJointPoseScores().score(Scores::Type::TP,det.resp);
    }    
  }
  
  // return a vector of distances in cm for each finger.
  PerformanceRecord analize_score_one_frame(
    string method_name, Mat Z, BaselineDetection det, MetaData&gt)
  {
    // the results
    PerformanceRecord record;
    Mat total_vis;
    
    // check that the bb is valid
    Point2d det_center = rectCenter(det.bb);
    if(det_center.x < 0 || det_center.y < 0 || det_center.x >= Z.cols || det_center.y >= Z.rows)
    {
      log_file << "warning: " << det_center << " not in " << Z.size() << endl;
      det.bb = Rect();
    }
    
    // score the hand
    if(analize_score_one_frame_hand(method_name, Z, det, gt, record, total_vis))    
    {
      // score the fingers
      analize_score_one_frame_fingers(method_name, Z, det, gt, record, total_vis);
    }
      
    return record;
  }
  
  PerformanceRecord  analyze_pxc(string person)
  {
    PerformanceRecord record;
    
    // proc the PXC Results
    string pxc_result_file = printfpp("intelCap/PXC_%s_Test_Detections.yml",person.c_str());
    FileStorage pxcStorage(pxc_result_file,FileStorage::READ);
    assert(pxcStorage.isOpened());
    for(auto line : pxcStorage.root())
    {
      // load the current result
      cout << line.name() << endl;
      PXCDetection pxcDet;
      pxcStorage[line.name()] >> pxcDet;
      boost::regex repl_pattern("Z:\\\\dropbox\\\\");
      pxcDet.filename = boost::regex_replace(pxcDet.filename,repl_pattern,string(""));
      boost::regex win_sep("\\\\");
      pxcDet.filename = boost::regex_replace(pxcDet.filename,win_sep,string("/"));
      cout << pxcDet.filename << endl;
      
      // load the corresponding MetaData
      MetaData_DepthCentric metadata(pxcDet.filename,true);
      shared_ptr<ImRGBZ> im = metadata.load_im();
      cout << "Metadata.pos.size() = " << metadata.get_positives().size() << endl;
      
      // correct BB for the cropping to intersection (see Metadata impl.)
      if(pxcDet.bb != Rect())
	pxcDet.bb -= metadata.RGBandDepthROI.tl();
      for(auto&finger_det : pxcDet.parts)
	if(finger_det.second.bb != Rect())
	  finger_det.second.bb -= metadata.RGBandDepthROI.tl();
      
      // show and score the detection
      DetectorResult det = convert(pxcDet);
      det->resp = inf;
      for(string part_name : det->part_names())
	det->getPart(part_name).resp = inf;
      record.merge(analize_score_one_frame("PXC",im->Z, *det, metadata));
    }
    pxcStorage.release();     
    
    return record;
  }
  
  PerformanceRecord analyze_pbm(string det_filename, string person)
  {
    PerformanceRecord perf_record;
    cout << "reading dets from: " << det_filename << endl;
    
    // proc the PBM results
    FileStorage pbmStorage(det_filename,FileStorage::READ);
    for(auto record : pbmStorage.root())
    {
      // load the current result
      cout << record.name() << endl;
      if(!boost::regex_match(static_cast<std::string>(record.name()),boost::regex(".*BestDet.*")))
	continue;
      BestDetection bestDet;
      pbmStorage[record.name()] >> bestDet;
      cout << bestDet.detection->src_filename << endl;
      
      // filter only certain individuals?
      boost::regex vivRE(".*" + person + ".*");
      if(!boost::regex_match(bestDet.detection->src_filename,vivRE))
	continue;
      if(boost::regex_match(bestDet.detection->src_filename,boost::regex(".*flip.*")))
	continue;
      
      // now, get the relavant metadata
      MetaData_DepthCentric metadata(bestDet.detection->src_filename,true);
      shared_ptr<ImRGBZ> im = metadata.load_im();
      Mat Z = imageeq("",im->Z,false,false);
      
      // process the frame with the info we loaded
      perf_record.merge(analize_score_one_frame("Ours",im->Z, *bestDet.detection, metadata));
    }
    pbmStorage.release(); 
    return perf_record;
  }
  
#ifdef DD_ENABLE_OPENNI    
  static bool cfged_to_skip_frame(const string&video_filename,int iter)
  {
    ostringstream oss; oss << video_filename << "/" << iter << "/";
    for(auto match : g_params.matching_keys("TEST_SKIP_FRAME.*"))
    {
      if(boost::regex_match(oss.str(),boost::regex(match.second)))
	return true;
    }
    return false;
  }

  static void do_score_video_one_frame(
    const shared_ptr<Video>&video,
    MetaData_YML_Backed&metadata,PerformanceRecord&perf_record,
    const BaseLineTest&test,const vector<BaselineDetection>&track,int iter)
  {
    // get the detection to score
    Rect_<double> handBB = metadata.get_positives()["HandBB"];    
    shared_ptr<ImRGBZ> im = metadata.load_im();
    BaselineDetection baseline_detection = track[iter];
    int times_printed = 0;
    auto print_info = [&]()
    {
      log_file << times_printed++ << test.title << " scoring frame: " << iter << " " << metadata.get_filename() << " " << baseline_detection.bb << " im size = " << im->Z.size() << endl;
    };

    // ok, score the thing
    bool is_forth = boost::regex_match(test.title,boost::regex(".*FORTH.*"));
    bool is_rc    = boost::regex_match(test.title,boost::regex(".*RC.*"));
    if(is_forth or is_rc)
      baseline_detection.scale(.5); // damn, coords are wrong for Forth3D
    print_info();    
    bool is_baseline = false; //boost::regex_match(test.title,boost::regex(".*Baseline.*"));
    bool is_ours = not is_baseline; //boost::regex_match(test.title,boost::regex(".*Ours.*"));
    assert(im->Z.size().area() > 0);
    baseline_detection.bb = clamp(im->Z,baseline_detection.bb);
    print_info();
    //if(true)
    if(is_ours and not is_baseline) //
    {      
      log_file << "Updating Baseline Detection BB: " << baseline_detection.bb << endl;
      if(g_params.option_is_set("POST_POSE_CONVEXITY"))
	baseline_detection = FingerConvexityDefectsPoser().pose(baseline_detection,*im);
      if(g_params.option_is_set("POST_POSE_VORONOI"))
	baseline_detection = VoronoiPoser().pose(baseline_detection,*im);
    }
    else
    {
      //baseline_detection = DumbPoser().pose(baseline_detection,*im);
    }
    cout << "baseline: " << baseline_detection.bb << endl;
    print_info();

    // show the result
    Mat vis_depth = imageeq("",im->Z,false,false);
    Mat vis_color = im->RGB.clone();
    if(g_params.has_key("VIS_GT"))
      rectangle(vis_depth,handBB.tl(),handBB.br(),Scalar(0,255,0));
    baseline_detection.draw(vis_depth);
    if(g_params.has_key("VIS_GT"))
      rectangle(vis_color,handBB.tl(),handBB.br(),Scalar(0,255,0));
    baseline_detection.draw(vis_color);    
    log_im_decay_freq(printfpp("%s_BaselineOutput",test.title.c_str()),
	   horizCat(horizCat(vis_depth,vis_color),im->RGB)); waitKey_safe(10);
    
    // score it!
    PerformanceRecord frameRecord = analize_score_one_frame(
      safe_printf("%_%_%",test.title.c_str(),iter,video->get_name()),im->Z, baseline_detection, metadata);
    static mutex m; unique_lock<mutex> l(m);
    perf_record.merge(std::move(frameRecord));	     
  }

  static bool reject_freespace(string video_filename,int iter,MetaData & datum, int depth = 0)
  {
    string freespace_label = datum.getAnnotation("hand in freespace?");
    log_file << datum.get_filename() << freespace_label << " key root " << datum.leveldbKeyRoot() << endl;
    bool has_yes = (freespace_label.find("y") != string::npos);
    bool has_no  = (freespace_label.find("n") != string::npos);
    for(int iter = 0; iter < depth; ++iter)
      log_file << "\t";
    if(g_params.option_is_set("FREESPACE") and (has_no))
    {
      log_file << video_filename << " skipping frame (FREESPACE): " << iter << ", " << datum.get_filename() << " with \"" << 
	freespace_label << "\"" << endl;
      return true;
    }
    else
    {
      log_file << video_filename << " accepting frame (FREESPACE): " << iter << ", " << datum.get_filename() << " with \"" 
	       << freespace_label << "\"" << endl;
      return false;
    }
  }

  void score_video_one_frame(
    const string&video_filename,const string&baseline_results,
    const shared_ptr<Video>&video,
    const vector<BaselineDetection>&track,int iter,
    PerformanceRecord&perf_record,const BaseLineTest&test
			    )
  {
    log_file << "score_video_one_frame: " << video->get_name() << " " << iter << endl;
    
    // get and display the frame
    if(cfged_to_skip_frame(video_filename,iter))
    {
      log_file << video_filename << " skipping frame: " << iter << endl;
      return;
    }
    
    // allocate a name for the metadata
    log_file << "==========" << endl;
    string frame_metadata_name = 
      printfpp("frame_metadata_name = %s.frame%d",video_filename.c_str(),iter);
    log_file << frame_metadata_name << endl;
    log_file << "==========" << endl;
      
    // get the video frame as a metadata object
    shared_ptr<MetaData_YML_Backed> root_metadata = video->getFrame(iter-1,true);
    if(!root_metadata)
    {
      log_once(safe_printf("warning, root_metadata was null % %",uuid(),iter));
      return;
    }
    if(reject_freespace(root_metadata->get_filename(),iter,*root_metadata))
      return;
    for(auto && pair : root_metadata->get_subdata_yml())
    {
      auto & metadata = pair.second;
      if(reject_freespace(video_filename,iter,*metadata,1))
	continue;
      if(g_params.option_is_set("SCORE_SKIP_LEFT") && metadata->leftP())
      {
	log_file << video_filename << " skipping frame (LEFT): " << iter << endl;
	continue;
      }
      if(g_params.option_is_set("SCORE_SKIP_RIGHT") && !metadata->leftP())
      {
	log_file << video_filename << " skipping frame (RIGHT): " << iter << endl;
	continue;
      }
      //if(!FingerConvexityDefectsPoser().is_hard_example(*metadata))
      //continue;
      
      Rect_<double> handBB = metadata->get_positives()["HandBB"];
      if(handBB == Rect_<double>())
      {
	log_once(printfpp("skipping No HandBB in %s",metadata->get_filename().c_str()));
	continue;
      }
      float z = manifoldFn_apxMin(*metadata->load_im(),handBB).back();
      float max_score_z = fromString<double>(g_params.require("SCORE_MAX_Z"));
      if(z >= max_score_z)
      {
	log_once(safe_printf("skipping frame % because % >= %",metadata->get_filename(),z,max_score_z));
	continue;
      }

      log_once(safe_printf("accepting frame %",metadata->get_filename()));
      do_score_video_one_frame(video,*metadata,perf_record,test,track,iter);
    }    
  }
#endif
  
  static vector<BaselineDetection> load_track(Video*video,const string&baseline_results)
  {       
    vector<BaselineDetection> track = loadBaseline(baseline_results,video->getNumberOfFrames());
    if(video->getNumberOfFrames() > track.size())
    {
      cout << "failed to load: " << baseline_results << endl;
      cout << printfpp("%d > %d",(int)video->getNumberOfFrames(),(int)track.size());
      assert(false);
    }
    if(g_params.option_is_set("INTERPOLATE_TRACK"))
    {
      interpolate(track);
      post_interpolate_parts(track);
    }
    if(g_params.option_is_set("INTERPOLATE_PARTS_IK"))
    {
      //if(!boost::regex_match(test.title,boost::regex(".*Ours.*")))
      interpolate_ik_regress_full_hand_pose(track);
    }

    // write the track we loaded?
    log_once("load_track baseline_results = " + baseline_results);
    std::vector<std::string> parts = regex_matches(baseline_results, boost::regex("/[^/]+/"));
    log_file << "parts = " << parts.size() << " " << parts << endl;
    string method_name = parts.back();
    string dir = params::out_dir() + "/" + method_name + "/";
    log_file << "creating dir = " << dir << endl;
    if(!boost::filesystem::exists(dir))
      assert(boost::filesystem::create_directory(dir));    
    dir += "/" + video->get_name() + "/";
    cout << "creating second dir = " << dir << endl;
    if(!boost::filesystem::exists(dir))
      assert(boost::filesystem::create_directory(dir));
    for(int frame_iter = 0; frame_iter < track.size(); ++frame_iter)
    {
      ofstream det_file(safe_printf("%/frame_%.txt",dir,frame_iter));
      // print the parts
      for(int finger_iter = 1; finger_iter <= 5; ++finger_iter)
      {
	Rect dist_phalan_bb = track.at(frame_iter).
	  parts[safe_printf("dist_phalan_%",finger_iter)].bb;
	if(dist_phalan_bb != Rect())
	{
	  Point2d uv = rectCenter(dist_phalan_bb);
	  det_file << safe_printf("finger%joint3tip",finger_iter) << "," <<
	    uv.x << "," << uv.y << "," << qnan << "," << qnan << "," << "qnan" << false << endl;
	}	  
      }      
    }
    
    return track;
  }

  static void enqueue_score_frame(
    TaskBlock&score_video,int iter, 
    BaseLineTest&test,
    shared_ptr<Video> video,string video_filename,string baseline_results,vector<BaselineDetection>&track,
    shared_ptr<map<int,PerformanceRecord>>&record_per_frame,PerformanceRecord&perf_record)
  {
    // annotate every 100th frame.
    if(video->is_frame_annotated(iter))
    {
      score_video.add_callee(
	[&,iter,record_per_frame,video_filename,baseline_results,video,track]()
	{
	  log_once(safe_printf("trying % %",video->get_name(),iter));
	  // per-video, per-frame, per-method
	  PerformanceRecord frame_record;
	  score_video_one_frame(video_filename,baseline_results,
				video,track,iter,frame_record,test);
	  static mutex m; unique_lock<mutex> l(m); // critical sec
	  perf_record.merge(frame_record);
	  record_per_frame->insert(pair<int,PerformanceRecord>(iter,frame_record));
	  // export the CM errors for handBB and fingers
	});
    }    
  }

  static void enqueue_write_video_perf(
    TaskBlock&write_video_perf,
    BaseLineTest&test,
    shared_ptr<Video> video,
    shared_ptr<map<int,PerformanceRecord>>&record_per_frame)
  {
    write_video_perf.add_callee([&test,record_per_frame,video]()
				{
				  log_once(safe_printf("writing_perf %",video->get_name()));
				  // write a file describing the errors in this method/video pair
				  string filename = printfpp((params::out_dir() + "/%s:%s.txt").c_str(),
							     test.title.c_str(),video->get_name().c_str());
				  ofstream video_method_errors(filename);
				  assert(video_method_errors.is_open());
				  for(auto&frame_record : *record_per_frame)
				  {
				    double mean_finger_error_cm = mean(frame_record.second.getFingerErrorsCM());
				    vector<double> P,R,V;
				    double hand_bb_error_cm = frame_record.second.getHandScores().compute_pr_simple
				      (P,R,V,[](RectAtDepth gt, RectAtDepth det) -> double {return gt.dist(det);});
				    video_method_errors << frame_record.first << endl;
				    video_method_errors << hand_bb_error_cm << endl;
				    video_method_errors << mean_finger_error_cm << endl;
				    video_method_errors << endl;
				  }
				  video_method_errors.close();    	
				});
  }

  static void enqueue_show_skeletonization()
  {
    
  }

  PerformanceRecord score_video(BaseLineTest test)
  {
#ifdef DD_ENABLE_OPENNI   
    PerformanceRecord perf_record;
    TaskBlock score_video("score_video");
    TaskBlock write_video_perf("write_video_perf");
    for(int video_iter = 0; video_iter < test.videos.size(); ++video_iter)
    {
      // get the paths from the user
      string video_filename = test.videos[video_iter];
      string baseline_results = test.labels[video_iter];
      shared_ptr<map<int,PerformanceRecord>> record_per_frame(new map<int,PerformanceRecord>());
      shared_ptr<Video> video = load_video(video_filename);    
      assert(video != nullptr);
  
      // load the results and the test data.
      vector<BaselineDetection> track = load_track(video.get(),baseline_results);
      
      int stride = video->annotation_stride();
      int frame_count = video->getNumberOfFrames();
      for(int iter = 0; iter < frame_count; iter += stride)
      {      
	enqueue_score_frame(score_video,iter, test,video,
			    video_filename,baseline_results,track,
			    record_per_frame,perf_record);
      }// end iteration of frames
      
      enqueue_write_video_perf(write_video_perf,test,video,record_per_frame);
    }
    score_video.execute();
    write_video_perf.execute(*empty_pool);
    
    return perf_record;
#else
    throw std::runtime_error("unsupported");
#endif
  }  
}

 
