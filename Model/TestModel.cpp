/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "TestModel.hpp"
#include "Video.hpp"
#include "BaselineDetection.hpp"
#include "Eval.hpp"
#include "Faces.hpp"
#include "Poselet.hpp"
#include "Log.hpp"
#include "OcclusionReasoning.hpp"
#include "ONI_Video.hpp"
#include "Semaphore.hpp"
#include "MetaDataKITTI.hpp"
#include <boost/filesystem.hpp>
#include "HL_IK.hpp"

namespace deformable_depth
{
  void test_model_images(shared_ptr<Model> model)
  {
    TaskBlock testKitti("testKitti");
    for(shared_ptr<MetaData> data : KITTI_validation_data())
    {
      MetaDataKITTI * md_kitti = dynamic_cast<MetaDataKITTI *>(&*data);
      log_file << (printfpp("KITTI: will validate %d",md_kitti->getId())) << endl;
      testKitti.add_callee([&,data,md_kitti]()
      {
	test_kitti(*data,model,true);
	log_file << (printfpp("KITTI: has validated %d",md_kitti->getId())) << endl;
      });
    }
    for(shared_ptr<MetaData> data : default_test_data())
    {
      testKitti.add_callee([&,data]()
      {
	test_kitti(*data,model,false);
      });
    }
    testKitti.execute();
  }
  
  ///
  /// SECTION: Use OpenNI to load a recorded depth/RGB video and run our model on it.
  ///
  vector< string > test_video_filenames()
  {
    map<string,string> test_vid_keys = g_params.matching_keys("TEST_VIDEO_FILE.*");
    vector<string> oni_file_names;
    
    for(auto pair : test_vid_keys)
      oni_file_names.push_back(pair.second);
    
    return oni_file_names;
  }

  void train_on_test_videos(Model&model)
  {
    // load the pointers to the test set
    vector<string> oni_file_names = test_video_filenames();
    vector<shared_ptr<MetaData> > test_data;
    for(string oni_file_name : oni_file_names)
    {
      shared_ptr<Video> pvideo = load_video(oni_file_name);
      assert(pvideo);
      Video&video = *pvideo;
      for(int iter = 0; iter < video.getNumberOfFrames(); ++iter)
      {
	if(iter % params::video_annotation_stride() == 0)
	{
	  shared_ptr<MetaData_YML_Backed> datum = video.getFrame(iter,true);
	  if(datum)
	    test_data.push_back(datum);
	}
      }
    }

    // now send them to the model. This is for daignostics and development *only*.
    model.train_on_test(test_data);
  }
  
  static void test_model_video_one_frame(
    Video&video,int iter,Model&model,
    multimap<int,BaselineDetection>&track,
    mutex&track_m)
  {
    cout << "++test_model_video_one_frame" << endl;
    
    // load the frame from the video
    shared_ptr<MetaData_YML_Backed> metadata = video.getFrame(iter,true);
    if(!metadata)
    {
      log_file << safe_printf("warning: frame % failed to load metadatum",iter) << endl;;
      return;
    }
    shared_ptr<ImRGBZ> im = metadata->load_im();
	    
    // warn if we don't have a hand in this frame.
    Rect_<double> gt_handBB = metadata->get_positives()["HandBB"];
    if(gt_handBB == Rect_<double>())
    {
      log_once(printfpp("No GT HandBB in %s",metadata->get_filename().c_str()));
    }
	    
    // run the detector
    //static Semaphore det_bw(fromString<int>(g_params.require("DET_BW")));
    //det_bw.P();
    shared_ptr<MetaData> md = metadata;
    DetectionSet best_dets = detect_default(
      model,md,im,metadata->get_filename());
    //det_bw.V();
    DetectorResult best_det; 
	      
    // log the result.
    string vid_vis_name = safe_printf("video_dets_%_%_",video.get_name(),iter);
    if(best_dets.size() > 0)
    {
      best_det = best_dets[0];	      
      test_model_show(model,*metadata,best_dets,vid_vis_name);
      if(model.is_part_model())
      {
	Mat vis_exemplar = test_model_show_exemplar(best_det);
	imwrite(params::out_dir() + "/" + 
		printfpp("%s_%d.png",video.get_name().c_str(),iter),vis_exemplar);
      }
    } 
    else
    {
      Mat rot = Quaternion(1,0,1,1).rotation_matrix();
      log_im(vid_vis_name,vertCat(image_text("No Detections"),rotate_colors(im->RGB,rot)));
    }
	    
    // nms
    DetectionSet nms_dets = nms_w_list(best_dets,.5);

    // convert and store the result
    unique_lock<mutex> l(track_m);
    for(auto && det : nms_dets)
      if(det != nullptr)
	track.insert({iter,BaselineDetection(*best_det)});
    dump_heap_profile();
  }

  void test_model_oni_video(Model&model)
  {
    if(g_params.option_is_set("TRAIN_ON_TEST"))
      train_on_test_videos(model);       
    vector<string> oni_file_names = test_video_filenames();
    
    for(int vid_iter = 0; vid_iter < oni_file_names.size(); ++vid_iter)
    {
      string oni_file_name = oni_file_names.at(vid_iter);
      shared_ptr<Video> pvideo = load_video(oni_file_name);
      assert(pvideo);
      Video&video = *pvideo;
      // now, process the video
      multimap<int,BaselineDetection> track;
      static mutex track_m; 
      TaskBlock test_model_oni_video("test_model_oni_video");
      int frame_count = video.getNumberOfFrames();      
      int stride = video.annotation_stride();
      require(stride > 0,safe_printf("Bad Stride: % %",oni_file_name,typeid(video).name()));
      int test_count = frame_count / stride;
      log_file << safe_printf("test_video = % #frames = % stride = % test_count = %",oni_file_name,frame_count,stride,test_count) << endl;
      // define the function to write the track
      auto write_track = [&]()
      {
	lock_guard<mutex> l(track_m);
	// save the result to a file.
	boost::filesystem::path oni_path(oni_file_name);
	FileStorage store(params::out_dir() + "/" + oni_path.filename().string() + ".DD.yml",FileStorage::WRITE);
	for(auto && det : track)
	  store << safe_printf("frame%_%",(int)det.first,uuid()) << det.second;
	store.release();
      };
      // Parallel (PARFOR) block over frames.
      for(int iter = 0; iter < frame_count; ++iter)
      {	
	test_model_oni_video.add_callee([&,iter]()
	{	  	  
	  // only bother running the detector on annotated frames.
	  if(iter % stride == 0)
	  {
	    log_once(safe_printf("info: testing % frame = %",oni_file_name,iter));
	    test_model_video_one_frame(video,iter,model,track,track_m);
	  }
	  else
	  {
	    log_once(safe_printf("info: skipping % frame = %",oni_file_name,iter));
	    unique_lock<mutex> l(track_m);	    
	    track.insert({iter,BaselineDetection()});
	  }
	  progressBars->set_progress("testing frame", iter, video.getNumberOfFrames());
	  write_track();
	});
      }
      test_model_oni_video.execute(); // empty_pool for serial evaluation.
      progressBars->set_progress("testing frame", 1, 1);      
      write_track();
    }
  }  
  
  template<typename T>
  static T reduce_thresh(T thresh)
  {
    if(thresh < 0)
      return 2*thresh;
    else if(thresh == 0)
      return -1;
    else
    {
      T newThresh = thresh/2;
      if(newThresh < thresh)
	return newThresh;
      else return 0;
    }
  }
  
  DetectionSet efficent_detection(
    Model&model,MetaData&metadata,shared_ptr<ImRGBZ> imRGBZ,
    string filename, bool allow_reject)
  {  
    // filter out face detections...
    FaceDetector faceDetector;
    
    const int DET_QUANT = numeric_limits<int>::max();
    float det_thresh = -inf;
    float old_thresh = det_thresh;
    DetectionSet dets;
    bool repeat = true;
    do
    {
      // process detections
      printf("Testing %s w/ thresh = %f\n",filename.c_str(),det_thresh);
      DetectionFilter filter(det_thresh,DET_QUANT);
      filter.supress_feature = g_supress_feature;
      filter.verbose_log = true;
      dets = model.detect(*imRGBZ,filter);
      dets = faceDetector.filter(dets,*imRGBZ,.35);
      dets = metadata.filter(dets);
      dets = nms_w_list(dets,0);
      
      // update thresholds
      old_thresh = det_thresh;
      det_thresh = reduce_thresh(det_thresh);
      if(!allow_reject)
	det_thresh = -inf;
      
      // 
      repeat = 
	  !allow_reject && 
	  dets.size() <= 0 
	  && det_thresh != old_thresh;
    } while(repeat);
	
    return dets;
  }
  
  DetectionSet detect_all(
    Model&model,shared_ptr<MetaData>&metadata,shared_ptr<ImRGBZ> imRGBZ,
    string filename)
  {
    // note the start time
    log_once(printfpp("++detect_all: %s @ %s",metadata->get_filename().c_str(),current_time_string().c_str()));
    auto started_at = std::chrono::system_clock::now();

    // run the detector
    const int int_max = numeric_limits<int>::max();
    DetectionFilter filter(-inf,int_max);
    filter.testing_mode = true;
    filter.supress_feature = g_supress_feature; // used for visualization
    filter.verbose_log = true;
    filter.gt_bbs_hint = params::defaultSelectFn()(metadata->get_positives());
    filter.cheat_code = metadata;
    DetectionSet dets = model.detect(*imRGBZ,filter);
    auto part_debug_info = dets.part_candidates;
    //dets = FaceDetector().filter(dets,*imRGBZ,.35);
    dets = metadata->filter(dets);
    //dets = nms_w_list(dets,0);    
    dets.part_candidates = part_debug_info;
    
    // note the elapsed time and stop time
    auto stopped_at = std::chrono::system_clock::now();
    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopped_at - started_at);
    log_file << "detect_all" <<  metadata->get_filename() << " took " << duration.count() << " milliseconds" << endl;
    cout << "detect_all" <<  metadata->get_filename() << " took " << duration.count() << " milliseconds" << endl;
    log_once(printfpp("--detect_all: %s @ %s",metadata->get_filename().c_str(),current_time_string().c_str()));

    // regardless of score, check that the positive BB is in the detection set
    logMissingPositives(*metadata,dets);
    
    return dets;
  }
  
  DetectionSet detect_default(
    Model&model,shared_ptr<MetaData>&metadata,shared_ptr<ImRGBZ> imRGBZ,
    string filename)
  {
    return detect_all(model,metadata,imRGBZ,filename);
  }
  
  Mat test_model_show_text(
    DetectorResult best_det,
    DetectionSet&dets_by_dist_from_gt,
    shared_ptr<ImRGBZ> im)
  {
    Mat vis_text;
    if(best_det != nullptr)
    {
      double z = best_det->getDepth(*im);
      string line_one = printfpp("Pose = %s Depth = %f(%f) WorldArea = %f",
	  best_det->pose.c_str(),
	  z, best_det->depth,
	  im->camera.worldAreaForImageArea(z,best_det->BB));
      string lr_str = best_det->lr_flips%2?"right":"left";
      string line_two = printfpp("\n%s",lr_str.c_str());
      string line_three = printfpp("Resp = %s",best_det->print_resp_computation().c_str());
      string line_four = printfpp("ClosestRsp = %s",dets_by_dist_from_gt[0]->print_resp_computation().c_str());
      vis_text = vertCat(
	  image_text(line_one),vertCat(
	    image_text(line_two),vertCat(
	      image_text(line_three),image_text(line_four))));
    }
    else
      vis_text = image_text("No Detection was found...");    
    return vis_text;
  }
  
  Mat test_model_show_exemplar(DetectorResult best_det)
  {
    Mat exemplarRGB, exemplarZ;
    if(best_det)
    {
      // get the image from the exmplar
      shared_ptr<ImRGBZ> exemplar_im = best_det->exemplar->load_im();
      exemplarRGB = exemplar_im->RGB.clone();
      exemplarZ = exemplar_im->Z.clone();
      
      // extract the handBB and depth
      Rect_<double> handBB = best_det->exemplar->get_positives()["HandBB"];
      double handZ = extrema(exemplarZ(handBB)).min;
      
      // set ROIs
      exemplarRGB = exemplarRGB(handBB);
      exemplarZ   = exemplarZ(handBB);
      
      // turn all pixels behind the hand to blue
      for(int yIter = 0; yIter < exemplarZ.rows; yIter++)
	for(int xIter = 0; xIter < exemplarZ.cols; xIter++)
	  if(exemplarZ.at<float>(yIter,xIter) > handZ + best_det->z_size)
	    exemplarRGB.at<Vec3b>(yIter,xIter) = Vec3b(255,0,0);
	  
      // flip the extracted exemplar
      if(best_det->lr_flips % 2)
      {
	cv::flip(exemplarRGB,exemplarRGB,1);
      }

      // add the filename
      exemplarRGB = vertCat(image_text(exemplar_im->filename),imVGA(exemplarRGB));
    }
    else
      exemplarRGB = image_text("NO DETECTION");//Mat(params::vRes,params::hRes,DataType<Vec3b>::type,Scalar(255,0,255));

    return exemplarRGB;
  }

  void test_model_show_closests_exemplar(
    // inputs
    const Model&model,MetaData&metadata,DetectionSet&dets,
    // outputs
    DetectionSet&dets_by_dist_from_gt,
    Mat&vis_closest_exemplar)
  {
    // load the im
    shared_ptr<ImRGBZ> im = metadata.load_im();        

    // visualize the closest exemplar
    if(dets.size() > 0)
    {
      shared_ptr<MetaData> closestExemplar = closest_exemplar_ik(metadata);
      if(closestExemplar)
	vis_closest_exemplar = vertCat(image_text("Closest Exemplar"),
				       closestExemplar->load_im()->RGB);
    }    
  }

  void test_model_show_closests_detection(
    // inputs
    const Model&model,MetaData&metadata,DetectionSet&dets,
    // outputs
    DetectionSet&dets_by_dist_from_gt,
    Mat&vis_closest_detection)
  {
    // load the im
    shared_ptr<ImRGBZ> im = metadata.load_im();            
    
    // visulize the closest hand detection
    Rect bb_gt = metadata.get_positives()["HandBB"];
    dets_by_dist_from_gt = dets;
    std::sort(dets_by_dist_from_gt.begin(),dets_by_dist_from_gt.end(),[&](
      const DetectorResult &v1,const DetectorResult&v2)
    {
      float ol1 = rectIntersect(bb_gt,v1->BB);
      float ol2 = rectIntersect(bb_gt,v2->BB);
      if(ol1 == ol2)
	return v1->resp > v2->resp;
      return ol1 > ol2; // sort from large to small
    });
    // visualize its closest parts
    DetectorResult closesetDet(new Detection());
    DetectionSet modified_closest_detection_vector;
    if(dets_by_dist_from_gt.size() > 0)
    {
      (*closesetDet) = *dets_by_dist_from_gt[0];
      modified_closest_detection_vector.push_back(closesetDet);
      for(string part_name : closesetDet->part_names())
      {
	Rect_<double> part_bb = metadata.get_positives()[part_name];
	auto&closeset_part = closesetDet->getPartCloseset(part_name, part_bb);
	closesetDet->emplace_part(part_name,closeset_part,false);
      }
      dets_by_dist_from_gt = modified_closest_detection_vector;
    }
    // generate the image
    vis_closest_detection = model.Model::vis_result(*im,im->RGB,modified_closest_detection_vector);
    // draw the GT
    rectangle(vis_closest_detection,bb_gt.tl(),bb_gt.br(),Scalar(255,255,0));
    vis_closest_detection = vertCat(image_text("closest"),vis_closest_detection);
  }
  
  void test_model_show_closests_fingers(
    // inputs
    const Model&model,MetaData&metadata,DetectionSet&dets,
    // outputs
    DetectionSet&dets_by_dist_from_gt,
    Mat&vis_closest_fingers)
  {
    // only run with big memory enabled
    if(!g_params.has_key("DEBUG_PAIRWISE_BIG_MEMORY"))
    {
      //dets_by_dist_from_gt.clear();
      vis_closest_fingers = image_text("NO DEBUG_PAIRWISE_BIG_MEMORY");
      return;
    }
    
    // load the im
    shared_ptr<ImRGBZ> im = metadata.load_im(); 
    
    // for the detected pose, show the parts closest to the ground truth.
    Mat root_reject_reason = image_text("NULL");
    vis_closest_fingers = im->RGB.clone(); 
    if(dets.size() > 0)
    {       
      string target_pose = dets_by_dist_from_gt[0]->pose;
      
      set<string> part_names = dets[0]->part_names();
      part_names.insert("HandBB");
      for(string part_name : part_names)
      {
	Rect_<double> target_box = metadata.get_positives()[part_name];
	if(target_box == Rect_<double>())
	  continue; // no annotation for this part
	
	// find the closest finger BB and draw it.
	DetectorResult closest_finger;
	double min_dist = inf;
	for(auto && candidate : dets.part_candidates[part_name])
	{
	  if( candidate->pose != target_pose || 
	      candidate->lr_flips != dets_by_dist_from_gt[0]->lr_flips)
	    continue;
	  
	  Vec2d tl_offset = (candidate->BB.tl() - target_box.tl());
	  Vec2d br_offset = (candidate->BB.br() - target_box.br());
	  double dist = ::sqrt(tl_offset.dot(tl_offset)) + ::sqrt(br_offset.dot(br_offset));
	  
	  if(dist < min_dist)
	  {
	    min_dist = dist;
	    closest_finger = candidate;
	  }
	}
	
	// show the closest finger
	if(closest_finger != nullptr)
	{
	  rectangle(vis_closest_fingers,closest_finger->BB.tl(),closest_finger->BB.br(),Scalar(0,0,255));
	  
	  if(part_name == "HandBB")
	  {
	    assert(closest_finger->pw_debug_stats != nullptr);
	    root_reject_reason = image_text(closest_finger->pw_debug_stats->toLines());
	  }
	}
      }
    }
    
    vis_closest_fingers = vertCat(image_text("ClosestFingers"),
				  horizCat(vis_closest_fingers,root_reject_reason));
  }  
  
  void test_model_show_closests(
    // inputs
    const Model&model,MetaData&metadata,DetectionSet&dets,
    // outputs
    DetectionSet&dets_by_dist_from_gt,
    Mat&vis_closest_detection,Mat&vis_closest_exemplar, Mat&vis_closest_fingers)
  {    
    test_model_show_closests_exemplar(model,metadata,dets,dets_by_dist_from_gt,vis_closest_exemplar);
    test_model_show_closests_detection(model,metadata,dets,dets_by_dist_from_gt,vis_closest_detection);  
    test_model_show_closests_fingers(model,metadata,dets,dets_by_dist_from_gt,vis_closest_fingers);
  }
  
  // show the result of testing
  TestModelShowResult test_model_show(const Model&model,MetaData&metadata,DetectionSet&dets,string prefix)
  {
    // load the im
    shared_ptr<ImRGBZ> im = metadata.load_im();
    
    // print the pose 
    DetectorResult best_det;
    if(dets.size() > 0)
      best_det = dets[0];
    cout << "estimated pose: " << ((dets.size()>0)?dets[0]->pose:"Unknown") << endl;
    
    // (1) show the general result result
    Visualization general_viz = model.Model::visualize_result(*im,im->RGB,dets);
    log_file << "test_model_show: general image generated" << endl;
    
    // (2) show the features (model params corresponding to this result.
    Mat depth_vis = imageeq("",im->Z,false,false);
    Visualization param_viz = model.visualize_result(*im,depth_vis,dets);
    log_file << "test_model_show: param image generated" << endl;       
 
    // (4) visualize the detection which most closely matches the ground truth
    DetectionSet dets_by_dist_from_gt;
    Mat vis_closest_detection, vis_closest_exemplar, vis_closest_fingers;
    if(model.is_part_model())
      test_model_show_closests(model,metadata,dets,
			      dets_by_dist_from_gt,vis_closest_detection,
			      vis_closest_exemplar,vis_closest_fingers);
    
    // (5) vis the face detections
    static FaceDetector face_detector;
    Mat face_vis = vertCat(image_text("FaceDetection"),face_detector.detect_and_show(*im));
    
    // (3) draw some debug information
    Mat vis_text;
    if(model.is_part_model())
      vis_text = test_model_show_text(best_det,dets_by_dist_from_gt,im); 
    
    // (6) show the exemplar
    Mat exemplarRGB;
    if(model.is_part_model())
      exemplarRGB = vertCat(image_text("chosen exemplar"),test_model_show_exemplar(best_det));
    
    // cirtical section, make these appear sequentally on disk when sorted by date
    static mutex m;
    {
      unique_lock<mutex> L(m);
      log_file << "test_model_show: aquired mutex" << endl;
      //image_safe(prefix,general_image);  
      //log_im(prefix,general_image);
      //image_safe(prefix+"params",param_image);  
      //log_im(prefix+"params",param_image);    
      Mat top = tileCat(vector<Mat>{
	  param_viz.image(),general_viz.image(),vis_closest_exemplar,
	vis_closest_detection,face_vis,exemplarRGB,vis_closest_fingers});
      Mat bottom = vis_text;
      log_im(prefix,vertCat(top,bottom));
      Visualization(general_viz,"general_viz",param_viz,"param_viz").write(prefix);
      //waitKey_safe(10);    
    }
    log_file << "-test_model_show" << endl;
    
    if(dets_by_dist_from_gt.size() > 0)
      return TestModelShowResult{dets_by_dist_from_gt[0]};
    else
      return TestModelShowResult{nullptr};
  }
  
  static shared_ptr<MetaData> test_model_one_example_check_metadata(
    shared_ptr<MetaData>&metadata,Model&model)
  {
    // load test example
    if(metadata == nullptr)
    {
      log_file << "bad test (BB labeled): " << metadata->get_filename() << endl;
      return nullptr;
    }
    Rect BB = metadata->get_positives()["HandBB"];
    if(BB.area() < model.min_area())
    {
      log_file << "bad test: (BB Size): " << metadata->get_filename() << endl;
      return nullptr;
    }
    assert(metadata->get_positives().size() >= 1);
    // check depth
    shared_ptr<ImRGBZ > imRGBZ= metadata->load_im();
    if(not (0 <= BB.x && 0 <= BB.width && BB.x + BB.width <= imRGBZ->Z.cols 
      && 0 <= BB.y && 0 <= BB.height && BB.y + BB.height <= imRGBZ->Z.rows))
    {
      log_file << "bad test (BB location): " << metadata->get_filename() << endl;
      return nullptr;
    }
    Mat_<float> Zbb = imRGBZ->Z(BB);
    vector<float> zs;
    for(int rIter = 0; rIter < Zbb.rows; rIter++)
      for(int cIter = 0; cIter < Zbb.cols; cIter++)
	zs.push_back(Zbb.at<float>(rIter,cIter));
    std::sort(zs.begin(),zs.end());
    if(order(zs,.25) == params::MAX_Z())
    {
      log_file << "bad test (BB Depth): " << metadata->get_filename() << endl;
      return nullptr;
    }
    
    // only consider where Intel's PXC fired
    //if(!PXCFile_PXCFired(test_dir + test_stems[iter] + ".labels.yml"))
      //continue;    
    
    return metadata;
  }
  
  // test_model_one_example(test_data[iter],model,scores,test_dir,scoresByPose
  void test_model_one_example
    (shared_ptr<MetaData> metadata,
     Model&model,
     Scores&scores,
     map<string,Scores>&scoresByPose,
     DetectionOutFn write_best_det
    )
  {
    metadata = test_model_one_example_check_metadata(metadata,model);
    if(metadata == nullptr)
      return;
    shared_ptr<ImRGBZ > imRGBZ= metadata->load_im();
    Rect BB = metadata->get_positives()["HandBB"];
  
    // do detection
    DetectionSet dets = 
      detect_default(model,metadata,imRGBZ,metadata->get_filename());
    if(dets.size() == 0)
    {
      shared_ptr<Detection> no_detection(new Detection);
      no_detection->resp = -inf;
      dets.push_back(no_detection);
    }
    if(model.is_part_model())
      assert(dets.part_candidates.size() > 0);
	    
    // score
    Rect_<double> detBB = (dets.size() > 0)?dets[0]->BB:Rect_<double>();
    double detResp = (dets.size() > 0)?dets[0]->resp:-inf;
    printf("detBB = (%d %d) to (%d %d)\n",
	  (int)detBB.tl().x,(int)detBB.tl().y,(int)detBB.br().x,(int)detBB.br().y);
    bool correct = false;
    #pragma omp critical
    {
      // score and log failure
      static int outNum = 0;
      outNum++;
      correct = rectScore(BB, detBB,detResp,scores);
      
      // also, record scores for each pose.
      Scores&scorePerPose = scoresByPose[metadata->get_pose_name()];
      bool pose_correct = rectScore(BB, detBB, detResp, scorePerPose);
    }
    
    // display the detections
    TestModelShowResult tms_result = test_model_show(model,*metadata,dets);

    // write the detection
    if(dets.size() > 0)
    {
      assert(dets[0] != nullptr);
      write_best_det(metadata->get_filename(),dets[0],tms_result.closest,correct);
    }
    else
    {
      DetectorResult no_det;
      write_best_det(metadata->get_filename(),no_det,no_det,false);
    }    
  }
  
  void test_model(Model&model,string test_dir, 
		  Scores&scores, 
		  map<string,Scores>&scoresByPose,
		  DetectionOutFn write_best_det)
  {
    log_file << "+test_model on " << test_dir << endl;
    vector<shared_ptr<MetaData> > test_data;
    if(test_dir == "**TRAIN**")
    {
      test_data = metadata_build_all(default_train_dirs(),true); 
      metadata_insert_lr_flips(test_data);
      test_data = random_sample(test_data,25);
    }
    else
      test_data = metadata_build_all(test_dir, false,true);
    
    TaskBlock test_tasks("test_model"); 
    for(int iter = 0; iter < test_data.size(); iter++)
    {
      test_tasks.add_callee([&,iter]()
      {
	test_model_one_example(test_data[iter],model,scores,scoresByPose,write_best_det);
	printf("%s p = %f r = %f\n",test_dir.c_str(),(float)scores.p(),(float)scores.r());
      });
    }
    test_tasks.execute(*default_pool);
    cout << "test_model: tasks complete" << endl;
    log_file << "-test_model on " << test_dir << endl;
  }  
  
  void test_kitti(MetaData&data,shared_ptr<Model>&model,bool validation)
  {
    // make sure we have a validation output directory
    boost::filesystem::path validation_dir(params::out_dir()+"/valid/");
    boost::filesystem::create_directory(validation_dir);
    
    // execute the detector
    shared_ptr<ImRGBZ> im = data.load_im();
    log_once(printfpp("testing: %s",data.get_filename().c_str()));
    DetectionFilter filter(-inf,numeric_limits<int>::max());
    filter.supress_feature = g_supress_feature;
    filter.verbose_log = false;
    DetectionSet dets = model->detect(*im,filter);
    dets = nms_w_list(dets,.5);
    dets = data.filter(dets);
    
    // draw the detections
    string type_label = validation?"validation":"test";
    log_im(printfpp("Detections%s",type_label.c_str()),drawDets(data,dets));
    
    // store the detections into a file
    MetaDataKITTI * md_kitti = dynamic_cast<MetaDataKITTI *>(&data);
    ofstream ofs;
    if(validation)
      ofs.open(params::out_dir() + "/valid/" + printfpp("%06d.txt",md_kitti->getId()));
    else
      ofs.open(params::out_dir() + printfpp("%06d.txt",md_kitti->getId()));
    for(DetectorResult det : dets)
      ofs << params::target_category() + " 0 0 0 " << det->BB.tl().x << " " << det->BB.tl().y << " " 
          << det->BB.br().x << " " << det->BB.br().y << " 0 0 0 0 0 0 0 " << det->resp << endl;
  }
}
