/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "ExternalModel.hpp"
#include "Model.hpp"
#include "KITTI_Eval.hpp"
#include "MetaDataKITTI.hpp"
#include "Detection.hpp"
#include "Annotation.hpp"
#include "LibHandRenderer.hpp"
#include "LibHandMetaData.hpp"
#include "LibHandSynth.hpp"
#include "OcclusionReasoning.hpp"
#include "TestModel.hpp"
#include "BaselineDetection.hpp"
#include "Plot.hpp"
#include "ICL_MetaData.hpp"
#include "RegEx.hpp"
#include "FingerPoser.hpp"
#include "KinectPose.hpp"
#include "NYU_Hands.hpp"
#include "PXCSupport.hpp"
#include "Colors.hpp"

#include <boost/filesystem.hpp>

namespace deformable_depth
{
  // local declarations
  static string baseline_file(string stem, string imfilename);
  static int frameNumber(const string&im_filename);
}

namespace deformable_depth
{
  static vector<double> feature(double baseline_conf)
  {
    double f = ((clamp<double>(-5,baseline_conf,5))) + 5;
    // f in {0 10}
    // f*f in {0,100}
    vector<double> feat{f,f*f/10,1};
    return feat*BaselineModel_ExternalKITTI::BETA_EXTERN_CONF;
  }
  
  ///
  /// SECTION: External Model
  ///
  vector<double> BaselineModel_ExternalKITTI::minFeat() 
  {
    return feature(-inf); 
  }
  
  vector<double> BaselineModel_ExternalKITTI::getW() const
  {
    return learner->getW();
  }
  
  DetectionSet BaselineModel_ExternalKITTI::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    // parse the filename
    //cout << "Detecting on image: " << im.filename << endl;
    string KITTI_Title;
    int id;
    bool training;
    istringstream iss (im.filename);
    iss >> KITTI_Title >> id >> training;
    
    // load the baseline detections
    map<string,AnnotationBoundingBox> baseline_boxes;
    for(string dir : (training?train_dirs:test_dirs))
    {
      string det_file = dir + printfpp("%06d",id) + ".txt";
      //cout << "reading " << det_file << endl;
      map<string,AnnotationBoundingBox> some_boxes = params::defaultSelectFn()(KITTI_GT(det_file));
      baseline_boxes.insert(some_boxes.begin(),some_boxes.end());
    }
    
    DetectionSet result;
    
    for(auto & baseline_det : baseline_boxes)
    {
      vector<double> feat = feature(baseline_det.second.confidence);
      double conf = dot(getW(),feat) + learner->getB();
      
      DetectorResult det(new Detection);
      det->src_filename = im.filename;
      det->BB = baseline_det.second;
      det->resp = conf;
      det->feature = [feat](){return SparseVector(feat);};
      result.push_back(det);
    }
    
    log_once(printfpp("BaselineModel_ExternalKITTI::detect got %d from %s",
		      (int)result.size(),im.filename.c_str()));
    return result;
  }

  BaselineModel_ExternalKITTI::BaselineModel_ExternalKITTI(vector< string > train_dirs, vector< string > test_dirs) : 
    train_dirs(train_dirs), test_dirs(test_dirs)
  {
    learner.reset(new FauxLearner(vector<double>{1,0,0},0));
  }
  
  BaselineModel_ExternalKITTI::BaselineModel_ExternalKITTI()
  {
    for(string category : vector<string>{"pedestrian","cyclist","car"})
    {
      string train_dir = params::KITTI_dir() + "/object/training/det_2/lsvm4_2_" + category + "/";
      string test_dir  = params::KITTI_dir() + "/object/testing/det_2/lsvm4_" + category + "_2/";
      train_dirs.push_back(train_dir);
      test_dirs.push_back(test_dir);
    }
      
    learner.reset(new FauxLearner(vector<double>{1,0,0},0));
  }
  
  void BaselineModel_ExternalKITTI::train(vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
  }

  bool BaselineModel_ExternalKITTI::write(FileStorage& fs)
  {
    return deformable_depth::Model::write(fs);
  }

  Mat BaselineModel_ExternalKITTI::show(const string& title)
  {
    return Mat();
  }
  
  SparseVector BaselineModel_ExternalKITTI::extractPos(MetaData& metadata, AnnotationBoundingBox bb) const
  {
    DetectionSet dets = BaselineModel_ExternalKITTI::detect(*(metadata.load_im()), DetectionFilter());
    DetectorResult nn = nearest_neighbour(dets,bb);
    if(rectIntersect(nn->BB,bb) < BB_OL_THRESH)
      return vector<double>{minFeat()};
    else
      return nn->feature();
  }

  LDA& BaselineModel_ExternalKITTI::getLearner()
  {
    return *learner;
  }

  void BaselineModel_ExternalKITTI::setLearner(LDA* lda)
  {
    learner.reset(lda);
  }
  
  void BaselineModel_ExternalKITTI::update_model()
  {    
  }

  ///
  /// SECTION: IKAnnotatedModel
  ///
#ifdef DD_ENABLE_HAND_SYNTH
  class Synther
  {
  public:
    LibHandRenderer* no_arm, *armed, *segm;

  public:
    Synther()
    {
      no_arm = renderers::no_arm();
      armed  = renderers::with_arm();
      segm   = renderers::segmentation();
    }

    void render_all(bool fliplr, libhand::HandCameraSpec cam_spec,libhand::FullHandPose hand_pose)
    {
      for(auto & renderer : vector<LibHandRenderer*>{no_arm,armed,segm})
      {
	renderer->set_flip_lr(fliplr);      
	renderer->get_hand_pose() = hand_pose;      
	renderer->get_cam_spec() = cam_spec;
	renderer->render();
      }
    }

    Mat correct_metric_size(
      MetaData&hint,
      const ImRGBZ&im,
      Rect handBB) const
    {
      // default value
      Mat Z = armed->getDepth();
      //Mat Z = no_arm.getDepth();
      
      // Strategy #1: use bounding boxes
      // Mat Z = armed.getDepth();
      auto printSizeRatio = [&](string prefix) -> float
	{
	  const Camera& cam = im.camera;
	  // calc synth area
	  float synth_area = cam.worldAreaForImageArea(extrema(Z).min, handBB);
	  // calc gt area
	  Rect gt_bb = clamp(im.Z,hint.get_positives().at("HandBB"));
	  float gt_depth = manifoldFn_default(im,gt_bb)[0];
	  float gt_area = cam.worldAreaForImageArea(gt_depth, gt_bb);
	  // compare areas
	  log_file << prefix << " synth_area/gt_area(bb depth) = " << synth_area << " / " << gt_area << "(" << to_string(gt_bb) << " " << gt_depth << ")" << endl;
	  if(gt_area > 0)
	    return std::sqrt(gt_area/synth_area);
	  else
	    return 1.0;
	};
      //printSizeRatio("before");
      Z = Z * printSizeRatio("before");
      // printSizeRatio("after");
      // return Z;

      printSizeRatio("after");
      return Z;
  }


    shared_ptr<LibHandMetadata> commit(bool to_disk,MetaData&hint)
    {
      // compute the BB
      Rect handBB = compute_hand_bb(*no_arm,no_arm->getDepth(),LibHandSynthesizer::ExemplarFilters::FILTER_BAD_BB);

      // fix metric size
      shared_ptr<ImRGBZ> im = hint.load_im();
      Mat Z = correct_metric_size(hint,*im,handBB);
    
      // allocate the metadata
      static int id = 0;
      bool read_only = false;
      string directory = params::out_dir() + "/oracle_synth/";
      boost::filesystem::create_directory(directory); // make sure the directory exists
      shared_ptr<LibHandMetadata> frame_metadata = make_shared<LibHandMetadata>(
	directory + printfpp("gen%d",id++)+".yml.gz",
	armed->getRGB(),Z,no_arm->getSegmentation(),segm->getRGB(),
	handBB,no_arm->getJointPositionMap(),hint.load_im()->camera,!to_disk);
      return frame_metadata;
    }
  };

  shared_ptr<MetaData> oracle_synth(MetaData&hint) 
  {
    // mutex for libhand
    static recursive_mutex m; lock_guard<recursive_mutex> l(m);

    // get the image
    shared_ptr<ImRGBZ> im = hint.load_im();

    // load the hand pose
    if(!hasFullHandPose(hint))
      return nullptr;
    map<string,Vec3d> set_keypoints, all_keypoints;
    bool fliplr;
    libhand::HandCameraSpec cam_spec;
    libhand::FullHandPose hand_pose;
    getFullHandPose(hint,set_keypoints,all_keypoints,fliplr,cam_spec,hand_pose);    

    // TODO: remove the wrist bend?
    // bend(15) => -side(17)
    // side(15) => +twist(17)
    // twist(15) => -bend(17)
    // hand_pose.side(17) = -hand_pose.bend(15);
    // hand_pose.twist(17) = hand_pose.side(15);
    // hand_pose.bend(17) = -hand_pose.twist(15);
    // hand_pose.bend(15) = hand_pose.side(15) = hand_pose.twist(15) = 0;

    // render the hand pose 
    Synther synther;
    synther.render_all(fliplr,cam_spec,hand_pose);
    all_keypoints = synther.armed->get_jointPositionMap();
    //putFullHandPose(hint,set_keypoints,all_keypoints,fliplr,cam_spec,hand_pose);
    
    // show what's going on.
    Rect handBB = compute_hand_bb(*synther.no_arm,synther.no_arm->getDepth(),LibHandSynthesizer::ExemplarFilters::FILTER_BAD_BB);    
    if(handBB == Rect())
      return nullptr;
    Rect gt_bb = clamp(im->Z,hint.get_positives().at("HandBB"));
    Mat rend_rgb = synther.no_arm->getRGB().clone();
    Mat data_rgb = im->RGB.clone();
    cv::rectangle(rend_rgb,handBB.tl(),handBB.br(),Scalar(0,255,0));
    cv::rectangle(data_rgb,gt_bb.tl(),gt_bb.br(),Scalar(0,255,0));
    log_im("IKAnnotatedModel::detect",horizCat(rend_rgb,data_rgb));

    // create the flipped result
    auto result = synther.commit(false,hint);

    // metadata class assumes unfliped... so correct
    fliplr = false;
    synther.render_all(fliplr,cam_spec,hand_pose);
    
    // load the annotated hand model      
    auto frame_metadata = synther.commit(true,hint);
    return result;
  }

  DetectionSet IKAnnotatedModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    // concurrency bug... only allow one thread until we factor libhand into a sub processes
    assert(params::cpu_count() == 1);
    log_file << "IKAnnotatedModel::detect " << im.filename << endl;
 
    // not thead safe because it uses libhand, should be superfast anyway
    static mutex m; lock_guard<mutex> l(m);
    lock_guard<recursive_mutex> ll(exclusion_high_gui);

    // check that we have a cheat code
    DetectionSet result;
    if(filter.cheat_code.expired())
      return result;
    shared_ptr<MetaData> metadata = filter.cheat_code.lock();

    // get the bounding box
    Rect HandBB = metadata->get_positives()["HandBB"];
    
    // 
    if(HandBB != Rect())
    {
      // get the BB and setup the detection
      DetectorResult det(new Detection);
      det->BB = HandBB;
      det->resp = 1.0;

      //
      shared_ptr<MetaData> frame_metadata = oracle_synth(*metadata);
      if(!frame_metadata)
	return result;

      // 
      auto gt_poss = metadata->get_positives();
      auto ik_poss = frame_metadata->get_positives();
      Mat afT = affine_transform(ik_poss.at("HandBB"),gt_poss.at("HandBB"));
      map<string,AnnotationBoundingBox>  parts;
      for(auto && ik_part : ik_poss)
      {
	parts[ik_part.first].write(rect_Affine(ik_part.second,afT));
      }
      det->set_parts(parts);
      det->exemplar = frame_metadata;
      result.push_back(det);
    }

    return result;
  }

  void IKAnnotatedModel::train
  (vector<shared_ptr<MetaData>>&training_set,
   TrainParams train_params)
  {
  }

  Mat IKAnnotatedModel::show(const string&title)
  {
    return image_text("IKAnnotatedModel");
  }
#endif

  ///
  /// SECTION Model Builder
  ///
  ExternalModel_Builder::ExternalModel_Builder(double C, 
			  double area ,
			  shared_ptr<IHOGComputer_Factory> fc_factory,
			  double minArea,
			  double world_area_varianceiance)
  {
  }
  
  Model* ExternalModel_Builder::build(Size gtSize,Size imSize) const
  {
#ifdef DD_ENABLE_HAND_SYNTH
    //return new IKAnnotatedModel;
    return new ExternalLRF_Model;
#else
    return new BaselineModel_ExternalKITTI;
#endif
  }
  
  string ExternalModel_Builder::name() const
  {
    return "ExternalModel_Builder";
  }

  ///
  /// [1] D. Tang and T. Kim, “Latent Regression Forest : Structured Estimation of 3D Articulated Hand Posture.”
  ///
  DetectionSet ExternalLRF_Model::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet result{};
    
    // parse the filename
    vector<string> numbers = deformable_depth::regex_match(im.filename,boost::regex("\\d+"));
    assert(numbers.size() == 2);
    int vid_number = fromString<int>(numbers[0]);
    int frame_number = fromString<int>(numbers[1]);
    log_once(printfpp("ExternalLRF_Model::detect %s %d %d",im.filename.c_str(),vid_number,frame_number));    

    // load the result file
    string filename = icl_base() + "/Results/LRF_Results_seq_" + std::to_string(vid_number) + ".txt";
    log_once(printfpp("Parsing %s",filename.c_str()));
    vector<string> results = parse_lines(filename);
    string result_line = printfpp("image_%04d.png ",frame_number) + results[frame_number];
    log_once(printfpp("Result %s %s",filename.c_str(),result_line.c_str()));

    // load an ICL Metadata
    string base_path = icl_base() + "/Testing/Depth/test_seq_" + std::to_string(vid_number) + "/";
    shared_ptr<MetaData_YML_Backed> datum(load_ICL(base_path,result_line,false));  
    auto poss = datum->get_positives();
    assert(datum);

    // convert the datum to a detection
    DetectorResult det = make_shared<Detection>();
    det->BB = poss["HandBB"];
    det->set_parts(poss);
    result.push_back(det);

    return result;
  }

  void ExternalLRF_Model::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
  }

  Mat ExternalLRF_Model::show(const string&title)
  {
    return image_text(title);
  }

  ///
  /// Kinect Segmentation based detection + Kinect RF Pose Estimation
  /// 
  static int frameNumber(const string&im_filename)
  {
    std::vector<std::string> frameNum = regex_match(im_filename, boost::regex("(\\d+)"));
    if(frameNum.empty())
      return -1;    
    int frame_num = fromString<int>(frameNum.back());
    return frame_num;
  }

  DetectionSet KinectSegmentationAndPose_Model::detect_with_seg_heuristic(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet dets;

    // find and load the segmentation image for this image...
    // has form data/depth_video/greg.oni.frame500.im.gz
    int frame_num = frameNumber(im.filename);
    if(frame_num < 0)
      return dets;

    // find the segmentation directory
    vector<string> seg_dirs = find_dirs("/home/grogez/Segmentations/",boost::regex(".*"));
    map<int,string> lcs;
    for(string & seg_dir : seg_dirs)
    {
      //log_file << seg_dir << endl;
      string s1 = im.filename;
      string s2 = seg_dir;
      std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
      std::transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
      int csl = longestCommonSubstring(s1,s2);
      if(lcs.find(csl) == lcs.end() or lcs.at(csl).length() > s2.length())
	lcs[csl] = seg_dir;
    }
    string lcs_dir = lcs.rbegin()->second;
    int lcs_len = lcs.rbegin()->first;
    log_file << safe_printf("KinectSegmentationAndPose_Model::detect_with_seg_heuristic % % % %",
			    im.filename,frame_num,lcs_dir,lcs_len) << endl;

    // load and show the segmentation heuristic
    string seg_filename = safe_printf("/%/frame%.png",lcs_dir,frame_num);
    log_file << "loading: " << seg_filename << endl;
    Mat im_segh = imread(seg_filename);
    //assert(not im_segh.empty());
    log_im("im_segh",im_segh);

    // now extract the bounding box for the right hand (red)
    Rect right_handBB = bbWhere(im_segh,[&](Mat&im,int y, int x)
			   {
			     Vec3b pixel = im_segh.at<Vec3b>(y,x);			     
			     return pixel == Vec3b(0,0,255);
			   });
    DetectorResult right_hand = make_shared<Detection>();
    right_hand->resp = .5;
    right_hand->BB = right_handBB;

    dets.push_back(right_hand);
    return dets;
  }

  static string baseline_file(string stem, string imfilename)
  {
    string filename;
    if(boost::regex_match(imfilename,boost::regex(".*test_seq_1*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci//Dropbox/out/%/1.ICL.yml.gz.DD.yml",stem);
    else if(boost::regex_match(imfilename,boost::regex(".*test_seq_2.*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci//Dropbox/out/%/2.ICL.yml.gz.DD.yml",stem);
    else if(boost::regex_match(imfilename,boost::regex(".*depth_video/greg.oni.*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci//Dropbox/out/%/greg.oni.yml.gz.DD.yml",stem);
    else if(boost::regex_match(imfilename,boost::regex(".*dennis.*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci//Dropbox/out/%/dennis_test_video1.oni.yml.gz.DD.yml",stem);
    else if(boost::regex_match(imfilename,boost::regex(".*sam.*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci//Dropbox/out/%/sam.oni.yml.gz.DD.yml",stem);
    else if(boost::regex_match(imfilename,boost::regex(".*test_data11.*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci//Dropbox/out/%/test_data11.oni.yml.gz.DD.yml",stem);
    else if(boost::regex_match(imfilename,boost::regex(".*Marga.*",boost::regex::icase)))
      filename = safe_printf("/home/jsupanci/Dropbox/out/%/Marga.yml.gz.DD.yml",stem);
    else
      filename = safe_printf("/home/jsupanci/Dropbox/out/%/Greg.yml.gz.DD.yml",stem);

    log_file << "baseline_file = " << filename << endl;
    return filename;
  }

  void KinectSegmentationAndPose_Model::emplace_kinect_poses(const ImRGBZ&im, DetectionFilter filter, DetectionSet&dets) const
  {
    // grab the corresponding kinect pixel classifications
    //string stem = "2014.08.28-RF-LR-EGO2";
    string stem = "2014.08.27-RF-LR-3RD";

    string kinect_pose_file = baseline_file(stem,im.filename);
    vector<BaselineDetection> kinect_poses = loadBaseline(kinect_pose_file);    
    log_file << "KinectSegmentationAndPose_Model::emplace_kinect_poses loaded " << kinect_poses.size() << endl;

    // add them
    int frame_index = frameNumber(im.filename)-1;
    if(frame_index < kinect_poses.size())
    {
      BaselineDetection&pose_det = kinect_poses.at(frame_index);
      for(auto && det : dets)
      {
	double ol = rectIntersect(det->BB,pose_det.bb);
	if(ol > .05)
	{
	  for(auto && part : pose_det.parts)
	  {
	    auto new_part = make_shared<Detection>();
	    new_part->BB = part.second.bb;
	    new_part->resp = .5;
	    det->emplace_part(part.first,*new_part);
	  }
	}
	else
	{
	  log_file << "detection wrong: " << im.filename << det->BB << " != " << pose_det.bb << " ol = " << ol << endl;
	}
      }
    }
    else
    {
      log_once(safe_printf("warning: no kinect pose for image (%)",im.filename));
    }
  }

  DetectionSet KinectSegmentationAndPose_Model::detect_with_hough_forest(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet dets;

    shared_ptr<MetaData> cheat = filter.cheat_code.lock();
    assert(cheat);
    int frame_idx = frameNumber(im.filename);
    string stem = "2014.09.01-Hough3RD2";

    string hough_file = baseline_file(stem,im.filename);
    vector<BaselineDetection> hough_dets = loadBaseline(hough_file);
    log_file << "KinectSegmentationAndPose_Model::detect_with_hough_forest loaded track = " << hough_dets.size() << endl;

    // add them    
    if(frame_idx < hough_dets.size())
    {
      BaselineDetection&hough_det = hough_dets.at(frame_idx);   
      auto && det = make_shared<Detection>();
      det->BB = hough_det.bb;
      log_file << "KinectSegmentationAndPose_Model::detect_with_hough_forest: got " << det->BB << endl;
      det->resp = .5;
      dets.push_back(det);
    }
    else
    {
      log_once(safe_printf("warning: not enough dets (%) for index (%) in frame (%) from cheat (%)",hough_dets.size(),frame_idx,im.filename),cheat->get_filename());
    }
    
    return dets;
  }

  DetectionSet KinectSegmentationAndPose_Model::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet dets;
    
    // 
    //dets = detect_with_seg_heuristic(im,filter);
    dets = detect_with_hough_forest(im,filter);

    log_once(safe_printf("KinectSegmentationAndPose_Model % % dets",im.filename,dets.size()));

    // do pose with kinect style system
    emplace_kinect_poses(im,filter,dets);

    log_once(safe_printf("KinectSegmentationAndPose_Model % % poses",im.filename,dets.size()));

    return dets;
  }
  
  void KinectSegmentationAndPose_Model::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    // NOP
  }
  
  Mat KinectSegmentationAndPose_Model::show(const string&title)
  {
    // NO
    return Mat();
  }

  ///
  /// Yi's Deep Model
  ///
  DetectionSet DeepYiModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet result;

    // cheat for the hand bb
    shared_ptr<MetaData> cheat = filter.cheat_code.lock();
    if(cheat and g_params.has_key("CHEAT_HAND_BB"))
    {
      for(auto && sub_datum : cheat->get_subdata())
      {
	// get the cheat bb
	auto poss = sub_datum.second->get_positives();
	bool has_hand_bb = (poss.find("HandBB") != poss.end());
	if(!has_hand_bb)
	  continue;
	Rect sub_datum_bb = rectResize(poss["HandBB"],1.4,1.4);
	Point2d sub_datum_center = rectCenter(sub_datum_bb);      
	if(sub_datum_bb == Rect())
	  continue;

	// make the cheat det
	auto det = make_shared<Detection>();
	det->resp = .5;
	det->BB = sub_datum_bb;
	result.push_back(det);

	// segment
	Mat seg = DumbPoser().segment(sub_datum_bb,im);
	
	// file code will have format
	// /home/jsupanci//Dropbox/out//test_data11.oni.yml.gz.DD.yml
	string file_code = baseline_file("", im.filename);
	file_code  = boost::regex_replace(file_code,boost::regex("/home/jsupanci/+Dropbox/out//"),"");
	file_code  = boost::regex_replace(file_code,boost::regex(".yml.gz.DD.yml"),"");
	string public_name = vid_public_name(file_code);
	int frame_idx = frameNumber(im.filename)/params::video_annotation_stride() + 1;
	
	// now load the right things...
	vector<Mat> finger_pIms{};
	for(int fingerId = 1; fingerId <= 5; ++fingerId)
	{
	  int seg_index = 19 - (fingerId-1)*3;
	  string finger_file = printfpp("/home/jsupanci/data/hand-gesture-james/%s/%05d-preds.mat.%d.exr",
					public_name.c_str(),frame_idx,seg_index);
	  //log_im("pinkymap",horizCat(im.RGB,imageeq("",pinky_map,false,false)));
	  Mat finger_map = imread(finger_file,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);	  
	  if(boost::filesystem::exists(finger_file))
	    log_file << "finger_map size = " << finger_map.size() << endl;
	  else
	    log_file << "finger_map size = file not found" << endl;
	  if(finger_map.empty())
	  {
	    log_file << "warning, couldn't load finger_file: " << finger_file << endl;
	    continue;
	  }	  
	  cv::resize(finger_map,finger_map,im.Z.size(),0,0,params::DEPTH_INTER_STRATEGY);
	  assert(finger_map.type() == DataType<float>::type);
	  for(int yIter = 0; yIter < finger_map.rows; yIter++)
	    for(int xIter = 0; xIter < finger_map.cols; xIter++)
	      if(!(seg.empty() or seg.at<uint8_t>(yIter,xIter) <= 100))
		finger_map.at<float>(yIter,xIter) = 0;		
	  
	  // 
	  finger_pIms.push_back(imageeq("",finger_map,false,false));

	  // use the loaded probablisty mask with mean shift
	  finger_map.convertTo(finger_map,DataType<double>::type);
	  auto finger_det = predict_joints_mean_shift(im.Z,finger_map,im.camera);	  
	  det->emplace_part(safe_printf("dist_phalan_%",fingerId),*finger_det);
	} // end finger detection
	log_im("fingerProbIms",tileCat(finger_pIms));
	log_im("seg",seg);
      } // end hand detection
    }

    return result;
  }

  void DeepYiModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
  }

  Mat DeepYiModel::show(const string&title)
  {
    return image_text(title);
  }

  ///
  /// Jonathan Tompson's [NYU] model.
  ///
  DetectionSet NYU_Model::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    log_file << "NYU Model Detect: " << im.filename << endl;

    // get the annotations
    int frame_index = frameNumber(im.filename);   
    if(frame_index >= us.cols)
      return DetectionSet();
    KeypointFn getKeypoint = [&](int num)
    {
      double u = 2*at(us,num,frame_index);
      double v = 2*at(vs,num,frame_index);
      return Point2d(u,v);
    };
    shared_ptr<MetaData_YML_Backed> nyu_labeled = NYU_Video_make_datum(getKeypoint,im.filename,im);
    log_once(safe_printf("info(NYU_Model::detect_%) got % positive annotations",
			 im.filename,
			 nyu_labeled->get_positives().size()));

    return fromMetadata(*nyu_labeled);
  }

  void NYU_Model::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    us = read_csv_double(nyu_prefix() + "/test_predictions_u");
    vs = read_csv_double(nyu_prefix() + "/test_predictions_v");        

    log_file << "read us = " << us.size() << endl;
    log_file << "read vs = " << vs.size() << endl;
  }

  Visualization NYU_Model::visualize_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const
  {
    Mat skeleton = monochrome(background,BLUE);
    int frame_index = frameNumber(im.filename);   

    auto edge = [&](int id1, int id2)
    {
      int u1 = at(us,id1,frame_index);
      int v1 = at(vs,id1,frame_index);
      int u2 = at(us,id2,frame_index);
      int v2 = at(vs,id2,frame_index);
      cv::circle(skeleton,Point(u1,v1),5,toScalar(getColor(id1)),-1);
      cv::circle(skeleton,Point(u2,v2),5,toScalar(getColor(id2)),-1);
      cv::line(skeleton,Point(u1,v1),Point(u2,v2),toScalar(DARK_ORANGE));
    };

    edge(0,3);
    edge(3,32);
    edge(6,9);
    edge(9,32);
    edge(12,15);
    edge(15,32);
    edge(18,21);
    edge(21,32);
    edge(24,25);
    edge(25,27);
    edge(27,30);
    edge(32,30);

    return Visualization(skeleton,"NYU_Model_visualize_result");
  }

  Mat NYU_Model::show(const string&title)
  {
    return image_text("NYU Model");
  }

  ///
  /// Human (Manual) model
  /// 
  DetectionSet HumanModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    string depthHash = hashMat(im.Z);
    string cache_filename = params::cache_dir() + "/" + depthHash + ".yml";
    static mutex m; lock_guard<decltype(m)> l(m);

    DetectionSet dets;

    // get the handbb via cheating
    auto cheat = filter.cheat_code.lock();
    Rect handBB = cheat->get_positives()["HandBB"];
    if(handBB != Rect())
    {
      // try loading from the cache
      FileStorage cache_file_in(cache_filename,FileStorage::READ);
      map<string,PointAnnotation> notes;
      if(cache_file_in.isOpened())
      {
	cout << "got human's annotations from cache" << endl;
	read(cache_file_in["notes"],notes);
      }
      else
      {
	// now for the tricky part!     
	notes = labelJoints(im,handBB,false);            
      }

      // create the metadata
      shared_ptr<Metadata_Simple> annotated = std::make_shared<Metadata_Simple>(
	uuid()+".yml",true,true,false);
      annotated->setIm(im);
      Rect handBB;
      for(auto && note : notes)
      {
	Point2d click = note.second.click;
	if(click.x > 0 and click.y > 0)
	{
	  string kp_name = safe_printf("Z_%",note.first);
	  cout << safe_printf("HumanModel: setting keypoint % %",kp_name,click) << endl;
	  annotated->keypoint(kp_name,click,note.second.visibility);
	  if(handBB == Rect())
	    handBB = Rect(click,Size(1,1));
	  else
	    handBB |= Rect(click,Size(1,1));
	}
      }
      cout << safe_printf("HumanModel: setting handBB%",handBB) << endl;
      annotated->set_HandBB(rectResize(handBB,1.6,1.6));
      
      // store the result
      if(not cache_file_in.isOpened())
      {
	FileStorage cache_file(cache_filename,FileStorage::WRITE);
	cache_file << "notes" << notes;
	cache_file.release();
      }
      DetectorResult det = fromMetadata(*annotated).front();
      det->resp = .5;
      dets.push_back(det);
    }
    
    return dets;
  }

  void HumanModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    // NOP
  }

  Mat HumanModel::show(const string&title)
  {
    return image_text("human-model");
  }

  ///
  /// SECTION: NOP Model
  ///
  DetectionSet NOP_Model::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet dets;

    string prefix = safe_printf("test_%",im.filename);
    log_im(alpha_substring(prefix),horizCat(im.RGB,imageeq(im.Z)));

    return dets;
  }

  void NOP_Model::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    for(auto && datum : training_set)
    {
      shared_ptr<ImRGBZ> im = datum->load_im();
      log_im(safe_printf("train_%",alpha_substring(im->filename)),horizCat(im->RGB,imageeq(im->Z)));
    }
  }

  Mat NOP_Model::show(const string&title)
  {
    return image_text("NOP Model");
  }

  ///
  /// SECTION: Kitani's model
  ///
  static multimap<int,Rect> load_maryam_bbs(const string&filename)
  {
    multimap<int,Rect> bbs;

    ifstream ifs;
    ifs.open(filename);
    assert(ifs.is_open());
    while(ifs)
    {
      string line; std::getline(ifs,line);
      istringstream iss(line);
      double frame; iss >> frame;
      double x1; iss >> x1;
      double y1; iss >> y1;
      double x2; iss >> x2;
      double y2; iss >> y2;
      bbs.insert(pair<int,Rect>(frame,Rect(x1,y1,x2,y2)));
    }

    return bbs;
  }

  DetectionSet KitaniModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet results;
    int frame_id = frameNumber(im.filename);
    
    // load the results provided by Kitani
    multimap<int,Rect> det_bbs;
    shared_ptr<VideoDirectoryDatum> vdd;
    if(boost::regex_match(im.filename,boost::regex(".*Greg.*")))
    {
      det_bbs = load_maryam_bbs("data/MaryamTracks/Greg_963.txt");
      vdd = load_video_directory_datum("data/depth_video/egocentric/Greg/", frame_id);
    }
    else if(boost::regex_match(im.filename,boost::regex(".*Greg.*")))
    {
      det_bbs = load_maryam_bbs("data/MaryamTracks/Marga_685.txt");
      vdd = load_video_directory_datum("data/depth_video/egocentric/Marga/", frame_id);
    }
    else
    {
      log_file << safe_printf("KitaniModel::detect Unrecognized image filename: %",im.filename) << endl;
      return results;
    }    
    assert(vdd);

    // load the corresponding UV map

    /// export the detection
    auto eq_range = det_bbs.equal_range(frame_id);
    for(auto iter = eq_range.first; iter != eq_range.second; ++iter)
    {
      //
      log_file << "Kitani yield: " << iter->second << endl;      
      
      //
      auto det = make_shared<Detection>();
      det->resp = .5;
      det->BB = clamp(im.RGB,rectRGBtoZ(iter->second,vdd->UV));
      results.push_back(det);
    }

    return results;
  }

  void KitaniModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
  }

  Mat KitaniModel::show(const string&title)
  {
    return image_text(string("KitaniModel") + title);
  }
}
