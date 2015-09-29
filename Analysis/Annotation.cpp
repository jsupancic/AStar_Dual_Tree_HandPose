/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "util.hpp"
#include "Annotation.hpp"
#include "ONI_Video.hpp"
#include "LibHandRenderer.hpp"
#include "LibHandSynth.hpp"
#include "InverseKinematics.hpp"
#include "Pointers.hpp"

#include <streambuf>

namespace deformable_depth 
{
  using namespace std;
  
  /// SECTION: Label      
  void labelPose(Mat&RGB,string label_file_name)
  {
    shared_ptr<MetaData_YML_Backed> metadata(new MetaData_DepthCentric(label_file_name,false));
    if(metadata->get_pose_name() == "Unknown")
    {
      printf("Pose for Image? ");
      string pose_name; cin >> pose_name;
      metadata->setPose_name(pose_name);
      printf("\n");
    }
  }
  
  void labelLeftRight(Mat&RGB,MetaData_YML_Backed&metadata)
  {
#ifdef CXX11
    launch_background_repainter();
#endif
    image_safe("Label LeftRight",RGB); waitKey_safe(10);
    metadata.set_is_left_hand(prompt_yes_no("Is it a left hand (y/n)? "));
  }
  
  static void putAnnotation(Mat&RGB, string pointName, bool visible, Point2i p, int minColor)
  { 
    Scalar color = visible?Scalar(255,minColor,minColor):Scalar(minColor,minColor,255);
    putText(RGB,pointName,p,FONT_HERSHEY_PLAIN,1,color);    
    cv::circle(RGB,p,3,color,-1);    
  }
  
  static PointAnnotation label_point(Mat RGB,string pointName,string winName)
  {
    bool visible;
    char code;
    Point2d p = getPt(winName,&visible,&code);

    // draw the point.
    putAnnotation(RGB,pointName,visible,Point2i(p.x,p.y),0);
    printf("putText @ (%d %d)\n",(int)p.x,(int)p.y);
    
    return PointAnnotation{p,visible,RGB,code};
  }
    
  void label_handBB(MetaData_YML_Backed&metadata)
  {
    metadata.set_HandBB(getRect(
      "RGB - Give Hand BB here",
      metadata.load_im()->Z,
      metadata.get_positives()["HandBB"],
      true));
  }
  
  void labelRGBSubArea(ImRGBZ&im)
  {
    Rect rgb_bb = getRect("Label RGB Area",im.RGB,Rect());
    cout << "User Labeled (tl), (br) = (" 
	 << rgb_bb.tl().x << "," << rgb_bb.tl().y << ") "
	 << rgb_bb.br().x << "," << rgb_bb.br().y << ") "
	 << endl;
  }
  
  PointAnnotation labelJoint(string partName,Mat&imDpy,double sf,Rect handBB,int&iter)
  {
    // get label from user
    auto r = label_point(imDpy,partName,"Label");
    r.click = Point2d(r.click.x/sf,r.click.y/sf) 
      + Point2d(handBB.tl().x,handBB.tl().y);
    image_safe("Label",r.RGB,false);
    printf("Get part: %s\n",partName.c_str());
    
    // store label
    if(r.code == 'x')
      return r;
    else if(r.code == 'r')
      return r;
    else if(r.code == 'p')
      iter-=2;
    else if(r.code == 'n')
      iter++;
    else if(r.code == '\0')
      return r;
    else
    {
      // repeat
      printf("unrecognized command: %c\n",r.code);
      iter--;
    }    
    
    r.code = 'g';
    return r;
  }
  
  Mat showAnnotations(Mat im, MetaData& metadata,double sf,Rect cropBB)
  {
    auto transform_pt = [sf,&cropBB](Point2d pt)->Point2d
    {
      return sf*(pt-Point2d(cropBB.tl().x,cropBB.tl().y));
    };
    
    // show keypoints
	vector<string> kp_names = metadata.keypoint_names();
    for(int idx = 0; idx < kp_names.size(); idx++)
    {
		string partName = kp_names[idx];
      auto r = metadata.keypoint(partName);
      Point2d pt = take2(get<0>(r)); pt = transform_pt(pt);
      bool vis = get<1>(r);
      // draw
      putAnnotation(im,partName,vis,Point2i(pt.x,pt.y),127);
    }
    
    // positive BBs
	map<string,AnnotationBoundingBox > poss = metadata.get_positives();
	for(map<string,AnnotationBoundingBox >::iterator iter = poss.begin(); iter != poss.end(); ++iter)
    {
		pair<string,AnnotationBoundingBox > positive = *iter;
      // round to int?
		positive.second.write(Rect_<double>(positive.second.tl(),positive.second.br()));
      cv::rectangle(im,transform_pt(positive.second.tl()),
		    transform_pt(positive.second.br()),Scalar(255,0,0));
    }
    
    return im;
  }  
  
  vector< string > essential_keypoints()
  {
    vector<string> partNames = 
      { "Z_J11", "Z_J12", "Z_J13","Z_J14",
	"Z_J21", "Z_J22", "Z_J23","Z_J24",
	"Z_J31", "Z_J32", "Z_J33","Z_J34",
	"Z_J41", "Z_J42", "Z_J43","Z_J44",
	"Z_J51", "Z_J52", "Z_J53","Z_J54"
      };
    return partNames;
  }
  
  static vector<string> allParts = 
  { "J11", "J12", "J13","J14","J15",
    "J21", "J22", "J23","J24","J25",
    "J31", "J32", "J33","J34","J35",
    "J41", "J42", "J43","J44","J45",
    "J51", "J52", "J53","J54",
    "P0"};
  static vector<string> fingerTips = 
  {
    "J11","J12",
    "J21","J22",
    "J31","J32",
    "J41","J42",
    "J51","J52"
  };
  static vector<string>&partNames = fingerTips;

  map<string,PointAnnotation> labelJoints(const ImRGBZ&src_im, Rect handBB,bool skip_labeled)
  { 
    map<string,PointAnnotation> annotations;    
                
    // zoom in in the correct region
    ImRGBZ imRGBZ = (src_im)(handBB);
    // find scale factor
    double sf = std::sqrt(640.f*480.f/imRGBZ.RGB.size().area());
    imRGBZ = (imRGBZ).resize(sf);
    Mat imRGB = imRGBZ.RGB;
    Mat imZ = imRGBZ.Z;
    imZ = imageeq("",imZ,false,false);
    Mat imDpy = imZ; // imZ/imRGB

    // draw the initial values    
    image_safe("RGB",imRGB,false);
    // init dpy    
    image_safe("Label",imDpy,false); waitKey_safe(10);    
    
    // label parts    
    for(int iter = 0; iter < partNames.size(); iter++)
    {
      string part_name = partNames[iter];
      PointAnnotation note = labelJoint(part_name,imDpy,sf,handBB,iter);      
      if(note.code == 'x')
	return annotations;
      annotations[part_name] = note;
    }
    return annotations;
  }

  void labelJoints(MetaData_YML_Backed&metadata, bool skip_labeled = false)
  {
    cout << "labelJoints: " << metadata.get_filename() << endl; 
    // if we already have enough labels, skip.
    bool all_labeled = true;
    for(string partName : partNames)
    {
      partName = "Z_" + partName;
      if(!metadata.hasKeypoint(partName) || metadata.keypoint(partName).first == Point3d())
      {
	cout << "needs: " << partName << endl;
	all_labeled = false;
      }
    }
    if(all_labeled && skip_labeled)
      return;    
        
    // extract info from metadata
    Rect handBB = metadata.get_positives()["HandBB"];	
    shared_ptr<ImRGBZ> imRGBZ = metadata.load_im();

    // otherwise get the labels
    map<string,PointAnnotation> annotations = labelJoints(*imRGBZ, handBB,true);    
    for(auto & note : annotations)
      metadata.keypoint(note.first,note.second.click,note.second.visibility);
  }
  
  string winName = "Full Pose Annotation";
  
#ifdef DD_ENABLE_HAND_SYNTH

  struct FullHandPoseAnnotationState
  {
    Mat RGB;
    Mat background;
    Mat vis;
    Mat legend;
    LibHandRenderer*renderer;
    map<string,int> name2id;
    map<string,Vec3d> keypoints;
    string selected_keypoint;
    map<string,Vec3d> set_keypoints;
    double tx, ty, s;
    bool changed;
    
    void draw_on(Mat&on)
    {
      Mat edges = renderer->getSegmentation().clone();
      cv::dilate(edges,edges,Mat());
      edges = edges - renderer->getSegmentation();
      cout << "scale = " << s << endl;
      Mat rz_edges; cv::resize(edges,rz_edges,Size(),s,s,params::DEPTH_INTER_STRATEGY);
      //cout << edges << endl;
      
      // draw the initial hand pose
      for(int rIter = 0; rIter < on.rows; rIter++)
	for(int cIter = 0; cIter < on.cols; cIter++)
	{
	  int rT = rIter + s*ty;
	  int cT = cIter + s*tx;
	  if(rT >= rz_edges.rows || rT < 0 || cT >= rz_edges.cols || cT  < 0)
	    continue;
	    
	  Vec3b& pixel = on.at<Vec3b>(rIter,cIter);
	  if(rz_edges.at<uchar>(rT,cT))
	  {
	    for(int c = 0; c < 3; c++)
	      pixel[c] = 255 - pixel[c];
	  }
	}
      // draw keypoints
      int iter = 0;
      for(auto keypoint : keypoints)
      {
	auto pt = keypoint.second;
	auto orig_pt = pt;
	if(set_keypoints.find(keypoint.first) != set_keypoints.end())
	  pt = set_keypoints.at(keypoint.first);
	Scalar color = getColorScalar(iter++);
	if(pt != orig_pt)
	{
	  cv::line(on,Point2i(pt[0],pt[1]),Point2i(orig_pt[0],orig_pt[1]),color);
	  cv::circle(on,Point2i(pt[0],pt[1]),5,color,-1);
	}
	else
	  cv::circle(on,Point2i(pt[0],pt[1]),5,color);
      }      
    }
    
    void draw()
    {
      vis = background.clone();
      draw_on(vis);
      if(legend.empty())
      {
	legend = Mat(vis.rows,vis.cols,DataType<Vec3b>::type,Scalar::all(255));
	draw_on(legend);
      }
      vector<Mat> tiles = {vis,legend,renderer->getRGB(),RGB};
      image_safe(winName,tileCat(tiles),false);      
    }
    
    void read_keypoints()
    {
      for(auto & kp : renderer->get_jointPositionMap())
      {
	keypoints[kp.first] = s*kp.second  - Vec3d(s*tx,s*ty,0);
// 	if(set_keypoints.find(kp.first) != set_keypoints.end())
// 	{
// 	  set_keypoints[kp.first] = s*kp.second  + Vec3d(s*tx,s*ty,0);
// 	}
      }      
    }
    
    void run_ik()
    {
      if(set_keypoints.size() < 3)
      {
	cout << "Need at least 3 set keypoints to run IK" << endl;
	return;
      }
      
      cout << "IK START" << endl;
      log_file << "running inverse kinnmatics" << endl;
      map<string,Vec3d> fixed_keypoints;
      for(auto keypoint : set_keypoints)
      {
	log_file << "fixed_kp : " << keypoint.first << endl;
	fixed_keypoints[keypoint.first] = keypoint.second;
      }
      auto dist = incremental2DIk(*renderer,fixed_keypoints);
      dist.s = 1 / dist.s;
      renderer->render();
      tx = dist.tx;
      ty = dist.ty;
      s = dist.s;
      read_keypoints();
      log_file << printfpp("tx = %f ty = %f s = %f",tx,ty,s) << endl;
      changed = true;
      cout << "IK DONE" << endl;
    }
    
    void flip()
    {
      bool new_flr_state = !renderer->get_flip_lr();
      cout << "new_flr_state = " << new_flr_state << endl;
      renderer->set_flip_lr(new_flr_state);
      renderer->render();
      read_keypoints();
      legend = Mat();
      draw();
    }
    
    string selected_pt(Vec3d loc)
    {
      double min_dist = inf;
      string min_pt;
      int id = 0;
      auto update_min = [&](const pair<string,Vec3d>&keypoint)
      {
	name2id[keypoint.first] = id++;
	auto pt = keypoint.second;
	pt[2] = loc[2] = 0;
	double dist = std::sqrt((pt - loc).ddot(pt - loc));
	cout << "Dist = " << dist << endl;
	if(dist < min_dist)
	{
	  min_dist = dist;
	  min_pt = keypoint.first;
	}	
      };
      
      for(auto keypoint : keypoints)
      {
	if(set_keypoints.find(keypoint.first) == set_keypoints.end())
	  update_min(keypoint);
      }
      for(auto keypoint : set_keypoints)
	update_min(keypoint);
      return min_pt;
    }
    
    void reset()
    {
      renderer->get_hand_pose() = libhand::FullHandPose (renderer->get_scene_spec().num_bones());
      renderer->get_cam_spec().phi = deg2rad(90);
      renderer->get_cam_spec().tilt = deg2rad(90);
      renderer->get_cam_spec().theta = deg2rad(0);
      renderer->render();  
      tx = ty = 0;
      s = 1;
      read_keypoints();
      draw();
    }
    
    bool handle(int key_code)
    {
      //cout << "key_code: " << key_code << endl;
      if(static_cast<char>(key_code) == ' ')
	run_ik(); // space bar
      else if(key_code == 1048586 || key_code == KEY_CODES::ENTER) // enter
      {
	cout << "committing pose" << endl;
	return false;
      }
      else if(key_code == KEY_CODES::ESCAPE) // escape
      {
	cout << "Skipping frame" << endl;
	changed = false;
	return false;
      }
      else if(static_cast<char>(key_code) == 'f')
      {
	cout << "flipping hand" << endl;
	flip();
      }
      else if(static_cast<char>(key_code) == 'r')
      {
	cout << "resetting pose" << endl;
	set_keypoints.clear(); 
	reset();
      }
      else
      {
	cout << "unrecognized key code " << key_code << " " << static_cast<char>(key_code) << endl;
      }      
      
      return true;
    }
  };
  
  void label_full_hand_pose_on_mouse(int event, int x, int y, int flags, void*data)
  {
    //cout << "label_full_hand_pose_on_mouse" << endl;
    FullHandPoseAnnotationState*state = static_cast<FullHandPoseAnnotationState*>(data);
    
    switch(event)
    {
      case CV_EVENT_LBUTTONDOWN:
      {
	string selected_name = state->selected_pt(Vec3d(x,y,0));
	cout << "selected: " << selected_name << endl;
	state->selected_keypoint = selected_name;
      }
	break;
      case CV_EVENT_MOUSEMOVE:
      {
	if(state->keypoints.find(state->selected_keypoint) != state->keypoints.end())
	{
	  Vec3d&kp_loc = state->set_keypoints[state->selected_keypoint];
	  kp_loc[0] = x;
	  kp_loc[1] = y;
	  state->draw();
	}
      }
	break;
      case CV_EVENT_MBUTTONUP:
      {
	string selected_name = state->selected_pt(Vec3d(x,y,0));
	cout << "erasing " << selected_name << endl;
	state->set_keypoints.erase(selected_name);
	state->draw();
      }
	break;
      case CV_EVENT_LBUTTONUP:
      {
	if(state->selected_keypoint != "")
	  state->set_keypoints[state->selected_keypoint] = Vec3d(x,y,0);
 	state->selected_keypoint = "";
      }
	break;
      default:
	break;      
    };
  }
  
  //
  // example invokation
  // make -j16 && gdb ./deformable_depth -ex 'r label video data/depth_video/bailey.oni FULL_POSE=1 '
  //
  void label_libhand_theta(MetaData&metadata)
  {
    cout << "label_libhand_theta: " << metadata.get_filename() << endl;
    shared_ptr<ImRGBZ> im = metadata.load_im();
    if(im->RGB.empty())
    {
      log_once(printfpp("bad frame %s",metadata.get_filename().c_str()));
      return;
    }
    Rect HandBB = metadata.get_positives()["HandBB"];
    HandBB = rectResize(HandBB,2,2);
    
    LibHandRenderer*renderer = renderers::no_arm();
    
    FullHandPoseAnnotationState state;
    state.background = resize_pad(
      imageeq("",crop_and_copy<float>(im->Z,HandBB),false,false),im->RGB.size());
    state.RGB = im->RGB.clone();
    state.renderer = renderer;
    state.changed = false;
    state.reset();
    // try to read the existing data from the db.
    if(metadata.hasAnnotation("libhand pose"))
    {
      bool fliplr;
      getFullHandPose(metadata,state.set_keypoints,state.keypoints,fliplr,renderer->get_cam_spec(),renderer->get_hand_pose());
      renderer->set_flip_lr(fliplr);
      //state.draw();
      //state.run_ik();
      renderer->render();
      state.draw();
    }
    else
      cout << "No pose annotation found" << endl;
    int key_code;
    do
    {
      state.draw();
      setMouseCallback(winName,label_full_hand_pose_on_mouse,&state);
      key_code = waitKey(0);
      //break;
    } while(state.handle(key_code));
      
    // store the entered pose
    if(state.changed)
    {
      bool fliplr = renderer->get_flip_lr();
      putFullHandPose(metadata,state.set_keypoints,state.keypoints,fliplr,
		      renderer->get_cam_spec(),renderer->get_hand_pose());
    }
  }
 
#endif

  static atomic<unsigned long> id_counter(0);

  void getFullHandPose(MetaData&metadata,
		       map<string,Vec3d>&set_keypoints,map<string,Vec3d>&all_keypoints,
		       bool&fliplr,
		       libhand::HandCameraSpec&cam_spec,libhand::FullHandPose&hand_pose)
  {
    assert(metadata.hasAnnotation("libhand pose"));
    string annotation_string = metadata.getAnnotation("libhand pose");
    string filename;
    int fd = alloc_file_atomic_unique(
				      params::out_dir() + "/in_temp_file%lu.dat",filename,id_counter);
    ofstream ofs(filename); ofs << annotation_string; ofs.close();
    FileStorage fs(filename,FileStorage::READ);
    deformable_depth::read(fs["keypoints"],set_keypoints);
    if(!fs["all_keypoints"].empty())
      deformable_depth::read(fs["all_keypoints"],all_keypoints);
    fs["fliplr"] >> fliplr; 
    deformable_depth::read(fs["cam_spec"],cam_spec,libhand::HandCameraSpec());
    fs["hand_pose"] >> hand_pose;
    cout << "Loaded Annotation" << annotation_string << endl;
    close(fd);
    unlink(filename.c_str());
  }
  
  bool hasFullHandPose(MetaData&metadata)
  {
    return metadata.hasAnnotation("libhand pose") and metadata.getAnnotation("libhand pose") != "";
  }

  void putFullHandPose(MetaData&metadata,
		       map<string,Vec3d>&set_keypoints,map<string,Vec3d>&all_keypoints,
		       bool&fliplr,
		       libhand::HandCameraSpec&cam_spec,libhand::FullHandPose&hand_pose)
  {
    string filename;
    int fd = alloc_file_atomic_unique(
				      params::out_dir() + "/temp_file%lu.dat",filename,id_counter);
    FileStorage fs(filename,FileStorage::WRITE);
    string s;
    fs << "output" << "true";
    fs << "filename" << metadata.get_filename();
    fs << "cam_spec"; deformable_depth::write(fs,s,cam_spec);
    fs << "hand_pose" << hand_pose;
    fs << "keypoints"; deformable_depth::write(fs,s,set_keypoints);
    fs << "all_keypoints"; deformable_depth::write(fs,s,all_keypoints);
    fs << "fliplr" << fliplr;
    fs.release();
    ifstream ifs(filename);
    string written_string((istreambuf_iterator<char>(ifs)),istreambuf_iterator<char>());
    cout << "wrote: " << written_string << endl;
    metadata.putAnnotation("libhand pose",written_string);
    close(fd);
    unlink(filename.c_str());
  }
  
  // make -j16 && gdb ./deformable_depth -ex 'r label video data/depth_video/library1.oni LABEL_HAND=1 HAND=left'
  static void label_video(string video_name)
  {
#ifdef DD_ENABLE_OPENNI
    shared_ptr<Video> video = load_video(video_name);
    
    int vidlen = video->getNumberOfFrames();
    int trackbar_value = 0;
    cv::namedWindow(winName);
    cv::createTrackbar("frame_trackbar",winName,&trackbar_value,vidlen);	  
	  
    for(trackbar_value = 0; trackbar_value < vidlen; ++trackbar_value)
    {
      // get and display the frame
      cout << printfpp("frame: %d of %d",trackbar_value,vidlen) << endl;
      cv::setTrackbarPos("frame_trackbar",winName,trackbar_value);
      
      // annotate every 100th frame.
      if(trackbar_value % params::video_annotation_stride() == 0)
      {
	// allocate a name for the metadata
	string frame_metadata_name = 
	  printfpp("%s.frame%d",video_name.c_str(),trackbar_value);
	
	// get the video frame as a metadata object
	shared_ptr<MetaData_YML_Backed> aggregate = video->getFrame(trackbar_value,false);
	valid_ptr<MetaData_YML_Backed> metadata = aggregate->get_subdata_yml().at(g_params.require("HAND"));

	// show the metadata
	shared_ptr<ImRGBZ> im = metadata->load_im();
	//imageeq("video's depth",im->Z); waitKey_safe(0);
	
	if(g_params.has_key("LABEL_HAND"))
	{
	  // label the HandBB
	  label_handBB(*metadata);
	  
	  // label the keypoints
	  Rect HandBB = metadata->get_positives()["HandBB"];
	  if(HandBB != Rect())
	    labelJoints(*metadata);
	}
	if(g_params.has_key("LABEL_FREE"))
	{
	  Mat vis_z = imageeq("",im->Z,false,false);
	  image_safe("Image",horizCat(vis_z,im->RGB));
	  waitKey_safe(20);
	  cout << "is the hand in free space? ";
	  string freespace; std::cin >> freespace;
	  metadata->putAnnotation("hand in freespace?",freespace);
	}
	if(g_params.has_key("FULL_POSE"))
	{
	  Rect HandBB = metadata->get_positives()["HandBB"];
	  if(HandBB != Rect())
	    label_libhand_theta(*metadata);
	}
      }
    }
#else
    throw std::runtime_error("unsupported");
#endif
  }
  
  static void label_directory(int argc, char**argv)
  {
    if(argc < 4)
    {
      printf("usage: deformable_depth label aspect directory\n");
      return;
    }
    
    string aspect(argv[2]);
    string dir(argv[3]);
    vector<string> filenames = allStems(dir,".gz");
    sort(filenames.begin(),filenames.end());
    int idx = 0;
    for(string filename : filenames)
    {
      filename = dir + filename;
      //cout << bgPath << endl;
      //cout << ext << endl;
      //cv::namedWindow("RGB - Apply Labels Here",CV_WINDOW_NORMAL|CV_WINDOW_FREERATIO);
      // show RGB
      shared_ptr<MetaData_YML_Backed> metadata(new MetaData_DepthCentric(filename,false));
      shared_ptr<ImRGBZ> im = metadata->load_im();
      Mat RGB = im->RGB;
      Mat Z = im->Z;
      printf("RGB Size = %d rows %d cols\n",RGB.rows, RGB.cols);
      printf("labeling: %d of %d\n",idx,(int)filenames.size());
      
      // Display
      //imageeq("Z - eq",Z); //imagesc("Z - sc",Z);
      //image_safe("RGB",RGB);
      
      // ask for and apply the correct labels
      if(aspect == "HandBB")
      {
	metadata.reset(
	  new MetaData_DepthCentric(filename,false));
	label_handBB(*metadata);
	metadata.reset();
      }
      else if(aspect == "pose_name")
      {
	metadata.reset();
	labelPose(RGB,filename);
      }
      else if(aspect == "left_right")
	labelLeftRight(RGB,*metadata);
      else if(aspect == "RGBSubArea")
	labelRGBSubArea(*im);
      else if(aspect == "joints")
	labelJoints(*metadata);
      else
      {
	printf("Error: Bad Aspect Name\n");
	return;
      }
      
      idx++;
    } // for each filename
    printf("All images labeled in directory\n");    
  }    
  
  void label(int argc, char**argv)
  {
    string aspect(argv[2]);
    string location(argv[3]);
    
    if(aspect == "HandBB" || aspect == "pose_name" ||
      aspect == "left_right" || aspect == "RGBSubArea" ||
      aspect == "joints")
      label_directory(argc, argv);
    else if(aspect == "video")
      label_video(location);
  }

  /// 
  /// SECTION: Serialization
  /// 
  void write(cv::FileStorage&fs, std::string&, const deformable_depth::PointAnnotation&note)
  {
    fs << "{";
    fs << "click" << note.click;
    fs << "visibility" << note.visibility;
    //fs << "RGB" << RGB;
    fs << "code" << note.code;
    fs << "}";
  }

  void read(const cv::FileNode&fn, deformable_depth::PointAnnotation&note, deformable_depth::PointAnnotation)
  {
    fn["click"] >> note.click;
    fn["visibility"] >> note.visibility;
    string s_code;
    fn["code"] >> s_code;
    note.code = 'i';
  }
}
