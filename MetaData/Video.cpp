/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Video.hpp"
#include "params.hpp"
#include "ONI_Video.hpp"
#include "util.hpp"
#include <boost/filesystem.hpp>
#ifdef DD_CXX11
#include <boost/filesystem/path.hpp>
#endif
#include "IKSynther.hpp"
#include "ICL_MetaData.hpp"
#include "MetaData_Pool.hpp"
#include "TestModel.hpp"
#include "RegEx.hpp"
#include "PXCSupport.hpp"
#include "MiscDataSets.hpp"
#include "Cache.hpp"
#include "NYU_Hands.hpp"
#include "ASTAR_Hands.hpp"
#include "Orthography.hpp"
#include "Colors.hpp"
#include "PTB.hpp"

namespace deformable_depth 
{
  static string titleOfDirec(string filename)
  {
    boost::filesystem::path path(filename);
    return path.parent_path().leaf().c_str();    
  }

  //
  // SECTION: Video, abstract base class
  //
  Video::~Video()
  {
  }
  
  bool Video::is_frame_annotated(int index)
  {
    return index % annotation_stride() == 0;
  }

  int Video::annotation_stride() const
  {
    return params::video_annotation_stride();
  }

  shared_ptr<MetaData> Video::getFrame_raw(int index,bool read_only)
  {
    return getFrame(index,read_only);
  }
  
  //
  // SECTION: Video: OpenCV/Def. Depth Video
  //  
  shared_ptr< MetaData_YML_Backed > Video_CV_DD::HandDatumAssemblage
  (shared_ptr<BaselineDetection> annotation,shared_ptr<ImRGBZ> im)
  {
    // setup the image
    shared_ptr<Metadata_Simple> metadata(new Metadata_Simple(uuid(),true));
    metadata->setIm(*im);    

    // now setup the annotations
    if(boost::regex_match(annotation->notes,boost::regex(".*left.*",boost::regex::icase)))
    {
      metadata->set_is_left_hand(true);
      log_file << "metadata->set_is_left_hand(true);" << endl;
    }
    else
    {
      metadata->set_is_left_hand(false);
      log_file << "metadata->set_is_left_hand(false);" << endl;
    }

    // set the bbs
    map<string,AnnotationBoundingBox> parts;
    metadata->set_HandBB(annotation->bb);
    for(auto && sub_det : annotation->parts)
    {
      static_cast<Rect_<double>&>(parts[sub_det.first]) = AnnotationBoundingBox(sub_det.second.bb,true);
    }
    metadata->setPositives(parts);

    return metadata;
  }

  shared_ptr< MetaData_YML_Backed > Video_CV_DD::getFrame(int index, bool read_only)
  {
    lock_guard<decltype(monitor)> l(monitor);
    log_file << "Video_CV_DD::getFrame: " << index << endl;

    ensure_read_ready();
    // load the image for frame and convert to a metadata
    string key = printfpp("frame%d",(int)index);
    if((*store_read)[key].empty())
    {
      log_file << "warning: (*store_read)[key].empty()" << endl;
      return nullptr;
    }
    shared_ptr<ImRGBZ> im; (*store_read)[key] >> im;
    
    // strip the image filename and get the metadata filename
    // test_data11.oni.frame0.im.gz => test_data11.oni.frame0(.labels.yml)
    const string img_filename = 
      boost::regex_replace(im->filename,boost::regex(".im.gz"),string(""));
    string oni_filename = boost::regex_replace(    filename,boost::regex("/yml_videos/"),"/depth_video/");
           oni_filename = boost::regex_replace(oni_filename,boost::regex(".oni.yml.gz"),".oni");	   
    string metadata_filename = 
	    printfpp("%s.frame%d",oni_filename.c_str(),index);
    log_once(safe_printf("Video_CV_DD::getFrame replaced % => %",im->filename,oni_filename));
      
    // load the metadata
    vector<shared_ptr<BaselineDetection>> gts;
    read_in_map((*store_read)[key + "gts"],gts);
    map<string, shared_ptr<MetaData_YML_Backed> > sub_data;
    for(auto && gt : gts)
    {
      shared_ptr< MetaData_YML_Backed > new_assemblage = HandDatumAssemblage(gt,im);
      auto poss = new_assemblage->get_positives();
      Rect handBB = poss["HandBB"];
      if(handBB != Rect())
      {
	log_file << "loaded: " << index << " \"" << gt->notes << "\"" << endl;
	sub_data[gt->notes] = new_assemblage;
      }
    }

    // load the left right annotation
    shared_ptr< MetaData_YML_Backed > result;
    if(sub_data.empty())
    {      
      auto metadata = make_shared<Metadata_Simple>(metadata_filename,read_only);
      metadata->setIm(*im);
      log_file << "Video_CV_DD::getFrame simple" << endl;
      result = metadata;
    }
    else
    {
      result = make_shared<MetaDataAggregate>(sub_data);
      log_file << "Video_CV_DD::getFrame aggregate" << endl;
    }
    result->change_filename(metadata_filename);

    return result;
  }

  void Video_CV_DD::ensure_read_ready() const
  {
    lock_guard<decltype(monitor)> lock(monitor);
    if(store_read == nullptr)
      store_read.reset(new FileStorage(filename,FileStorage::READ));
  }

  void Video_CV_DD::ensure_store_ready()
  {
    lock_guard<decltype(monitor)> lock(monitor);

    while(store_write == nullptr)
    {
      cout << safe_printf("Video_CV_DD::ensure_store_ready allocating %",filename) << cout;
      store_write.reset(new FileStorage(filename,FileStorage::WRITE));
    }

    while(not store_write->isOpened())
    {
      cout << safe_printf("Video_CV_DD::ensure_store_ready opening %",filename) << cout;
      store_write->open(filename,FileStorage::WRITE);
    }
    
    assert(store_write->isOpened());
  }
  
  void Video_CV_DD::writeFrame(shared_ptr< ImRGBZ >& frame, const vector< shared_ptr<BaselineDetection> >&gts,bool leftP,int index)
  {
    ensure_store_ready(); 
    
    string key = printfpp("frame%d",index < 0?(int)length:index);
    *store_write << key << frame;
    *store_write << (key + "leftP") << leftP;
    string s;
    *store_write << (key + "gts"); write_in_map(*store_write,s,gts);
    //*store_write << (key + "gts") << gts;     
    //*store_write << (key + "poss"); write(*store_write,s,poss);

    length++;
  }

  void Video_CV_DD::setAnnotationStride(int stride)
  {
    ensure_store_ready(); 

    *store_write << "annotation_stride" << stride;
  }

  int Video_CV_DD::annotation_stride() const 
  {    
    ensure_read_ready();
    int annotation_stride; (*store_read)["annotation_stride"] >> annotation_stride;
    return annotation_stride;
  }
  
  int Video_CV_DD::getNumberOfFrames()
  {
    lock_guard<decltype(monitor)> l(monitor);
    if(store_read == nullptr)
      store_read.reset(new FileStorage(filename,FileStorage::READ));
   
    if(length == -1)
    {
      // find the max frame numbers
      for(auto iter = store_read->root().begin(); iter != store_read->root().end(); ++iter)  
      {
	FileNode current_node = *iter;
	string node_name = current_node.name();
	vector<string> matches = deformable_depth::regex_match(node_name,boost::regex("\\d+"));
	for(string s : matches)
	{
	  int frame_number = fromString<int>(s);
	  length = std::max(length,frame_number);
	}
      }
    }
 
    //int frame_cnt = store_read->root().size();
    return length;
  }

  Video_CV_DD::Video_CV_DD(string filename) : 
    filename(filename),
    length(-1)
  {
  }

  Video_CV_DD::~Video_CV_DD()
  {
  }
  
  string Video_CV_DD::get_name()
  {
    boost::filesystem::path v_path(filename);
    return v_path.leaf().string();
  }
  
  // section, show a frame from a video
  void show_frame()
  {     
    // parse the command line
    string video_filename = g_params.get_value("video");
    int frame_index = fromString<int>(g_params.get_value("frame"));
   
    Mat frame_image;
    Mat depth_frame;
    if(boost::regex_match(video_filename,boost::regex(".*oni")))
    {
      cout << "Using dd::ONI_Video" << endl;
#ifdef DD_ENABLE_OPENNI
      ONI_Video video(video_filename);
      shared_ptr<MetaData> frame = video.getFrame_raw(frame_index,true);
      shared_ptr<ImRGBZ> im = frame->load_im();
      frame_image = im->RGB.clone();
      depth_frame = im->Z.clone();
      cout << "depth size: " << depth_frame.size() << endl;
      cout << "color size: " << frame_image.size() << endl;
      // let the user crop
      Rect cropRect = getRect("Crop Rect", frame_image);
      cout << "crop Rect = " << cropRect << endl;
      cout << "skin_ratio = " << im->skin_ratio(cropRect) << endl;
      frame_image = frame_image(cropRect);
      depth_frame = depth_frame(cropRect);
#else
      assert(false);
#endif
    }
    else
    {
      // grab the frame
      cout << "Using cv::VideoCapture" << endl;
      cv::VideoCapture frame_reader(video_filename);
      cout << "video's frame count: " << frame_reader.get(CV_CAP_PROP_FRAME_COUNT) << endl;
      frame_reader.set(CV_CAP_PROP_POS_FRAMES,frame_index);
      frame_reader.read(frame_image);
      frame_reader.release();
    }
    
    // show it
    image_safe("Frame - RGB",frame_image);
    if(!depth_frame.empty())
      imageeq("Frame - Depth",depth_frame);
    waitKey_safe(0);
  }  

  void export_annotations_codalab(int trackbar_value,shared_ptr<MetaData> datum)
  {
    //string prefix = safe_printf(params::our_dir() + "annotations_%d",trackbar_value);
    //ofstream ofs(prefix);
    // TODO?
  }
  
  // make -j16 &&  ./deformable_depth show_video VIDEO=data/depth_video/bailey.oni NO_BACKGROUND_REPAINTER=TRUE
  void show_video()
  {
    shared_ptr< Video > video = load_video(g_params.require("VIDEO"));
    if(!video)
    {
      cout << "Failed to load Video!" << endl;
      return;
    }
    cout << "Loaded video with frame count = " << video->getNumberOfFrames() << endl;
    int trackbar_value;
    namedWindow("frame");
    createTrackbar("frame_trackbar","frame",&trackbar_value,video->getNumberOfFrames());
    
    shared_ptr<MetaData> datum = video->getFrame(0,true);
    for(trackbar_value = 0; true; ++trackbar_value)
    {
      // get the images
      shared_ptr<MetaData> newDatum = video->getFrame(trackbar_value,true);      
      if(newDatum)
	datum = newDatum;
      Mat RGB(params::depth_vRes,params::depth_hRes,DataType<Vec3b>::type,Scalar(0,0,0));
      Mat Z(params::depth_vRes,params::depth_hRes,DataType<float>::type,Scalar(0,0,0));
      if(datum)
      {
	RGB = datum->load_im()->RGB.clone();
	Z = datum->load_im()->Z.clone();
	// show to codalab
	export_annotations_codalab(trackbar_value,datum);
      }
      Mat visZ = imageeq("",Z,false,false);

      // decorate the images
      if(g_params.option_is_set("SHOW_VIDEO_ANNOTATIONS"))
      {
	for(auto && sub_datum : datum->get_subdata())
	{
	  Rect handBB = sub_datum.second->get_positives()["HandBB"];
	  cv::rectangle(RGB,handBB.tl(),handBB.br(),Scalar(0,0,255));
	  cv::rectangle(visZ,handBB.tl(),handBB.br(),Scalar(0,0,255));
	  log_file << "drawing: " << sub_datum.first << " " << handBB.size() << endl;
	  // draw the parts
	  for(auto && part : sub_datum.second->get_positives())
	  {
	    if(part.first != "HandBB")
	    {
	      cv::rectangle(RGB,part.second.tl(),part.second.br(),Scalar(255,0,0));
	      cv::rectangle(visZ,part.second.tl(),part.second.br(),Scalar(255,0,0));
	    }
	  }
	  // draw the keypoitns
	  for(auto && kp_name : sub_datum.second->keypoint_names())
	  {
	    pair<Point3d,bool> kp = sub_datum.second->keypoint(kp_name);
	    Point2i kp_center(kp.first.x,kp.first.y);
	    cv::circle(RGB,Point(kp_center),5,toScalar(BLUE),-1);
	    cv::circle(visZ,Point(kp_center),5,toScalar(BLUE),-1);
	  }
	}
      }

      // paint an orthographic version
      //shared_ptr<ImRGBZ> im = datum->load_im();
      //Mat vis_ortho = imageeq("",paint_orthographic(*im),false,false);

      // show the images
      image_safe("frame",horizCat(RGB,visZ));
      setTrackbarPos("frame_trackbar","frame",trackbar_value);
      waitKey_safe(20);

      if(trackbar_value >= video->getNumberOfFrames())
	trackbar_value = 0;
    }
  }
  
  //
  // SECTION: Video format conversion
  //
  static void export_frame_2_dataset(
    shared_ptr<Video> input_video,
    string export_dir, 
    shared_ptr<MetaData> metadata,
    int frameIter)
  {
    shared_ptr<ImRGBZ> im = metadata->load_im();
    Mat vis = horizCat(imageeq("Converting Depth Frame",im->Z,false,false),im->RGB);
    image_safe("converting",vis);
    log_im(input_video->get_name() + std::to_string(frameIter),vis);
    
    // save to the export directory
    if(export_dir != "")
    {
      // write the depths
      imwrite(export_dir + printfpp("depth_%05d",(int)frameIter) + ".exr",im->Z);
      imwrite(export_dir + printfpp("depth_%05d",(int)frameIter) + ".png",depth_float2uint16(im->Z));
      imwrite(export_dir + printfpp("raw_depth_%05d",(int)frameIter) + ".png",depth_float2uint16(metadata->load_raw_depth()));
    
      // write the RGB
      Mat ZVis = imageeq("",im->Z,false,false);	  
      Mat rawRGB = metadata->load_raw_RGB();
      Mat RGB_Vis; cv::resize(rawRGB,RGB_Vis,im->Z.size());
      imwrite(export_dir + printfpp("registered_color_%05d",(int)frameIter) + ".png",im->RGB);
      imwrite(export_dir + printfpp("raw_color_%05d",(int)frameIter) + ".png",rawRGB);
      imwrite(export_dir + printfpp("both_%05d",(int)frameIter) + ".png",vertCat(RGB_Vis,ZVis));

      // write the semantic segmentation and annotations
      string note_file = printfpp((export_dir + "/gen%05d.annotations").c_str(),(int)frameIter);
      metadata->export_annotations(note_file);
      string kp_file = printfpp((export_dir + "/gen%05d.keypoints").c_str(),(int)frameIter);
      metadata->export_keypoints(kp_file);
      //string gt_file = printfpp((export_dir + "/gt_%05d.csv").c_str(),(int)frameIter);
      //metadata->export_one_line_per_hand(gt_file);
      Mat seg = metadata->getSemanticSegmentation();
      if(not seg.empty())
	imwrite(export_dir + printfpp("seg_%05d",(int)frameIter) + ".png",seg);	        
    }  
  }

  static void do_export_video(shared_ptr<Video>   input_video, string YML_filename, string export_dir = "")
  {
    TaskBlock do_export_video("do_export_video");
    int n_frames = input_video->getNumberOfFrames();
    for(int frameIter = 0; frameIter < n_frames; frameIter += input_video->annotation_stride())
    {
      do_export_video.add_callee([&,frameIter]()
				 {
				   shared_ptr<MetaData> metadata = input_video->getFrame_raw(frameIter,true);
				   if(metadata)
				     export_frame_2_dataset(input_video,export_dir, metadata,frameIter);
				 });      
    }    
    do_export_video.execute();
  }
 
  static void do_convert_export_video(shared_ptr<Video>   input_video, string YML_filename, string export_dir = "")
  {
    // create the videos    
    log_file << "do_convert_export_video: " << YML_filename << 
      " export_dir = " << export_dir << 
      " frames = " << input_video->getNumberOfFrames() << endl;
    Video_CV_DD cv_video(YML_filename);    
    cv_video.setAnnotationStride(input_video->annotation_stride());
    
    // copy each frame from the Input_Video to the CV Video
    int n_frames = input_video->getNumberOfFrames();
    for(int frameIter = 0; frameIter < n_frames; frameIter += 1)
    {
      if(!input_video->is_frame_annotated(frameIter))
	continue;
      
      // write the image
      cout << printfpp("frame %d of %d",frameIter,n_frames) << endl;
      shared_ptr<MetaData> metadata = input_video->getFrame_raw(frameIter,true);
      if(metadata)
      {
	// write the annotations?
	auto input_poss = metadata->get_positives();	
	bool leftP = metadata->leftP();

	// export to the dataset
	export_frame_2_dataset(input_video,export_dir, metadata, frameIter);
	
	// commit the frame
	shared_ptr<ImRGBZ> im = metadata->load_im();
	auto gts = metadata->ground_truths();
	cv_video.writeFrame(im,gts,leftP,frameIter);
      }
    }
  }

  void convert_video(int argc, char**argv)
  {
#ifdef DD_ENABLE_OPENNI
    {
      // get the filenames
      string oni_filename = g_params.get_value("INPUT");
      string YML_filename  = g_params.get_value("YML");
    
      shared_ptr<Video>   input_video = load_video(oni_filename);
      do_convert_export_video(input_video,YML_filename);
    }

    std::exit(0);
#else
    throw std::runtime_error("unsupported");
#endif    
  }
  
  // rm -rv ./outs/*; rm -rv ./data/yml_videos/*; make -j16 && gdb ./deformable_depth -ex 'catch throw' -ex 'r convert_all_videos CFG_FILE=scripts/hand_convert.cfg'
  // ./www/compress_vids.py  
  void convert_all_videos()
  {
    cout << "+convert_all_videos()" << endl;
    TaskBlock convert_all_videos("convert_all_videos");
    vector<string> vids = test_video_filenames();
    string message = safe_printf("+convert_all_videos() % vids",vids.size());
    log_file << message << endl;
    cout << message << endl;
    static std::recursive_mutex m; 

    auto export_vid = [&](string input_filename, bool convert)
    {
      // (1) load the video
      unique_lock<recursive_mutex> l(m);
      cout << "converting " << input_filename << endl;
      shared_ptr<Video>   input_video = load_video(input_filename);

      // (2) generate the output filename
      string output_filename = string("data/yml_videos/") + input_video->get_name() + ".yml.gz";
      log_once(printfpp("converting %s to %s",input_filename.c_str(),output_filename.c_str()));

      // (3) generate the output directory name
      boost::filesystem::path export_path(
	string("./data/yml_videos/") + input_video->get_name() + "/");
      log_file << safe_printf("export_dir = %",export_path.string()) << endl;

      // (4) create the output directory (if necessay)
      if(!boost::filesystem::exists(export_path.string()))
	boost::filesystem::create_directory(export_path.string());
      l.unlock();
      // (5) write the video into the output directory
      if(convert)
	do_convert_export_video(input_video,output_filename,export_path.string());
      else
	do_export_video(input_video,output_filename,export_path.string());
    };

    for(string & input_filename : vids)
    {
      convert_all_videos.add_callee([&,input_filename]()
				    {
				      export_vid(input_filename,true);
				    });
    }
    for(auto & export_only_pair : g_params.matching_keys("EXPORT_ONLY_VIDEO_FILE.*"))
    {
      convert_all_videos.add_callee([&,export_only_pair]()
				    {
				      export_vid(export_only_pair.second,false);
				    });
    }
    convert_all_videos.execute();
    cout << "-convert_all_videos()" << endl;
  }

  ///
  /// SECTION: LoadVideo
  ///
  static shared_ptr< Video > do_load_video(string filename)
  {
    // we have three types of video we support
    // .ONI
    // .yml.gz
    // directory
    boost::filesystem::path vid_path(filename);    
    shared_ptr<Video> ptr;
    if(vid_path.extension() != "")
    {
      log_file << printfpp("Loading video %s w/ ext = %s",filename.c_str(),vid_path.extension().c_str()) << endl;
      if(vid_path.extension() == ".ASTAR_HANDS")
	ptr = make_shared<ASTAR_Video>(fromString<int>(vid_path.stem().string()));
      else if(vid_path.extension() == ".PTB")
	ptr = make_shared<PTB_Video>(vid_path.stem().string());
      else if(vid_path == "0.NYU_HANDS")
	ptr = make_shared<NYU_Video>();
      else
#ifdef DD_ENABLE_OPENNI
      if(vid_path.extension() == ".oni")
	ptr = shared_ptr<Video>(new ONI_Video(filename));
      else
#endif
      if(vid_path.extension() == ".gz")
	ptr = shared_ptr<Video>(new Video_CV_DD(filename));
      else if(vid_path.extension() == ".ICL")
	ptr = shared_ptr<Video>(new ICL_Video(filename));
    }
    else
    {
      log_file << safe_printf("% has no extension",filename) << endl;
#ifdef DD_ENABLE_HAND_SYNTH
      if(boost::regex_match(filename,boost::regex(".*ego.*",boost::regex::icase)))
	ptr = shared_ptr<Video>(new VideoDirectory(filename));
      else
	ptr = shared_ptr<Video>(new VideoMetaDataDirectory(filename));
#endif
    }
    
    log_file << safe_printf("do_load_video: % %",filename,typeid(ptr.get()).name()) << endl;
    return ptr;
  }

  shared_ptr< Video > load_video(string filename)
  {
    static Cache<shared_ptr<Video> > vid_pool;

    return vid_pool.get(filename,[&]()
			{
			  return do_load_video(filename);
			});
  }
  
  ///
  /// SECTION: Video Directory
  ///
#ifdef DD_ENABLE_HAND_SYNTH
  string VideoDirectory::get_name()
  {
    return title;
  }

  void VideoDirectory::loadAnnotations(shared_ptr<MetaDataAggregate>&metadata,int index,const Mat&D)
  {
    log_once(safe_printf("VideoDirectory::loadAnnotations++  %",metadata->get_filename()));    
    // write the annotations    
    string annotation_filename1 = filename + "/" + printfpp("frame%d-1-Depth.txt",index);
    string annotation_filename2 = filename + "/" + printfpp("frame%d-2-Depth.txt",index);
    IK_Grasp_Pose pose;
    if(boost::filesystem::exists(annotation_filename1))
    {
      // 1 is right hand
      IK_Grasp_Pose pose1 = greg_load_grasp_pose_one(filename,annotation_filename1,false);
      setAnnotations(metadata->get_subdata_yml().at("right_hand"),pose1);
      metadata->get_subdata_yml().at("right_hand")->set_is_left_hand(false);
      log_once(safe_printf("VideoDirectory::loadAnnotations got RIGHT %",metadata->get_filename()));
    }
    if(boost::filesystem::exists(annotation_filename2))
    {
      // 2 is left hand
      IK_Grasp_Pose pose2 = greg_load_grasp_pose_one(filename,annotation_filename2,false);

      setAnnotations(metadata->get_subdata_yml().at("left_hand"),pose2);
      metadata->get_subdata_yml().at("left_hand")->set_is_left_hand(true);
      log_once(safe_printf("VideoDirectory::loadAnnotations got LEFT %",metadata->get_filename()));
    }
  }

  shared_ptr<VideoDirectoryDatum> load_video_directory_datum(string directory, int index)
  {
    string depth_filename = directory + "/" + printfpp("frame%d.mat.exr",index);
    string rgb_filename = directory + "/" + printfpp("frame%d.mat.png",index);
    log_file << "DirVideo: " << depth_filename << " " << rgb_filename << endl;
    Mat D   = imread(depth_filename,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
    if(D.empty())
    {
      cout << "warning couldn't load " << depth_filename << endl;
      return nullptr;
    }
    D = 0.1 * D.t();
    Mat RGB = imread(rgb_filename);
    
    // load the UV Map
    string u_filename = directory + "/" + printfpp("frame%d.matu.exr",index);
    string v_filename = directory + "/" + printfpp("frame%d.matv.exr",index);
    Mat u = imread(u_filename,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
    Mat v = imread(v_filename,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
    if(u.empty() or v.empty() or u.size() != v.size())
    {
      log_file << "warning bad UV map" << endl;
      return nullptr;
    }
    u.convertTo(u,DataType<float>::type);
    u = u.reshape(0,u.cols);
    v.convertTo(v,DataType<float>::type);
    v = v.reshape(0,v.cols);
    Mat UV; cv::merge(vector<Mat>{u,v},UV);
    assert(UV.type() == DataType<Vec2f>::type);

    return make_shared<VideoDirectoryDatum>(D,RGB,UV);
  }

  // here we load a video converted from Greg's format using the matlab/mat2exr.m script
  shared_ptr< MetaData_YML_Backed > VideoDirectory::getFrame(int index, bool read_only)
  {
    index++;
    // load the image data
    shared_ptr<VideoDirectoryDatum> vdd = load_video_directory_datum(filename,index);
    if(!vdd)
      return nullptr;

    // form the metadatum
    string metadata_filename = filename + "/" + printfpp("frame%d_metadata.yml.gz",index);
    shared_ptr<MetaDataAggregate> metadata = form_Metadatum(vdd->RGB,vdd->D,vdd->UV,
							    get_name() + "_" + printfpp("_%d_raw",index),metadata_filename);
    if(is_frame_annotated(index))
      loadAnnotations(metadata,index,vdd->D);
    
    return metadata;
  }

  int VideoDirectory::getNumberOfFrames()
  {
    if(length == -1)
    {
      vector<string> gt_files = find_files(filename,boost::regex(".*exr"));      
      length =  gt_files.size();
    }
    return length;
  }

  VideoDirectory::VideoDirectory(string filename) : 
    filename(filename)
  {
    // extract the video's title from its filename   
    title = titleOfDirec(filename);
    log_once(safe_printf("opening VideoDirectory = %",filename));
  }

  VideoDirectory::~VideoDirectory()
  {
  }
  
  bool VideoDirectory::is_frame_annotated(int index)
  {
    string annotation_filename1 = filename + "/" + printfpp("frame%d-1-Depth.txt",index);
    string annotation_filename2 = filename + "/" + printfpp("frame%d-2-Depth.txt",index);
    string frame_gt_file = printfpp("frame%d_RGB_GT.png",index);
    string rgb_vis_path = filename + "/" + frame_gt_file;
    
    bool frame_annotated = false;
    for(auto path : vector<string>{annotation_filename1,annotation_filename2,rgb_vis_path})
      if(boost::filesystem::exists(path))
	 frame_annotated = true;

    if(frame_annotated)
    {
      log_once(safe_printf("VideoDirectory::is_frame_annotated YES %",annotation_filename1));
      return true;
    }
    else
    {
      log_once(safe_printf("VideoDirectory::is_frame_annotated NO %",annotation_filename1));
      return false;
    }
  }
#endif

  ///
  /// SECTION: VideoMetaDataDirectory
  /// 
  VideoMetaDataDirectory::VideoMetaDataDirectory(string filename) : filename(filename)
  {
    // extract the video's title from its filename
    title = titleOfDirec(filename);

    // load the frames
    frames = metadata_build_all(filename, false, true);
    log_once(safe_printf("VideoMetaDataDirectory::VideoMetaDataDirectory % %",filename,frames.size()));
  }

  VideoMetaDataDirectory::~VideoMetaDataDirectory()
  {
  }

  shared_ptr<MetaData_YML_Backed> VideoMetaDataDirectory::getFrame(int index,bool read_only)
  {
    shared_ptr<MetaData> datum = getFrame_raw(index,read_only);
    shared_ptr<MetaData_Pooled> pooled_datum = 
      std::dynamic_pointer_cast<MetaData_Pooled>(datum);
    assert(pooled_datum);    
    shared_ptr<MetaData_YML_Backed> yml_datum = 
      std::dynamic_pointer_cast<MetaData_YML_Backed>(pooled_datum->getBackend());
    assert(yml_datum);
    return yml_datum;
  }

  shared_ptr<MetaData> VideoMetaDataDirectory::getFrame_raw(int index,bool read_only)
  {
    int frame_index = clamp<int>(0,index,frames.size());
    shared_ptr<MetaData> datum = frames[frame_index];
    assert(datum);
    return datum;
  }

  int VideoMetaDataDirectory::getNumberOfFrames()
  {
    return frames.size();
  }

  string VideoMetaDataDirectory::get_name()
  {
    return title;
  }

  bool VideoMetaDataDirectory::is_frame_annotated(int index)
  {
    return true;
  }

  int VideoMetaDataDirectory::annotation_stride() const 
  {
    string annotation_file = filename + "annotation_stride";
    if(boost::filesystem::exists(annotation_file))
    {
      log_once(string("using annotation file = ") + annotation_file);
      ifstream ifs(annotation_file);
      int stride; ifs >> stride;
      return stride;
    }
    else
    {
      return 1;
      //return params::video_annotation_stride();
    }
  }

  //  
  // SECTION: Convert public to private video names
  // 
  string vid_public_name(string private_name)
  {
    string map_filename = "www/eq_names.csv";
    ifstream ifs(map_filename);
    while(ifs)
    {
      string line; std::getline(ifs,line);
      istringstream iss(line); 
      string priv; std::getline(iss,priv,',');
      string publ; std::getline(iss,publ,',');
      if(priv == private_name)
	return publ;
    }
    string err_msg = safe_printf("vid_public_name bad private name %",private_name);
    cout << err_msg << endl;
    log_file << err_msg << endl;
    throw std::runtime_error(err_msg);
  }

  int vid_type_id(string vidName)
  {
    if(boost::regex_match(vidName,boost::regex(".*(Greg|Marga).*")))
      return 5;
    if(boost::regex_match(vidName,boost::regex(".*sequ.*")))
      return 4;
    else if(boost::regex_match(vidName,boost::regex(".*NYU.*")))
      return 3;
    else if(boost::regex_match(vidName,boost::regex(".*ASTAR.*")))
      return 2;
    else if(boost::regex_match(vidName,boost::regex(".*ICL.*")))
      return 1;
    else 
      return 0;
  }
}

