/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include <boost/algorithm/string.hpp>
#include <boost/graph/graph_concepts.hpp>

#include "Plot.hpp"
#include "util.hpp"
#include "ONI_Video.hpp"
#include "BaselineDetection.hpp"
#include "Log.hpp"
#include "Faces.hpp"
#include "Cache.hpp"
#include "Colors.hpp"

namespace deformable_depth 
{
  /// general utility functions
  void write_set_fonts(ostream&os,int fontsize)
  {
    os << printfpp("set(gca,'FontSize',%d);",fontsize) << endl;
  }
  
  ////
  /// SECTION: MatplotlibPlot
  ///
  MatplotlibPlot::MatplotlibPlot(string save_file) : 
    save_file(save_file), plot_script(save_file)
  {
    assert(plot_script.is_open());
  }
  
  MatplotlibPlot::~MatplotlibPlot()
  {
    plot_script.close();    
  }

  void MatplotlibPlot::put(string s)
  {
    plot_script << s << endl;
  }
  
  void MatplotlibPlot::footer()
  {
    put("ax = plt.gca()");
    put("handles, labels = ax.get_legend_handles_labels()");
    put("ax.legend(handles, labels, loc = 2)");
    put("plt.show()");  
  }

  void MatplotlibPlot::header()
  {
    put("from pylab import *");
    put("from numpy import *");
    put("from scipy.stats import *");
  }
  
  ///
  /// SECTION: PrecisionRecallPlot
  ///
  
  void PrecisionRecallPlot::add_plot(PerformanceRecord record, string label)
  {
    records[label] = record;
  }

  void PrecisionRecallPlot::add_point(string title,PerformanceRecord record)
  {
    points[title] = record;
  }

  PrecisionRecallPlot::PrecisionRecallPlot(string filename, string type) : 
    filename(filename), type(type)
  {
    assert(type == "finger" || type == "hand" || type == "joint_pose" || type == "finger_agnostic");
  }
  
  void PrecisionRecallPlot::plot_one(
    ostream&out, bool curve, string title, PerformanceRecord& record, int number)
  {
    curve = true;
    
    // customize by type
    IScores * scores;
    if(type == "finger")
    {
      scores = &record.getFingerScores();
    }
    else if(type == "joint_pose")
    {
      scores = &record.getJointPoseScores();
    }
    else if(type == "hand")
    {
      scores = &record.getHandScores();
    }
    else if(type == "finger_agnostic")
    {
      scores = &record.getAgnosticFingerScores();
    }
    else
      assert(false);
    
    vector<double> P,R, V;
    if(curve)
    {
      scores->compute_pr(P,R,V);      
    }
    else
    {
      P.push_back(scores->p(-inf));
      R.push_back(scores->r(-inf));
      V.push_back(scores->v(-inf));
    }
    
    // write the TP and FP vectors
    string PVar = printfpp("P%s",varName(title).c_str());
    string RVar = printfpp("R%s",varName(title).c_str());
    string VVar = printfpp("V%s",varName(title).c_str());
    out << printfpp("%s = ",PVar.c_str()) << (vector<double>)P << endl;
    out << printfpp("%s = ",RVar.c_str()) << (vector<double>)R << endl;
    out << printfpp("%s = ",VVar.c_str()) << (vector<double>)V << endl;    
    // do the "max" trick
    out << printfpp("%s = cummax(%s(end:-1:1));",PVar.c_str(),PVar.c_str()) << endl;
    out << printfpp("%s = %s(end:-1:1);",PVar.c_str(),PVar.c_str()) << endl;    
    
    if(curve)
    {  
      string line_spec = (number<7)?"-":"--";
      out << safe_printf("h = plot(%,%,'%','DisplayName','%','LineWidth',5)",
			 RVar.c_str(),PVar.c_str(),line_spec,title.c_str()) << endl;      
      
      // print the error bars
      if(V.size() > 0)
      {
	out << "color = get(h,'Color');" << endl;
	out << printfpp("jbfill(%s,%s+%s,%s-%s,color,rand(1,3),0,.15)",
	  RVar.c_str(),PVar.c_str(),VVar.c_str(),PVar.c_str(),VVar.c_str()
	) << endl;
	//out << printfpp("errorbar(%s,%s,%s,'DisplayName','%s','LineWidth',5)",
	//		RVar.c_str(),PVar.c_str(),VVar.c_str(),title.c_str()) << endl;	
      }
    }
    else // plot a point
    {
      out << printfpp("errorbar(%s,%s,%s,'DisplayName','%s','LineWidth',5)",
      		RVar.c_str(),PVar.c_str(),VVar.c_str(),title.c_str()) << endl;
    }
  }
  
  //set(0,'DefaultAxesColorOrder',[1 0 0;0 1 0;0 0 1;1 .5 0; .79 .12 .48])
  PrecisionRecallPlot::~PrecisionRecallPlot()
  {
    ofstream out(filename);
    out << "hold all" << endl;
    int plot_number = 0;

    // generate the PR poitns
    for(auto && pt : points)
    {
      plot_one(out, false, pt.first, pt.second,plot_number++);
    }
    
    // generate the PR lines
    for(auto && subplot : records)
    {
      plot_one(out, true, subplot.first, subplot.second,plot_number++);
    }
    out << "legend show" << endl;
    out << "axis([0 1 0 1]);";
    int fontsize = 14;
    out << printfpp("xlabel('recall','FontSize',%d)",fontsize) << endl;
    out << printfpp("ylabel('precision','FontSize',%d)",fontsize) << endl;
    out << "grid on" << endl;
    write_set_fonts(out,fontsize);
    
    out.close();
  }
  
  ///
  /// SECTION ROC Plot
  ///
  
  ROC_Plot::ROC_Plot(string save_file, SelectScoresFn scoresFn) : 
    MatplotlibPlot(save_file), scoresFn(scoresFn)
  {
  };
    
  void ROC_Plot::add_plot(PerformanceRecord record, string label)
  {
    records[label] = record;
  }
  
  void ROC_Plot::add_point(string title, double p, double r)
  {
    points[title] = Vec2d(p,r);
  }
    
  ROC_Plot::~ROC_Plot()
  {
    header();
    put("from pyroc import *");
    
    for(auto & record : records)
    {
      // 
      vector<FrameDet> dets = scoresFn(record.second).getDetections();
      
      // generate the curve data
      ostringstream oss;
      oss << "data = [";
      for(int iter = 0; iter < dets.size(); ++iter)
      {
	oss << printfpp("(%f,%f)",(double)dets[iter].correct(),(double)dets[iter].score());
	if(iter != dets.size() - 1)
	  oss << ",";
      }
      oss << "]";
      string data_line = oss.str();
      put(data_line);
      put("roc = ROCData(data)");
      put("roc.plot(title='ROC Curve')");
      
      // generate the point data
      for(auto & pr_pt : points)
      {
	put(printfpp("plt.plot([%f],[%f], 'o', label = '%s')",
	    pr_pt.second[0],pr_pt.second[1],pr_pt.first.c_str()));
      }
    }
    
    footer();
  };
  
  ///
  /// SECTION: FingerErrorCumulativeDist
  /// 

  FingerErrorCumulativeDist::FingerErrorCumulativeDist(string save_file) : 
    MatplotlibPlot(save_file)
  {
  };
    
  void FingerErrorCumulativeDist::add_error_plot(vector<double> finger_errors, string label)
  {
    this->finger_errors[label] = finger_errors;
  }
    
  FingerErrorCumulativeDist::~FingerErrorCumulativeDist()
  {
    // generate the script to plot it    
    header();
    for(auto && error_line : finger_errors)
    {
      plot_script << "errors = array(" << error_line.second << ")" << endl;
      put("h = cumfreq(errors, 25)");
      put(printfpp("plt.plot(h[0]/h[0].max(), label = '%s')",error_line.first.c_str()));
    }
    put("plt.xlabel('error in cm')");
    put("plt.ylabel('ratio of errors')");    
    footer();
  }
    
  ///
  /// SECTION: PerformanceRecord
  /// 
    
  void PerformanceRecord::merge(PerformanceRecord&& other)
  {
    merge((PerformanceRecord&)other);
  }

  void PerformanceRecord::merge(PerformanceRecord& other)
  {
    hand_scores.merge(other.hand_scores);
    finger_scores.merge(other.finger_scores);
    joint_pose_scores = Scores(vector<Scores>{joint_pose_scores,other.joint_pose_scores});
    finger_errors_cm.insert(finger_errors_cm.end(),
			    other.finger_errors_cm.begin(),
			    other.finger_errors_cm.end());
    gt_count += other.gt_count;
    agnostic_finger_scores.merge(other.agnostic_finger_scores);
  }  
  
  PerformanceRecord::PerformanceRecord() : 
    gt_count(0),
    agnostic_finger_scores([](RectAtDepth gt, RectAtDepth Det, const string&frame)
    {
      return gt.dist(Det) < params::finger_correct_distance_threshold();
    }),
    finger_scores([](RectAtDepth gt, RectAtDepth Det, const string&frame)
    {
      return gt.dist(Det) < params::finger_correct_distance_threshold();
    }),    
    hand_scores([](RectAtDepth gt, RectAtDepth det, const string&frame)
    {
      return rectIntersect(gt,det) > .25;
    })
  {
  }

  PerformanceRecord::PerformanceRecord(
    DetectorScores hand_scores, 
    DetectorScores finger_scores, 
    Scores joint_pose_scores,
    DetectorScores agnostic_finger_scores,
    vector< double > finger_errors_cm, 
    double gt_count) : 
    hand_scores(hand_scores),
    finger_scores(finger_scores),
    joint_pose_scores(joint_pose_scores),
    agnostic_finger_scores(agnostic_finger_scores),
    finger_errors_cm(finger_errors_cm),
    gt_count(gt_count)
  {
  }
  
  DetectorScores& PerformanceRecord::getAgnosticFingerScores()
  {
    return agnostic_finger_scores;
  }
  
  vector< double >& PerformanceRecord::getFingerErrorsCM()
  {
    return finger_errors_cm;
  }

  DetectorScores& PerformanceRecord::getFingerScores()
  {
    return finger_scores;
  }

  DetectorScores& PerformanceRecord::getHandScores()
  {
    return hand_scores;
  }

  double&PerformanceRecord::getGtCount() 
  {
    return gt_count;
  }

  Scores& PerformanceRecord::getJointPoseScores()
  {
    return joint_pose_scores;
  }
  
  ///
  /// SECTION: Comparative Video
  /// 
  class ONI_Video;
  
#ifdef DD_ENABLE_OPENNI
  static int sample_rate = params::video_annotation_stride();   
  static constexpr double rect_thickness = 5; 
  
  typedef function<void (Vec3b&)> ColorFn;
  
  Mat draw_exemplar_overlay(
    Mat&exemplar,Mat&background,
    BaselineDetection&det, ColorFn colorFn, Mat segmentation)
  {
    if(det.bb.size().area() < 1)
      return image_text("draw_exemplar_overlay: det.bb.size().area() < 1");
    
    cv::resize(exemplar,exemplar,det.bb.size());
    if(!segmentation.empty())
      cv::resize(segmentation,segmentation,det.bb.size());
      
    Rect bb = det.bb;
    for(int yIter = 0; yIter < exemplar.rows; yIter++)
      for(int xIter = 0; xIter < exemplar.cols; xIter++)
	if(Rect(Point(0,0),background.size()).contains(Point(xIter+bb.x,yIter+bb.y)))
	{
	  Vec3b exemplar_pixel = exemplar.at<Vec3b>(yIter,xIter);
	  Vec3b&background_pixel = background.at<Vec3b>(yIter+bb.y,xIter+bb.x);
	  if(segmentation.empty() && ((exemplar_pixel != Vec3b(255,0,0))) || 
	    ((bool)(!segmentation.empty()) && (bool)(segmentation.at<uchar>(yIter,xIter) > 100)))
	  {
	    colorFn(background_pixel);
	  }
	}    

    return background;
  }
  
  void colorFn_ours(Vec3b&vis_pixel)
  {
    vis_pixel[1] = std::min<int>(100,vis_pixel[1]);
    vis_pixel[2] = std::min<int>(100,vis_pixel[2]);
    vis_pixel[0] = std::max<int>(255,vis_pixel[0]);
  }

  Mat generate_sub_frames_my_method(
    int iter,
    const Mat&rgb,vector< BaselineDetection > &our_track)
  {
    Mat vis_mine = rgb.clone();
    if(our_track[iter].bb != Rect() && 
       our_track[iter].bb.size().area() > 0 &&
       iter < our_track.size())
    {
      rectangle(vis_mine,our_track[iter].bb.tl(),our_track[iter].bb.br(),Scalar(255,255,0),rect_thickness);
      int ex_number = sample_rate*((iter+sample_rate/2)/sample_rate);
      string ex_filename = boost::regex_replace(
	g_params.require("OURS"),boost::regex("\\.DD\\.yml$"),printfpp("_%d.png",ex_number));
      log_once(printfpp("ex_filename = %s",ex_filename.c_str()));
      log_file << "BB: " << our_track[iter].bb << endl;
      Mat exemplar = imread(ex_filename);
      if(exemplar.empty())
      {
	vis_mine = image_text(vis_mine,string("NA"),Vec3b(255,255,0));
	resize(vis_mine,vis_mine,Size(320,240));
	return vis_mine;
      }
      draw_exemplar_overlay(exemplar,vis_mine,our_track[iter],colorFn_ours);
    }
    cv::resize(vis_mine,vis_mine,Size(320,240));
    return vis_mine;
  }
  
  Mat generate_sub_frames_gregs(int iter, const Mat&depth,const Mat&rgb)
  {
    if(!g_params.has_key("PERSON"))
    {
      Mat vis_greg = image_text("NA");
      cv::resize(vis_greg,vis_greg,Size(320,240));    
      return vis_greg;
    }
    
    string filename =
      printfpp("/home/grogez/Results-White-Bgd/%s_fr%05d-hand-WhiteBgd.png",
		g_params.require("PERSON").c_str(),
		iter);
    log_once(printfpp("GregFile = %s",filename.c_str()));
    Mat vis_greg = imread(filename);
    if(vis_greg.size().area() == 0)
    {
      log_once("Warning: Couldn't load Gregs file %s",filename.c_str());
      vis_greg = rgb.clone();
    }
    cv::resize(vis_greg,vis_greg,Size(320,240),0,0,params::DEPTH_INTER_STRATEGY); 
    //return vis_greg;
    return replace_matches(vis_greg,Vec3b(255,255,255),depth);
  }
  
  Mat generate_sub_frames_other(
    const Mat&rgb,
    int iter,
    vector< BaselineDetection >&pxc_track,
    vector< BaselineDetection >&nite2_track
 			      )
  {
    static mutex m; lock_guard<mutex> cs(m);
    static LibHandSynthesizer synth;
    synth.set_filter(
      LibHandSynthesizer::FILTER_BAD_BB |
      LibHandSynthesizer::FILTER_BAD_REGISTRATION);
    synth.set_synth_bg(false);
    //synth.set_render_armless(false);
    Mat vis_other = rgb.clone();
    
    // cache for the synthetic exemplars
    struct ExSeg
    {
      Mat exemplar, segmentation;
    };
    static Cache<ExSeg> exemplar_cache;
    
    auto vis_one_other = [&](string key, vector<BaselineDetection>&track,Scalar color,ColorFn colorFn)
    {
      if(track[iter].bb != Rect())
      {
	rectangle(vis_other,track[iter].bb.tl(),track[iter].bb.br(),color,rect_thickness);
	if(track[iter].pose_reg_point == nullptr)
	  return;
	//track[iter].pose_reg_point->cam_spec.r *= 2;
	const ExSeg& drawUs = exemplar_cache.get(printfpp("%s%d",key.c_str(),iter/params::video_annotation_stride()),[&]()
	{
	  log_once(printfpp("generate_sub_frames_other joints = %d",
			    track[iter].pose_reg_point->hand_pose.num_joints()));
	  synth.set_model(*track[iter].pose_reg_point,Size(320,240));
	  shared_ptr< LibHandMetadata > ex = synth.synth_one(true);
	  Rect roi = clamp(synth.getRenderedRGB(), ex->get_positives()["HandBB"]);
	  cout << "roi is " << roi << endl;
	  Mat cropEx = synth.getRenderedRGB()(roi);
	  Mat segCrop = synth.getSegmentation()(roi);
	  log_im("segCrop",segCrop);	
	  return ExSeg{cropEx,segCrop};
	});
	Mat ex = drawUs.exemplar.clone(), seg = drawUs.segmentation.clone();
	draw_exemplar_overlay(ex,vis_other,track[iter],colorFn,seg);
      }
    };
    
    // show the NiTE2 result
    vis_one_other("nite2",nite2_track,Scalar(0,255,0),[](Vec3b&vis_pixel)
      {
	vis_pixel[0] = std::min<int>(100,vis_pixel[0]);	
	vis_pixel[1] = std::max<int>(255,vis_pixel[1]);
	vis_pixel[2] = std::min<int>(100,vis_pixel[2]);
      });    
    
    // show the PXC result
    vis_one_other("pxc",pxc_track,Scalar(0,0,255),[](Vec3b&vis_pixel)
      {
	vis_pixel[0] = std::min<int>(100,vis_pixel[0]);	
	vis_pixel[1] = std::min<int>(100,vis_pixel[1]);
	vis_pixel[2] = std::max<int>(255,vis_pixel[2]);
      });
    
    cv::resize(vis_other,vis_other,Size(320,240));  
    return vis_other;
  }
  
  typedef function<void (Mat&)> BlurFn;
  
  vector<Mat> generate_sub_frames(
    int iter,
    ONI_Video&video,
    vector< BaselineDetection > &our_track,
    vector< BaselineDetection > &pxc_track,
    vector< BaselineDetection > &nite2_track,
    VideoCapture&forth_video,
    BlurFn blurFn
  )
  {
    int start_at = fromString<int>(g_params.require("START_FRAME"));
    
    // get the frame
    shared_ptr<MetaData_YML_Backed> frame = video.getFrame(iter,true);
    shared_ptr<ImRGBZ> im = frame->load_im();
    Mat rgb = im->RGB;
    blurFn(rgb);
    Mat depth = imageeq("",im->Z,false,false);
    
    // get my method
    Mat vis_mine = generate_sub_frames_my_method(iter,depth,our_track);
    
    // get greg's method
    Mat vis_greg = generate_sub_frames_gregs(iter,depth,rgb);
      
    // get FORTH
    Mat vis_forth;
    forth_video.set(CV_CAP_PROP_POS_FRAMES,iter - start_at);
    forth_video.read(vis_forth);
    if(vis_forth.size().area() == 0)
    {
      vis_forth = image_text("NA");
      cv::resize(vis_forth,vis_forth,Size(320,240));    
    }
    cv::resize(vis_forth,vis_forth,Size(320,240));
    blurFn(vis_forth);
    
    // get PXC and NiTE2
    Mat vis_other = generate_sub_frames_other(rgb,iter,pxc_track,nite2_track);
    
    // combine
    vector<Mat> sub_frames{vis_forth,vis_mine,vis_other,vis_greg};
    
    return sub_frames;
  }
#endif

  static Mat grab_frame(string dir,string re_str,string sub_vis)
  {
    log_file << "re_str = " << re_str << endl;
    boost::regex re(re_str);
    vector<string> matches = find_files(dir + "/",re,true);	
    if(matches.size() > 0)
    {
      log_file << "selected: " << matches.at(0) << endl;	  
      Visualization viz(matches.at(0));
      Mat im = viz.at(sub_vis);
      if(!im.empty())
      {
	im = imVGA(im);
	image_safe("frame",im);
	log_file << "got : " << re_str << endl;
	return imVGA(im);
      }
    }

    log_file << "didnt get : " << re_str << endl;
    return Mat();
  }

  static Mat grab_frame_ours(int frame_iter,int latent_stide)
  {
    string re_str = safe_printf(".*/exiv2video_dets_%_%_.*-.*",
				"NYU_Hands",frame_iter/latent_stide);
    return grab_frame("/home/jsupanci/workspace/deformable_depth/out/",
		      re_str,"Xmp.DD.param_vizvis_adj");
  }

  static Mat grab_frame_NYU(int frame_iter,int latent_stide)
  {
    string re_str = safe_printf(".*/exiv2video_dets_%_%_.*-.*",
				"NYU_Hands",frame_iter/latent_stide);
    return grab_frame(params::out_dir(),
		      re_str,"Xmp.DD.param_vizNYU_Model_visualize_result");
  }

  // make -j16 && gdb ./deformable_depth -ex "r comparative_video CFG_FILE=scripts/hand.cfg VIDEO_NAME=0.NYU_HANDS GRAB_WITH=NYU_Hands"
  void comparative_video_assemble()
  {
    shared_ptr<Video> video = load_video(g_params.require("VIDEO_NAME"));
    assert(video);    

    string vid_file = params::out_dir() + "/out.avi";
    log_once(string("vid_file = ") + vid_file);
    Size frame_size(2*640,480);
    VideoWriter video_out(vid_file,
			  CV_FOURCC('F','M','P','4'),15,frame_size,true);    
    auto writeFrame = [&](const Mat&frame)
    {
      Mat frame_out; cv::resize(frame,frame_out,frame_size);
      video_out.write(frame_out);
    };

    for(int frame_iter = 0; frame_iter < video->getNumberOfFrames(); ++frame_iter)
    {     
      // try to find the file
      bool generated = false;
      int LATENT_STRIDE = 1;
      if(frame_iter % LATENT_STRIDE == 0)
      {
	Mat NYU_frame = grab_frame_NYU(frame_iter,LATENT_STRIDE);
	Mat OUR_frame = grab_frame_ours(frame_iter,LATENT_STRIDE);

	if(!NYU_frame.empty() && !OUR_frame.empty())
	{
	  Mat frame = horizCat(NYU_frame,OUR_frame);
	  //log_im(safe_printf("frame%",frame_iter),frame);
	  for(int jter = 0; jter < 1; ++jter)	  
	    writeFrame(frame);
	  log_file << "wrote frame generated " << frame_iter << endl;
	  generated = true;
	}
      }
      
      if(not generated)
      {
	shared_ptr<MetaData_YML_Backed> datum = video->getFrame(frame_iter,true);    
	if(datum)
	{
	  auto im = datum->load_im();
	  Mat frame = imageeq("",imVGA(im->Z),false,false);
	  frame = monochrome(frame,BLUE);
	  writeFrame(horizCat(frame,frame));
	  log_file << "wrote frame default " << frame_iter << endl;
	}
	writeFrame(imVGA(image_text("N/A")));
      }
    }

    video_out.release();
  }

  void comparative_video()
  {
    comparative_video_assemble();
  }
  
  // command format
  // rm -v ./out/*; make -j16 && export VIDEO=greg; gdb ./deformable_depth -ex "r comparative_video ONI=data/depth_video/$VIDEO.oni OURS=~/Dropbox/out/2014.02.14-deva_feature_2/$VIDEO.oni.DD.yml PXC=data/depth_video/$VIDEO.oni.PXC.yml NiTE2=data/depth_video/$VIDEO.oni.Nite2.yml FORTH=data/depth_video/$VIDEO.oni.FORTH3D_THEIRS.avi FORTH_TRACK=data/depth_video/$VIDEO.oni.FORTH3D.yml START_FRAME=0"
  void comparative_video_track()
  {    
#ifdef DD_ENABLE_OPENNI
    // 2x2 video
    string video_filename = g_params.require("ONI");
    ONI_Video video(video_filename);
    
    // load the baseline tracks
    vector< BaselineDetection > our_track   = loadBaseline(g_params.require("OURS"));
    dilate(our_track,10);//interpolate(our_track);
    vector< BaselineDetection > pxc_track   = loadBaseline(g_params.require("PXC"));
    interpolate_ik_regress_full_hand_pose(pxc_track);
    vector< BaselineDetection > nite2_track = loadBaseline(g_params.require("NiTE2"));
    interpolate_ik_regress_full_hand_pose(nite2_track);
    vector< BaselineDetection > forth_track = loadBaseline(g_params.require("FORTH_TRACK"));
    
    Size outSize(4*320,4*240);
    string video_file = g_params.has_key("VIDOUT")?g_params.get_value("VIDOUT"):params::out_dir() + "/video.avi";
    VideoWriter video_out(video_file,
			  CV_FOURCC('F','M','P','4'),15,outSize,true);
    VideoCapture forth_video(g_params.require("FORTH"));
    
    vector<Rect> faces;
    int start_at = fromString<int>(g_params.require("START_FRAME"));
    for(int iter = start_at; 
	iter < video.getNumberOfFrames(); ++iter)
    { 
      // I believe this is done merely to make the videos smaller/faster?
      if(iter % 4 != 0)
      {
	cout << "Skipping Frame: " << iter << endl;
	continue;
      }
      
      int dup_times = (our_track[iter].bb != Rect())?3:1;
      for(int jter = 0; jter < dup_times; ++jter)
      {
	// blur the face      
	BlurFn blurFn = [](Mat&){};
	if(true)
	{
	  shared_ptr<MetaData_YML_Backed> frame = video.getFrame(iter,true);
	  shared_ptr<ImRGBZ> im = frame->load_im();      
	  vector<Rect> new_faces = SimpleFaceDetector().detect(*im);
	  if(!new_faces.empty())
	    faces = new_faces;
	  
	  blurFn = [faces](Mat&im)
	  {
	    if(faces.size() > 0)
	      for(Rect face : faces)
		cv::GaussianBlur(im(face),im(face),Size(9,9),5);	    
	  };
	}	
	
	// generate the sub-frames
	vector<Mat> sub_frames = generate_sub_frames(
	  iter,video,our_track,pxc_track,nite2_track,forth_video,blurFn);
	if(sub_frames.empty())
	{
	  cout << "Subframes empty: " << iter << endl;
	  goto DONE;
	}
		  
	sub_frames[0] = vertCat(
	  image_text(printfpp("FORTH3D",forth_track[iter].respString().c_str()),Vec3b(255,0,0),Vec3b(255,255,255)),sub_frames[0]);
	sub_frames[1] = vertCat(
	  image_text(printfpp("OURS",our_track[iter].respString().c_str()),Vec3b(255,255,0)),sub_frames[1]);
	sub_frames[2] = vertCat(horizCat(
	  image_text(printfpp("PXC ",pxc_track[iter].respString().c_str()),Vec3b(0,0,255)),
	  image_text(printfpp(" NiTE2 ",nite2_track[iter].respString().c_str()),Vec3b(0,255,0))),sub_frames[2]);
	sub_frames[3] = vertCat(
	  image_text(printfpp("RC"),Vec3b(255,0,255)),sub_frames[3]);	
	  
	//for(Mat&sub_frame : sub_frames)
	  //cv::resize(sub_frame,sub_frame,Size(320,240));
	Mat out_frame = tileCat(
	  vector<Mat>{sub_frames[1],sub_frames[0],sub_frames[2],sub_frames[3]},false);
	log_im("out_frame",out_frame);
	resize(out_frame,out_frame,outSize);
	video_out.write(out_frame);
      }
    }
    DONE:
    
    video_out.release();
#else
    throw std::runtime_error("unimplemented");
#endif
  }
  
  ///
  /// SECTION: Bootstrap PR Curve
  ///
  PerformanceRecord load_experiment(string expr_file)
  {
    PerformanceRecord record;
    
    RectAtDepth correct(Rect_<double>(42,42,42,42));
    RectAtDepth incorrect(Rect_<double>(4,4,4,4));    
    
    log_file << "loading experiment: " << expr_file << endl;
    ifstream ifs(expr_file);
    assert(ifs.is_open());
    
    while(!ifs.eof())
    {
      // read each line
      string header_id; getline(ifs,header_id);
      log_file << "header_id: " << header_id;
      string is_correct;  getline(ifs,is_correct); boost::algorithm::trim(is_correct);
      string str_resp; getline(ifs,str_resp); 
      double resp = fromString<double>(str_resp);
      log_file << "parsed resp: " << resp << endl;
      
      record.getHandScores().put_ground_truth(header_id,correct);
      if(is_correct == "0")
      {
	log_file << "Not Correct" << endl;
	record.getHandScores().put_detection(header_id,incorrect,resp);
      }
      else
      {
	log_file << "Correct" << endl;
	record.getHandScores().put_detection(header_id,correct,resp);
      }
    }
    
    return record;
  }
  
  PerformanceRecord load_experiment(int id)
  {
    PerformanceRecord record;
    
    string experiment_dir("/home/jsupanci/workspace/deformable_depth/data/PR_Updates/");
    vector<string> expr_files = find_files(experiment_dir, 
	boost::regex(printfpp(".*%d.*",toString<int>(id).c_str())));
        
    for(string expr_file : expr_files)
    {
      record.merge(load_experiment(expr_file));
    }
    
    return record;    
  }
  
  PerformanceRecord load_experiment(vector<string> files)
  {
    PerformanceRecord record;
    for(string file : files)
    {
      record.merge(load_experiment(file));
    }
    return record;
  }
  
  void bootstrap_pr_curve()
  {
    //  - 1st graph is Exp. 16,  13  and 1 
    // ('Task-Specific' 'Egocentric-Viewpoint' and 'Entire-Space')
    //PrecisionRecallPlot pr_plot(params::out_dir() + "/gen_1st_graph.m","hand");
    //pr_plot.add_plot(load_experiment("16"),"Task-Specific");
    //pr_plot.add_plot(load_experiment("13"),"Egocentric-Viewpoint");
    //pr_plot.add_plot(load_experiment("1"),"Entire-Space");
    
    //  -2nd graph is Exp.  18, 16 and  17 ('Depth + RGB' 'Depth'  'RGB');
    //PrecisionRecallPlot pr_plot(params::out_dir() + "/gen_2nd_graph.m","hand");
    //pr_plot.add_plot(load_experiment("18"),"Depth and RGB");
    //pr_plot.add_plot(load_experiment("16"),"Depth");
    //pr_plot.add_plot(load_experiment("17"),"RGB");   
    
    // only one graph with 2 curves:  NoObj vs Obj  
    // (Both correspond to the experiment Task-Specific-Exp18 )
    //PrecisionRecallPlot pr_plot(params::out_dir() + "/gen_obj_graph.m","hand");
    //pr_plot.add_plot(load_experiment("-Obj"),"Object");
    //pr_plot.add_plot(load_experiment("-NoObj"),"No Object");     
    
    // ECCV 2014 Curve
    string prefix("/home/jsupanci/workspace/deformable_depth/data/PR_Updates/");
    PrecisionRecallPlot pr_plot(params::out_dir() + "/gen_curve148v150v151.m","hand");
    pr_plot.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Marga-2014-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Greg-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Marga-Egocentric prior-Exp148.txt"})),"Exp148");
    pr_plot.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-Generic prior-Exp150.txt",
       prefix+"PR-Cascades-Marga-2014-Generic prior-Exp150.txt",
       prefix+"PR-Cascades-Greg-Generic prior-Exp150.txt",
       prefix+"PR-Cascades-Marga-Generic prior-Exp150.txt"})),"Exp150");
    pr_plot.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-Viewpoint prior-Exp151.txt",
       prefix+"PR-Cascades-Marga-2014-Viewpoint prior-Exp151.txt",
       prefix+"PR-Cascades-Greg-Viewpoint prior-Exp151.txt",
       prefix+"PR-Cascades-Marga-Viewpoint prior-Exp151.txt"})),"Exp151");    
    
    // second plot
    PrecisionRecallPlot pr_plot2(params::out_dir() + "/gen_curve144v148.m","hand");
    pr_plot2.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-Egocentric   Object prior-Exp144.txt",
       prefix+"PR-Cascades-Marga-2014-Egocentric   Object prior-Exp144.txt",
       prefix+"PR-Cascades-Greg-Egocentric   Object prior-Exp144.txt",
       prefix+"PR-Cascades-Marga-Egocentric   Object prior-Exp144.txt"})),"Exp144");
    pr_plot2.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Marga-2014-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Greg-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Marga-Egocentric prior-Exp148.txt"})),"Exp148");
    
    // third plot
    PrecisionRecallPlot pr_plot3(params::out_dir() + "/gen_curve125v148_2014only.m","hand");
    pr_plot3.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-User-specific prior-Exp125.txt",
       prefix+"PR-Cascades-Marga-2014-User-specific prior-Exp125.txt"})),"Exp125");
    pr_plot3.add_plot(load_experiment(vector<string>(
      {prefix+"PR-Cascades-Greg-2014-Egocentric prior-Exp148.txt",
       prefix+"PR-Cascades-Marga-2014-Egocentric prior-Exp148.txt"})),"Exp148");
  }
}

 
