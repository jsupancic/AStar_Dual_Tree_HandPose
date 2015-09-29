/**
 * Copyright 2014: James Steven Supancic III
 **/

#define use_speed_ 0

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <GL/glut.h>
#include <memory>
#include <stdio.h>
#include "HandSynth.hpp"
#include "Renderer.hpp"
#include <boost/filesystem.hpp>
#include <boost/graph/graph_concepts.hpp>
#include "params.hpp"
#include "util.hpp"
#include <limits>
#include <thread>
#include "Pipe.hpp"
#include "params.hpp"
#include "Capturer.hpp"
#include <ml.h>
#include "time.h"
#include <opencv2/objdetect/objdetect.hpp>
#include "PXCSupport.hpp"
#include "Detector.hpp"
#include "GlobalMixture.hpp"
#include "omp.h"
#include "Segment.hpp"
#include "MetaData.hpp"
#include "OneFeatureModel.hpp"
#include "FauxLearner.hpp"
#include <time.h>
#include "Eval.hpp"
#include "Log.hpp"
#include "Orthography.hpp"
#include "Kahan.hpp"
#include "MetaFeatures.hpp"
#include "ThreadPool.hpp"
#include "Tests.hpp"
#include "Analyze.hpp"
#include "LibHandSynth.hpp"
#include "Poselet.hpp"
#include "main.hpp"
#include "BackgroundThreads.hpp"
#include "Annotation.hpp"
#include "Video.hpp"
#include "Cluster.hpp"
#include "Skin.hpp"
#include "CyberGlove.hpp"
#include "KTHGrasp.hpp"
#include "InverseKinematics.hpp"
#include "IKSynther.hpp"
#include "PCA_Pose.hpp"
#include "google/heap-profiler.h"
#include "MetaDataKITTI.hpp"
#include "KITTI_Eval.hpp"
#include "Export2Caffe.hpp"
#include "ICL_MetaData.hpp"
#include "Scripts.hpp"
#include "MDS.hpp"
#include "ShowPairwiseErrors.hpp"
#include "ManualPose.hpp"

namespace deformable_depth
{
  using cv::Mat;
  using cv::Mat_;
  using cv::VideoCapture;
  using cv::waitKey;
  using cv::imshow;
  using cv::minMaxIdx;
  using cv::Canny;
  using cv::mean;
  using cv::Scalar;
  using cv::imshow;
  using cv::imread;
  using cv::MatND;
  using cv::calcHist;
  using cv::threshold;
  using cv::normalize;
  using cv::FileStorage;
  
  using std::shared_ptr;
  using std::cout;
  using std::endl;
  using std::numeric_limits;
  using std::string;
  
  using boost::filesystem::canonical;
  using boost::filesystem::path;
  using boost::filesystem::directory_iterator;
  
  shared_ptr<Renderer> tracker;
   
  /// SECTION: Debug
  void debug_iteration()
  {
    // capture
    static Capturer capturer;
    Mat Zcap,RGBcap; 
    capturer.doIteration(Zcap,RGBcap);
    
    // render
    static Renderer renderer;
    renderer.getHand().animate();
    renderer.getHand().setPos(0,0,-80);
    Mat Zrnd, RGBrnd;
    renderer.doIteration(Zrnd,RGBrnd);
    
    Mat Zmerg, RGBmerg;
    float v1,v2;
    merge(RGBrnd,Zrnd,RGBcap,Zcap,RGBmerg,Zmerg,v1,v2);
    imshow("Synthetic Hand - RGB",RGBmerg);
    imagesc("Synthetic Hand - Depth",Zmerg);
    cvWaitKey(1);
    
    // proc
    // extractHOSR(Zcap,25, 25);
    
    // release
    //imagesc("Captured Depth",log(Zcap));
    //cvWaitKey(1);
  }
  
  void debug(int argc, char**argv)
  {
    init_glut(argc, argv, debug_iteration);
    glutMainLoop();    
  }
    
  /// SECTION: Capture negative data
  void capture_data()
  {
    Capturer capturer;
    char filename[100];
    char filenameRGB[100];
    int keyCode = -1;
    string save_dir = "in";
    
    do
    {
      // capture
      Mat Zcap,RGBcap; 
      capturer.doIteration(Zcap,RGBcap);
      
      // show and save
      imagesc("captured depth",log(Zcap));
      imshow("captured rgb",RGBcap);
      
      if(keyCode == 's')
      {
	// find a file named 
	for(int iter = 0; true; iter++)
	{
	  snprintf(filenameRGB,100,(save_dir + "/%d.png").c_str(),iter);
	  snprintf(filename,100,(save_dir + "/%d.yml").c_str(),iter);
	  if(boost::filesystem::exists(filenameRGB) || boost::filesystem::exists(filename))
	    printf("Skiping used filename %s\n",filename);
	  else
	    break;
	}
            
	// save depth
	FileStorage bg_file(filename, FileStorage::WRITE);
	bg_file << "bg_depth" << Zcap;
	bg_file.release();
      
	// save the rgb
	cv::imwrite(filenameRGB,RGBcap);
      }
      keyCode = cvWaitKey(0);
    } while(keyCode != 'q'); 
  }
 
  /// SECTION: Show
  /// Show a data file.
  Mat show_one(MetaData&metadata,Scalar bb_color)
  {
    // load the file
    shared_ptr<ImRGBZ> im = metadata.load_im();
    
    // show the RGB and depth
    //Mat Zshow = imagesc("imagesc(Z)",im->Z);
    Mat Zshow = imageeq("imageeq(Z)",im->Z,false,false);
    
    // draw the labels...
    Mat RGB = im->RGB.clone();
    auto positives = metadata.get_positives();
    for(auto label : positives)
    {
      Scalar acting_color = bb_color;
      if(!label.second.visible)
      {
	acting_color = Scalar(0,0,255);
      }
      if(!g_params.has_key("HIDE_LABELS"))
      {
	      cv::rectangle(RGB,label.second.tl(),label.second.br(),acting_color);
	      cv::rectangle(Zshow,label.second.tl(),label.second.br(),acting_color);
      }
    }
    //image_safe("RGB - Raw",metadata.load_raw_RGB());
    log_im("RGB",image_safe("RGB",RGB));
    image_safe("imageeq(Z)",Zshow);
    
    // report the world area of the hand
    Rect_<double> handBB = positives["HandBB"];
    if(0 <= handBB.x && 0 <= handBB.width && 
      handBB.x + handBB.width <= im->Z.cols && 
      0 <= handBB.y && 0 <= handBB.height && 
      handBB.y + handBB.height <= im->Z.rows)
    {
      double z = extrema(im->Z(handBB)).min;
      double world_area = im->camera.worldAreaForImageArea(z,handBB);
      cout << "world area: " << world_area << endl;
    }
        
    // report leftp
    cout << "leftP: " << metadata.leftP() << endl;
    Extrema ex = extrema(im->Z);
    cout << "extrma: " << ex.min << " " << ex.max << endl;
    
    // Wait for a keypress
    waitKey_safe(0);    
    
    return Zshow;
  }
  
  void show(int argc, char**argv)
  {
    assert(argc >= 3);
    
    shared_ptr< MetaData > metadata = metadata_build(g_params.get_value("FILE"),true,false);
    assert(metadata);
    show_one(*metadata,Scalar(255,0,0));
  }
  
  void show_feats(MetaData&metadata)
  {
    // get the damn hand BB
    Rect_<double> handBB = metadata.get_positives()["HandBB"];
    shared_ptr<ImRGBZ> im = metadata.load_im();    
    
    // draw the HoG feature
    if(false)
    {
      int sbins = 4;
      assert(im->RGB.type() == DataType<Vec3b>::type);
      ImRGBZ hand_im = (*im)(handBB);
      assert(hand_im.RGB.type() == DataType<Vec3b>::type);
      shared_ptr<const ImRGBZ> im_crop = cropToCells(hand_im,sbins,sbins);
      auto hog_computer = COMBO_FACT_RGB_DEPTH().build(im_crop->RGB.size(),sbins);
      vector<float> feats; hog_computer->compute(*im_crop,feats);
      Mat feat_vis = hog_computer->show("feat_vis",vec_f2d(feats));
      image_safe("feat_vis",feat_vis,false);  
      image_safe("hand_im",metadata.load_raw_RGB()(handBB));    
    }
    
    // vis HoA
    if(false)
    {
      // scale the image for uniform hand size
      auto print_im = [&](const ImRGBZ&im)
      {
	cout << "***************************************" << endl;
	cout << "im cols: " << im.RGB.cols << endl;
	cout << "cam cols: " << im.camera.hRes() << endl;    
	cout << "fov: " << im.camera.hFov() << endl;      
	cout << "***************************************" << endl;
      };
      print_im(*im);
      double z = medianApx(im->Z,handBB,0);
      cout << "z = " << z << endl;

      // get a cardinal size
      double sf = std::sqrt(10.0/handBB.area());
      *im = im->resize(sf);
      print_im(*im);      
	
      // get a HoA computer
      HOGComputer_Factory<HOGComputer_Area> comp_fact;
      unique_ptr<DepthFeatComputer>comp(comp_fact.build(im->RGB.size(),1));
      // compute the image to a vector<dobule>
      auto im_crop = comp->cropToCells(*im);
      vector<float> im_feats;
      comp->compute(*im_crop,im_feats);
      print_im(*im_crop);
      // visualize the features
      vector<double> im_feats_double = vec_f2d(im_feats);
      Mat vis = comp->show("World Areas",im_feats_double);
      image_safe("World Areas",vis,false);
    }
  }
  
  /// SECTION: show_synthetic
  
  void show_dir()
  {
    string synth_dir = g_params.get_value("DIR");
    vector<string> stems = allStems(synth_dir, ".gz");

    for(string stem : stems)
    {
      string path = synth_dir+stem;
      cout << printfpp("show_synthetic: %s",path.c_str()) << endl;
      shared_ptr< MetaData > metadata = metadata_build(path, true, false);
      if(metadata)
      {
	show_feats(*metadata);
	show_one(*metadata);
      }
    }
  }
  
  /// SECTION Export RGB
  void export_rgb(int argc, char**argv)
  {
    // 0: progname
    // 1: export_rgb
    // 2: directory
    assert(argc > 3);
    string dir(argv[2]);
    string outdir(argv[3]);
    vector<shared_ptr<MetaData>> stored_examples = metadata_build_all(dir);
    cout << "loaded " << stored_examples.size() << " new examples" << endl;
    
    // export each example as a 
    for(shared_ptr<MetaData>&example : stored_examples)
    {
      // setup the filenames
      string stem = boost::filesystem::path(example->get_filename()).stem().string();
      string outname = outdir + stem + ".jpg";
      string labelName = outdir + stem + ".gt.txt";
      string posefName = outdir + stem + ".pose.txt";
      cout << "saving " << outname << " // " << labelName << endl;
      
      // save image
      imwrite(outname,example->load_im()->RGB);
      
      // save label
      Rect handBB = example->get_positives()["HandBB"];
      ofstream label_file;
      label_file.open(labelName);
      label_file << handBB.tl().x << " , " << handBB.tl().y << " , " <<
	handBB.br().x << " , " << handBB.br().y << endl;
      label_file.close();
      
      // save pose
      ofstream pose_file;
      pose_file.open(posefName);
      pose_file << example->get_pose_name() << endl;
      pose_file.close();
    }
  }
  
  /// SECTION: feature visualization
  void feat_vis(int argc, char**argv)
  {
    // locate the data and feature computor
    int sbins = 8;
    assert(argc >= 3);
    string filename(argv[2]);
    shared_ptr< MetaData > metadata = metadata_build(filename,true);
    shared_ptr<const ImRGBZ> im = cropToCells(*metadata->load_im(),sbins,sbins);
    std::unique_ptr<DepthFeatComputer> feat_comp(
      default_fact.build(Size(im->cols(),im->rows()),sbins));
    
    // compute the feature
    vector<float> feat;
    feat_comp->compute(*im,feat);
    
    // show the feature
    auto dfeat = vec_f2d(feat);
    image_safe("VGA feat of im",imVGA(feat_comp->show("feature of image",dfeat)));
    waitKey_safe(0);
  }
  
  /// SECTION: Main
  
  void usage()
  {
    cout << "usage: deformable_depth mode" << endl;
  }
    
  bool dispatch_libhand(string command)
  {
#ifdef DD_ENABLE_HAND_SYNTH 
    if(command == "cluster")
      cluster();
    else if(command == "export_cluster")
      export_cluster();
    else if(command == "gen_reg")
      gen_reg();
    else if(command == "test_regression")
      test_regression();
    else if(command == "visualize_synthetic")
      visualize_synthetic();
    else if(command == "cyberglove_reverse_engineer")
      cyberglove_reverse_engineer();
    else if(command == "comparative_video")
      comparative_video();
    else if(command == "kth_grasp_synth")
      kth_grasp_synth();
    else if(command == "greg_ik_synth")
      greg_ik_synth();    
    else if(command == "pca_pose_train")
      pca_pose_train();
    else if(command == "show_pairwise_errors")
      show_pairwise_errors();
    else if(command == "pose_hand")
      pose_hand();
    else
      return false;
    
    return true;
#else
    return false;
#endif
  }
    
  void dispatch(int argc, char**argv)
  {
    string command(argv[1]);
    if(command == "make_hist")
      skin_make_hist();
    else if(command == "gen_training_data")
      gen_training_data(argc, argv);
    else if(command == "segment")
      segment(argc, argv);
    else if(command == "debug")
      debug(argc, argv);
    else if(command == "eval_model")
      eval_model(argc, argv,params::model_builder());
    else if(command == "capture_data")
      capture_data();
    else if(command == "label")
      label(argc,argv);
    else if(command == "show")
      show(argc,argv);
    else if(command == "tune")
      tune_model(argc,argv);
    else if(command == "platt_test")
      platt_test();
    else if(command == "export_rgb")
      export_rgb(argc,argv);
    else if(command == "orthography")
      orthography(argc,argv);
    else if(command == "test_sparse")
      test_sparse();
    else if(command == "kahan_test")
      kahan_test();
    else if(command == "feat_vis")
      feat_vis(argc,argv);
    else if(command == "test")
      invoke_tests();
    else if(command == "analyze")
      analyze();
    else if(command == "export_responces")
      export_responces();
    else if(command == "analyze_video")
      analyze_video();
    else if(command == "analyze_anytime")
      analyze_anytime();
    else if(command == "analyze_egocentric")
      analyze_egocentric();
    else if(command == "regress_finger_conf")
      regress_finger_conf();
    else if(command == "show_dir")
      show_dir();
    else if(command == "show_frame")
      show_frame();
    else if(command == "show_video")
      show_video();
    else if(command == "poselet")
      poselet();
    else if(command == "convert_video")
      convert_video(argc,argv);
    else if(command == "bootstrap_pr_curve")
      bootstrap_pr_curve();
    else if(command == "show_baseline_on_video")
      show_baseline_on_video();
    else if(command == "KITTI_Demo")
      KITTI_Demo();
    else if(command == "kitti_main")
      kitti_main(argc,argv);
    else if(command == "export_2_caffe")
      export_2_caffe();
    else if(command == "export_depth_exr")
      export_depth_exr();
    else if(command == "imageeq")
      do_imageeq();
    else if(command == "convert_all_videos")
      convert_all_videos();
    else if(command == "show_ICL_training_set")
      show_ICL_training_set();
    else if(command == "script")
      dispatch_scripts();
    else if(command == "export_mds_data")
      export_mds_data();
    else if(!dispatch_libhand(command))
    {
      usage();
    }    
  }
  
  int abort_error_handler(
    int status, 
    char const* func_name, 
    char const* err_msg, 
    char const* file_name, 
    int line, 
    void*) 
  {
    ostringstream message;
    message << "ERROR: in " << func_name << "(" << 
      file_name << ":" << line << ") Message: " << err_msg;
    cout << message.str() << endl;
    log_file << message.str() << endl;
    log_file.flush();
    cout.flush();
    std::terminate();
  }
  
  int main(int argc, char**argv)
  {    
    // random PRNG
    srand ( time(NULL) );
    
    // configure OpenCV
    cvRedirectError(abort_error_handler);
    
    // make sure we have a command
    g_params.parse(argc,argv);
    if(argc <= 1)
    {
      usage();
      return 1;
    }
    string command(argv[1]);    

    // start the heap profiler
    if(g_params.has_key("HEAP_PROF"))
    {
      string prof_file = params::out_dir() + "/heap_prof.dat";
      log_once(printfpp("heap_prof_file = %s",prof_file.c_str()));
      HeapProfilerStart(prof_file.c_str());    
    }
    else
      HeapProfilerStop();
   
    // open log
    init_logs(command);
    
    // launch painter...
    //if(command != "label")
    //launch_background_repainter();
    
    // configure OpenMP
    int num_cpus = params::cpu_count();
    log_file << "NUM_CPUS = " << num_cpus << endl;
    active_worker_thread.set(num_cpus);
    omp_set_nested(true);
    omp_set_max_active_levels(2);
    
    // configure the new thread pool system which will replace OpenMP
    default_pool.reset(new ThreadPool(num_cpus-1));
    IO_Pool.reset(new ThreadPool(num_cpus));
    empty_pool.reset(new ThreadPool(0));
    
    // 
    dispatch(argc,argv);
    cout << "main complete\n" << endl;
    log_file << "main complete\n" << endl;
    
    // anykey to continue
    waitKey_safe(0);
    cout << "waiting for thread pool to shutdown" << endl;
    //default_pool->shutdown();
    cout << "thread pool has been shutdown" << endl;
    
    // close log
    log_file.close();
    
    // stop the heap profiler
    if(g_params.has_key("HEAP_PROF"))
      HeapProfilerStop();
    
    return 0;
  }
}

int main(int argc, char **argv)
{
  return deformable_depth::main(argc, argv);
}

