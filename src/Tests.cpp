/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Tests.hpp"
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include "params.hpp"
#include "MetaData.hpp"
#include "Eval.hpp"
#include "SubTrees.hpp"
#include "GlobalFeat.hpp"
#include "Video.hpp"
#include "Skin.hpp"
#include "Log.hpp"
#include "ConvexityDefects.hpp"
#include "Orthography.hpp"
#include "HornAbsOri.hpp"
#include "TestModel.hpp"
#include "ONI_Video.hpp"
#include "SphericalVolumes.hpp"
#include "SpiralmetricVolumes.hpp"

namespace deformable_depth
{
  using namespace cv;
  
  static void test_morphology()
  {
    // define the problem
    Mat Z(100,100,DataType<float>::type,Scalar::all(100));
    Z.at<float>(50,50) = 0;
    Rect bb0(Point(0,0),Size(11,11));
    display("pre",Z);
    
    // run the morphology
    Mat struct_elem = getStructuringElement(MORPH_RECT,bb0.size());
    Mat z_mins; erode(Z,z_mins,struct_elem);    
    
    // show the result
    z_mins.at<float>(50,50) = qnan;
    display("post",z_mins);
  }
  
  static void test_scores()
  {
    Scores scores;
    
    Rect gt(Point(10,10),Point(20,20));
    rectScore(gt,gt,-32,scores); 
    cout << scores.toString(-inf) << endl;
    rectScore(gt,Rect(),-32,scores); 
    cout << scores.toString(-inf) << endl;
    rectScore(gt,Rect(Point(0,0),Point(5,5)),-32,scores); // false positive
    cout << scores.toString(-inf) << endl;
  }
  
  static void test_SubTrees()
  {
    vector<shared_ptr<MetaData> > trainging_data = default_train_data();
    
    for(auto train_datum : trainging_data)
    {
      if(!train_datum->use_positives())
	continue;
      
      // load the datum
      shared_ptr<ImRGBZ> im = train_datum->load_im();
     
      // make the subtrees
      TreeStats subTrees = treeStatsOfImage_cached(*im);
      
      // get a test rectangle
      Rect query = getRect("give rect", imageeq("",im->Z,false,false));
      Rect match; 
      double best_ol = -inf;
      for(int xIter = query.tl().x; xIter < query.br().x; ++xIter)
	for(int yIter = query.tl().y; yIter < query.br().y; ++yIter)
	  for(Direction cur_dir : card_dirs())
	  {
	    Rect candidate = subTrees.pixelAreaDecorator.bbs[yIter][xIter][cur_dir];
	    double ol = rectIntersect(candidate,query);
	    if(ol > best_ol)
	    {
	      best_ol = ol;
	      match = candidate;
	    }
	  }
      //
      Mat vis = im->RGB.clone();
      rectangle(vis,query.tl(),query.br(),best_ol*Scalar(255,0,0)); // blue
      rectangle(vis,match.tl(),query.br(),Scalar(0,0,255));
      cout << "consistancy: " << best_ol << endl;
      cout << "entropy: " << colorConst(*im,query) << endl;
      image_safe("BBs",vis); waitKey_safe(0);
    }
  }

#ifdef DD_ENABLE_OPENNI  
  static void test_skin_detection()
  {
    //ONI_Video video("data/depth_video/greg.oni");
    //ONI_Video video("data/depth_video/dennis_test_video1.oni");
    //ONI_Video video("data/depth_video/sam.oni");
    //ONI_Video video("data/depth_video/test_data11.oni");
    ONI_Video video("data/depth_video/sequence1.oni");
    
    for(int iter = 0; iter < video.getNumberOfFrames(); ++iter)
      if(iter % params::video_annotation_stride() == 0)
      {
	// load the frame from the video
	shared_ptr<MetaData_YML_Backed> metadata = video.getFrame(iter,true);
	shared_ptr<ImRGBZ> im = metadata->load_im();	
	
	// run the skin detector on the RGB channel
	Mat fgProb, bgProb;
	Mat skinLikelihood = skin_detect(im->RGB,fgProb,bgProb);
	assert(skinLikelihood.type() == DataType<double>::type);
	Mat skin_detections(skinLikelihood.rows,skinLikelihood.cols,
			    DataType<Vec3b>::type,Scalar::all(0));
	for(int row = 0; row < skin_detections.rows; row++)
	  for(int col = 0; col < skin_detections.cols; col++)
	    if(skinLikelihood.at<double>(row,col) > .35)
	      skin_detections.at<Vec3b>(row,col) = Vec3b(255,255,255);
	    else
	      skin_detections.at<Vec3b>(row,col) = Vec3b(0,0,0);
	
	log_im("skinProbs", 
	       tileCat(vector<Mat>
	       {imagesc("",skinLikelihood),
		imageeq("SKIN PROBS",skinLikelihood),
		skin_detections
	      }));
	waitKey_safe(0);
      }
  }
#endif
  
  void test_convex_hull()
  {
    vector<shared_ptr<MetaData> > data = default_test_data();
    
    for(auto datum : data)
    {     
      // load the datum
      shared_ptr<ImRGBZ> im = datum->load_im();
      
      // get a test rectangle
      Rect query = getRect("give rect", imageeq("",im->Z,false,false));      
      BaselineDetection det;
      det.bb = query;
      
      // update
      BaselineDetection updated_det = FingerConvexityDefectsPoser().pose(det,*im);
      
      // visualize
      Mat showMe = im->RGB.clone();
      for(auto && part : updated_det.parts)
      {
	string part_name = part.first;
	auto orig_bb = det.parts[part_name].bb;
	auto new_bb  = updated_det.parts[part_name].bb;
	rectangle(showMe,orig_bb.tl(),orig_bb.br(),Scalar(255,0,0));
	rectangle(showMe,new_bb.tl(), new_bb.br(),Scalar(0,0,255));
      }
      waitKey_safe(0);
    }
  }
  
  void test_ortho_dist()
  {
    vector<shared_ptr<MetaData> > data = load_dirs(real_dirs());
    // Rect getRect(std::string winName, cv::Mat bg, cv::Rect init = Rect(), bool allow_null = false); 
    
    for(auto datum : data)
    {
      shared_ptr<ImRGBZ> im = datum->load_im();
      imageeq("Depth",im->Z,true,false);
      bool vis; 
      Point p1 = getPt("Depth",&vis);
      Point p2 = getPt("Depth",&vis);
      // get the Z offset
      double z1 = im->Z.at<float>(p1.y,p1.x);
      double z2 = im->Z.at<float>(p2.y,p2.x);
      cout << "p1: " << p1 << endl;
      cout << "p2: " << p2 << endl;
      
      Point2d o_p1 = map2ortho_cm(im->camera,p1, z1);
      Point2d o_p2 = map2ortho_cm(im->camera,p2, z2);
      Vec3d v1(o_p1.x,o_p1.y,z1), v2(o_p2.x,o_p2.y,z2);
      
      cout << "v1: " << v1 << endl;
      cout << "v2: " << v2 << endl;
      Vec3d disp = v1 - v2;
      cout << "dist = " << std::sqrt(disp.dot(disp)) << endl;
    }
  }
  
  void test_inverse_kinematics()
  {
#ifdef DD_ENABLE_HAND_SYNTH    
    vector<Vec3d> square1{{-1,-1,0},{1,-1,0},{1,1,0},{-1,1,0}};
    vector<Vec3d> square2{{1,-1,0},{1,1,0},{-1,1,0},{-1,-1,0}};
    for(Vec3d&v : square2)
    {
      v[0] = 2*v[0] + 3;
      v[1] = 2*v[1] + 3;
      v[2] = 2*v[2] + 3;
    }
    cout << "HornAO Dist = " << distHornAO(square1,square2).distance << endl;
#endif
  }
  
  void test_metric_feature_one(shared_ptr<MetaData> datum)
  {
    shared_ptr<ImRGBZ> im = datum->load_im();
    //Rect query = getRect("give rect", imageeq("",im->Z,false,false));
    Rect query = datum->get_positives()["HandBB"]; 
    if(query == Rect())
    {
      cout << "reject filename: " << datum->get_filename() << endl;
      log_file << "reject filename: " << datum->get_filename() << endl;
      cout << "query: " << query << endl;
      return;
    }
    
    // crop the query
    ImRGBZ imcrop = (*im)(query);
    ImRGBZ imtemplate = imcrop.resize(im->RGB.size());
    
    double max_area;
    Mat DID = HOGComputer_Area::DistInvarientDepths(imtemplate,max_area);
    
    imagesc("DID",DID, true, true);
    log_file << extrema(DID).min << endl;
    //waitKey_safe(0);    
    
    // print the BB's world area as well
    for(auto & part_name_bb : datum->get_positives())
    {
      Rect_<double> bb = part_name_bb.second;
      if(!(0 <= bb.x && 0 <= bb.width && bb.x + bb.width <= im->Z.cols 
	&& 0 <= bb.y && 0 <= bb.height && bb.y + bb.height <= im->Z.rows))
	continue;
      float depth = extrema(im->Z(bb)).min;
      cout << printfpp("GT World Area [%s]: ",part_name_bb.first.c_str()) << im->camera.worldAreaForImageArea(depth,bb) << endl;
      log_file << printfpp("GT World Area [%s]: ",part_name_bb.first.c_str()) << 
	im->camera.worldAreaForImageArea(depth,bb) << "/" << bb.area() << endl;
    }
  }
  
  void test_metric_feature_still()
  {
    vector<shared_ptr<MetaData> > data = load_dirs(
       vector<string>{params::synthetic_directory()},false);
    //vector<shared_ptr<MetaData> > data = load_dirs(default_train_dirs());
    //vector<shared_ptr<MetaData> > data = load_dirs(real_dirs(),false); 
    for(auto datum : data)    
    {
      test_metric_feature_one(datum);
    }    
  }
  
  void test_metric_feature_video()
  {
    for(string video_file : test_video_filenames())
    {
      shared_ptr<Video> video = load_video(video_file);
      for(int iter = 0; iter < video->getNumberOfFrames(); ++iter)
      {
	if(video->is_frame_annotated(iter))
	{
	  shared_ptr<MetaData> datum = video->getFrame(iter,true);
	  shared_ptr<ImRGBZ> im = datum->load_im();
	  //cout << "metric_correction = " << im->camera.metric_correction() << endl;
	  test_metric_feature_one(datum);
	}
      }
    }
  }
  
  void test_metric_feature()
  {
    //test_metric_feature_still();
    test_metric_feature_video();
  }
  
  void test_ortho_fb()
  { 
    float px,py, ox,oy;
    DepthCamera cam;
    Size size(320,240);
    float depth = 75;
    // test1
    map2ortho(cam,size,-215, -165.841, depth,ox, oy,true);
    cout << "ox = " << ox << " oy = " << oy << endl;
    mapFromOrtho(cam,size,
		ox, oy, depth,px, py);
    cout << "px = " << px << " py = " << py << endl;
    
    // test2
    mapFromOrtho(cam,size,
		0.000000,-1.958971, 64,px, py);    
    cout << "px = " << px << " py = " << py << endl;
    
    // test 3
    Rect_<double> pers_bb(21,32,43,32);
    cout << "expected: " << pers_bb << endl;
    cout << "actual: " << mapFromOrtho(cam,size,map2ortho(cam,size,pers_bb,depth),depth) << endl;
  }
  
  void test_vec()
  {
    vector<double> v1{.1, .3, .2, .5};
    vector<double> v2{.1, .2, .3, .5};
    cout << "spearman: " << spearman(v1,v2) << endl;
    cout << "H(0) = " << shannon_entropy(0) << endl;
    cout << "H(.5) = " << shannon_entropy(.5) << endl;
    cout << "H(1) = " << shannon_entropy(1) << endl;
  }
  
  void test_manifold_selection()
  {
    // load the frame from the video
    shared_ptr<Video> video = load_video("data/yml_videos/2.ICL.yml.gz");       
    for(int iter = 0; iter < video->getNumberOfFrames(); ++iter)
    {
      if(iter % 100 == 0)
      {
	shared_ptr<MetaData_YML_Backed> metadata = video->getFrame(iter,true);
	shared_ptr<ImRGBZ> im = metadata->load_im();	    
	
	// get rect from user
	Rect bb = getRect("give rect", imageeq("",im->Z,false,false));
	
	// 
	cout << "manifoldFn_all = " << manifoldFn_all(*im,bb) << endl;
	cout << "manifoldFn_kmax = " << manifoldFn_kmax(*im,bb) << endl;
	cout << "manifoldFn_boxMedian = " << manifoldFn_boxMedian(*im,bb) << endl;
	cout << "manifoldFn_apxMin = " << manifoldFn_apxMin(*im,bb) << endl;
	cout << "manifoldFn_min = " << manifoldFn_min(*im,bb) << endl;

	// extracted template
	float z = manifoldFn_boxMedian(*im,bb).front();
	auto subim = (*im)(bb);
	VolumetricTemplate T(subim,z,nullptr,RotatedRect());
	Mat Tim = T.getTIm();
	if(Tim.empty())
	  image_safe("T",image_text("invalid template"));
	else
	  imageeq("T",imVGA(Tim),true,false);
      }
    }
  }

  void invoke_tests()
  {
    cout << "++invoke_tests2" << endl;

    //test_spiral_volumetry();
    test_spherical_volumetry(); 
    //test_manifold_selection();
    //test_ortho_fb();
    //test_metric_feature();
    //test_inverse_kinematics();
    //test_ortho_dist();
    //test_convex_hull();
    //test_skin_detection();
    //test_SubTrees();
    //test_scores();
    //test_vis();
    //test_morphology();
    //test_vec();

    cout << "--invoke_tests" << endl;
  }
}
