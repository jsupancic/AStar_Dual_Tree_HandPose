/**
 * Copyright 2014: James Steven Supancic III
 **/ 

#include "KinectPose.hpp"
#include "Model.hpp"
#include "FingerPoser.hpp"
#include "boost/multi_array.hpp"
#include "RandomHoughFeature.hpp"
#include "Colors.hpp"

namespace deformable_depth
{
  static Mat skin_dets(const ImRGBZ&im)
  {
    if(not g_params.option_is_set("SKIN_FILTER"))
    {
      // assume all pixels are skin, eg no filter.
      return Mat(im.rows(),im.cols(),DataType<uint8_t>::type,Scalar::all(0));    
    }     
    assert(im.valid_region != Rect());

    Mat dets(im.rows(),im.cols(),DataType<uint8_t>::type,Scalar::all(255));    
    Mat dets_vis(im.rows(),im.cols(),DataType<Vec3b>::type,Scalar::all(255));    
    Mat skin_ratio = im.skin();
    for(int rIter = 0; rIter < dets.rows; rIter++)
      for(int cIter = 0; cIter < dets.cols-1; cIter++)
	// mark some pixels as skin
	if(skin_ratio.at<float>(rIter,cIter) > .1 and 
	   im.RGB.at<Vec3b>(rIter,cIter) != Vec3b(255,255,255) and
	   im.valid_region.contains(Point(cIter,rIter)) and
	   im.RGB.at<Vec3b>(rIter,cIter) != im.RGB.at<Vec3b>(rIter,cIter+1))
	{
	  dets.at<uint8_t>(rIter,cIter) = 0;
	  dets_vis.at<Vec3b>(rIter,cIter) = 0;
	}

    Mat viz = horizCat(dets_vis,im.RGB);
    log_im_decay_freq("skin_dets"+im.filename,viz);
    return dets;
  }

  ///
  /// SECTION: KinectPoseModel
  ///
  KinectPoseModel::Prediction KinectPoseModel::predict_un_in_plane_rot(
    const Mat&seg,const Mat&Z,const CustomCamera&camera,double theta) const
  {
    if(not g_params.option_is_set("IMPLICIT_IN_PLANE_ROTATION"))
      return this->predict(seg,Z,camera);
      
    Mat seg_rot = imrotate(seg,theta);
    Mat Z_rot   = imrotate(Z,theta);
    Mat rotMat = cv::getRotationMatrix2D(Point2d(Z.cols/2,Z.rows/2),rad2deg(theta),1); 
    Mat invRot; cv::invertAffineTransform(rotMat,invRot);
    invRot.convertTo(invRot,DataType<float>::type);
    log_im("hough_pseudo_rectification",imageeq("",Z_rot,false,false));
    
    Prediction pred = this->predict(seg_rot,Z_rot,camera);
    //pred.det->tighten_bb();

    pred.det->applyAffineTransform(invRot);
    pred.map = imrotate(pred.map,-theta);

    return pred;
  }

  static double angle(MetaData&sub_datum)
  {
    auto poss = sub_datum.get_positives();
    if(poss.find("HandBB") == poss.end() or poss.find("dist_phalan_3") == poss.end())
      return 0;

    Point2d center = rectCenter(poss.at("HandBB"));
    Point2d top = rectCenter(poss.at("dist_phalan_3"));
    double x = top.x - center.x;
    double y = top.y - center.y;
    return std::atan2(y,x) + params::PI/2;
  }

  DetectionSet KinectPoseModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  { 
    DetectionSet dets; 
 
    // use cheat code for segmentation, if given
    shared_ptr<MetaData> cheat = filter.cheat_code.lock();
    if(cheat)
    {
      for(auto && sub_datum : cheat->get_subdata())
      {
	Rect sub_datum_bb;
	Mat seg;
	if(g_params.option_is_set("CHEAT_HAND_BB"))
	{
	  // get the cheat bb
	  auto poss = sub_datum.second->get_positives();
	  bool has_hand_bb = (poss.find("HandBB") != poss.end());
	  if(!has_hand_bb)
	    continue;
	  sub_datum_bb = rectResize(poss["HandBB"],1.5,1.5);
	  Point2d sub_datum_center = rectCenter(sub_datum_bb);      
	  if(sub_datum_bb == Rect())
	    continue;
	  seg = DumbPoser().segment(sub_datum_bb,im);
	}
	else
	{
	  // come up with the segmentation using a Hough forest
	  Mat skin = skin_dets(im);
	  Mat det_map = detection_forest.predict_one_part(skin,im.Z,im.camera);	  
	  DetectorResult det = predict_joints_mean_shift(im.Z,det_map,im.camera,150);
	  Mat vis_det = imageeq("",det_map,false,false);
	  cv::rectangle(vis_det,det->BB.tl(),det->BB.br(),toScalar(RED));
	  Mat vis_depth = imageeq("",im.Z,false,false);
	  log_im("HoughDetection",horizCat(vis_det,vis_depth));
	  sub_datum_bb = det->BB;
	  seg = DumbPoser().segment(sub_datum_bb,im);
	}
	// run the detection operation	
	log_im(im.filename + "segmentation",seg);
	
	// if(sub_datum.second->leftP() != is_left)
	// detect flipped
	{
	  Mat segflip; cv::flip(seg,segflip,1);
	  ImRGBZ flip = im.flipLR();
	  auto prediction = this->predict_un_in_plane_rot(segflip,flip.Z,im.camera,angle(*sub_datum.second));
	  log_im(im.filename+"_flipLR",prediction.map);
	  Mat affine = affine_lr_flip(im.cols());
	  detection_affine(*prediction.det,affine);
	  prediction.det->pose = safe_printf("KinectPose lrcfg flipped_%_% lrcfg",sub_datum.second->leftP(),is_left);
	  prediction.det->BB = sub_datum_bb;

	  if(sub_datum.second->leftP() != is_left)
	    dets.push_back(prediction.det);
	}
	// else
	{
	  // detect orignal
	  Prediction prediction = this->predict_un_in_plane_rot(seg,im.Z,im.camera,angle(*sub_datum.second));
	  prediction.det->pose = safe_printf("KinectPose lrcfg standard_%_% lrcfg",sub_datum.second->leftP(),is_left);
	  prediction.det->BB = sub_datum_bb;
	  log_im(im.filename,prediction.map);

	  if(dets.empty() or sub_datum.second->leftP() == is_left)
	    dets.push_back(prediction.det);
	}
      }
    }

    return dets;
  }

  DetectorResult predict_joints_mean_shift(const Mat&Z,const Mat&probImage,const CustomCamera&camera,
					   double metric_side_length_cm)
  {   
    assert(probImage.type() == DataType<double>::type);
    Rect best_bb;
    double best_score = -inf;
    log_im_decay_freq("prob_image",imageeq("",probImage,false,false));
    Mat denseProbImage; probImage.convertTo(denseProbImage,DataType<float>::type);
    denseProbImage = fillDepthHoles(denseProbImage,5);

    auto score_window = [&](Rect bb)
    {
      double score = 0;
      double count = 0;
      Mat p_roi = probImage(bb);
      for(int yIter = 0; yIter < p_roi.rows; yIter++)
	for(int xIter = 0; xIter < p_roi.cols; xIter++)
	{
	  double p = p_roi.at<double>(yIter,xIter);
	  if(goodNumber(p))
	  {
	    score += p;
	    count ++;
	  }	  
	}
      return score/count;
    };

    for(int iter = 0; iter < 250; ++iter)
    {
      // initialize
      Point center0 = rnd_multinom(probImage);
      Rect bb = camera.bbForDepth(Z,center0.y,center0.x,metric_side_length_cm,metric_side_length_cm);
      bb = clamp(probImage,bb);
      if(bb.height <= 0 or bb.width <= 0)
	continue;
      // try the init position...
      double score = score_window(bb);
      if(score > best_score)
      {
	best_bb = bb;
	best_score = score;
      }
      // run meanshift to convergence      
      cv::TermCriteria termCrit(TermCriteria::MAX_ITER,100,qnan);
      cv::meanShift(denseProbImage,bb,termCrit);
      //Point center_ms = rectCenter(bb);
      //bb = camera.bbForDepth(Z,center_ms.y,center_ms.x,5,5);

      // calculate the score and take the best over random initializations
      score = score_window(bb);
      if(score > best_score)
      {
	best_bb = bb;
	best_score = score;
      }
    }

    DetectorResult part = make_shared<Detection>();
    part->resp = .5;
    part->BB = best_bb;
    return part;
  }

  static DetectorResult predict_joints_average(const Mat&Z,Mat&probImage,const CustomCamera&camera)
  {
    Vec2d mean_loc(0,0);
    double loc_count = 0;
    for(int yIter = 0; yIter < probImage.rows; yIter++)
      for(int xIter = 0; xIter < probImage.cols; xIter++)	  
      {	  	
	double weight = probImage.at<double>(yIter,xIter);
	mean_loc += weight*Vec2d(xIter,yIter);
	loc_count += weight;	    
      }
    mean_loc /= loc_count;
    
    DetectorResult part = make_shared<Detection>();
    part->resp = .5;
    part->BB = camera.bbForDepth(Z,mean_loc[1],mean_loc[0],5,5);
    return part;
  }
  
  Mat KinectPoseModel::show(const string&title)
  {
    return Mat();
  }

  KinectPoseModel::~KinectPoseModel()
  {
  }

  ///
  /// SECTION: Keskin's model
  /// 
  void KeskinsModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params )
  {
    log_file << "KinectPoseModel::train " << training_set.size() << endl;
    // collect the positive examples
    vector<shared_ptr<MetaData> >  positive_set;
    vector<shared_ptr<MetaData> >  negtive_set;    
    split_pos_neg(training_set,positive_set,negtive_set);
    log_once(safe_printf("KinectPoseModel pos_set = % neg_set = %",positive_set.size(),negtive_set.size()));
    int npos = fromString<int>(g_params.require("NPOS"));
    vector<shared_ptr<MetaData> > use_set = random_sample<shared_ptr<MetaData> >(positive_set,npos);
    log_once(safe_printf("KinectPoseModel use_set = ",use_set.size()));

    // init the forest
    for(int iter = 0; iter < 15; ++iter)
      forest.push_back(StochasticExtremelyRandomTree());

    // show the training set...
    int training_counter = 0;    
    for(int iter = 0; iter < 1; ++iter)
    {
      progressBars->set_progress("KinectPoseModel::train",iter,10);
      //forest.clear();
      //for(int iter = 0; iter < 15; ++iter)
      //forest.push_back(StochasticExtremelyRandomTree());

      use_set = pseudorandom_shuffle(use_set);
      if(iter == 0)
	is_left = use_set.at(0)->leftP();
      atomic<int> datum_count(0);
      for(auto & datum : use_set)
      {
	if(datum->leftP() != is_left)
	  continue;
	progressBars->set_progress("KinectPoseModel::train datum",datum_count++,use_set.size());
	
	// show the training example
	//imageeq("training depth",datum->load_im()->Z);
	Mat sem = datum->getSemanticSegmentation();
	shared_ptr<ImRGBZ> im = datum->load_im();
	Mat Z = im->Z;
	// Rect handBB = datum->get_positives().at("HandBB");
	// if(!rectContains(Z,handBB))
	// {
	//   cout << "warning: skipping " << datum->get_filename() << endl;
	//   continue;
	// }
	Mat seg(sem.rows,sem.cols,DataType<uint8_t>::type,Scalar::all(0));
	for(int rIter = 0; rIter < seg.rows; rIter++)
	  for(int cIter = 0; cIter < seg.cols; cIter++)
	    if(HandRegion::is_background(sem.at<Vec3b>(rIter,cIter)))
	      seg.at<uint8_t>(rIter,cIter) = 255;
	  //Mat seg = VoronoiPoser().segment(handBB,*im);
	assert(seg.type() == DataType<uint8_t>::type);

	// train the colors
	require_equal(seg.rows,im->rows());
	require_equal(seg.cols,im->cols());
	for(int yIter = 0; yIter < im->rows(); yIter++)
	  for(int xIter = 0; xIter < im->cols(); xIter++)
	    if(seg.at<uint8_t>(yIter,xIter) <= 100)
	    {
	      if(discrete_colors.find(sem.at<Vec3b>(yIter,xIter)) == discrete_colors.end())
	      {
		int id = next_color++;
		discrete_colors.insert({sem.at<Vec3b>(yIter,xIter),id});
		colors_discrete.insert({id,sem.at<Vec3b>(yIter,xIter)});
	      }
	    }
	log_once(safe_printf("discrete_colors.size() = %",discrete_colors.size()));

	// train the trees locally...	
	TaskBlock train_trees("train_trees");
	for(int iter = 0; iter < forest.size(); ++iter)
	{
	  train_trees.add_callee([&,this,iter]() 
				 {
				   auto&tree = this->forest.at(iter);
				   for(int yIter = 0; yIter < im->rows(); yIter++)
				     for(int xIter = 0; xIter < im->cols(); xIter++)
				     {
				       int row = thread_rand()%im->rows();
				       int col = thread_rand()%im->cols();
				       if(sem.at<Vec3b>(row,col) == HandRegion::wrist_color() and thread_rand()%2)
					 continue;

				       if(seg.at<uint8_t>(row,col) <= 100)
				       {					 
					 tree.train(sem,Z,Point(col,row));
				       }
				     }
				 });
	}
	Point2d hand_center = rectCenter(datum->get_positives().at("HandBB"));
	Mat skin = skin_dets(*im);
	detection_forest.train_one(im->Z,hand_center,train_trees,skin);
	train_trees.execute();

	// show the test
	Mat test_vis = predict(seg,Z,im->camera).map;

	// side by side comparison on training data
	vector<Mat> viss{sem,test_vis,imageeq("",im->Z,false,false),im->RGB};
	log_im(safe_printf("segmentation%",training_counter),tileCat(viss));
      }
      progressBars->set_progress("KinectPoseModel::train datum",1,1);
    }
    progressBars->set_progress("KinectPoseModel::train",1,1);
  }

  Mat KeskinsModel::vis_map(const Mat&seg,const Mat&Z,AtomicArray3D&posteriors) const
  {
    // compute normalization constants
    vector<double> sums;
    vector<Mat> planes;
    for(int color_iter = 0; color_iter < discrete_colors.size(); ++color_iter)
    {
      double sum = 0;

      for(int yIter = 0; yIter < Z.rows; yIter++)
	for(int xIter = 0; xIter < Z.cols; xIter++)
	{
	  double post = *posteriors[xIter][yIter][color_iter];
	  if(goodNumber(post))
	  {
	    sum += post;
	  }
	}

      sums.push_back(sum);
    }

    // take the max to get the MAP estimate
    Mat test_vis = Mat(Z.rows,Z.cols,DataType<Vec3b>::type,Scalar::all(0));
    for(int yIter = 0; yIter < Z.rows; yIter++)
      for(int xIter = 0; xIter < Z.cols; xIter++)
	if(seg.empty() or seg.at<uint8_t>(yIter,xIter) <= 100)
	{
	  double max = -inf;
	  for(int color_iter = 0; color_iter < discrete_colors.size(); ++color_iter)
	  {
	    double post = *posteriors[xIter][yIter][color_iter] / sums.at(color_iter);
	    if(post > max)
	    {
	      max = post;
	      test_vis.at<Vec3b>(yIter,xIter) = colors_discrete.at(color_iter);
	    }
	  }
	}

    return test_vis;
  }

  KinectPoseModel::Prediction KeskinsModel::predict(const Mat&seg,const Mat&Z,const CustomCamera&camera) const
  {
    assert(Z.type() == DataType<float>::type);
    assert(seg.empty() or seg.type() == DataType<uint8_t>::type);    
    AtomicArray3D posteriors(boost::extents[Z.cols][Z.rows][discrete_colors.size()]);
    for(int xIter = 0; xIter < Z.cols; xIter++)
      for(int yIter = 0; yIter < Z.rows; yIter++)
	for(int cIter = 0; cIter < discrete_colors.size(); ++cIter)
	  posteriors[xIter][yIter][cIter].reset(new atomic<long>(0));

    // accumulate the counts over the trees
    for(auto && tree : forest)
    {
      for(int yIter = 0; yIter < Z.rows; yIter++)
	for(int xIter = 0; xIter < Z.cols; xIter++)
	  if(seg.empty() or seg.at<uint8_t>(yIter,xIter) <= 100)
	  {
	    // use regression
	    // Vec3d prediction(0,0,0);
	    // for(auto && tree : forest)
	    //   prediction += tree.predict(Z,Point(xIter,yIter));
	    // double pred_b = prediction[0]/forest.size();
	    // double pred_g = prediction[1]/forest.size();
	    // double pred_r = prediction[2]/forest.size();
	    // test_vis.at<Vec3b>(yIter,xIter) = Vec3b(pred_b,pred_g,pred_r);
	    
	    // use classification	   
	    unordered_map<Vec3b,long> lcl_posterior = tree.posterior(Z,Point(xIter,yIter));
	    for(auto && pair : lcl_posterior)
	    {
	      int color_id = discrete_colors.at(pair.first);
	      //cout << "color_id += " << pair.second << endl;;
	      atomic<long>&ctr = (*posteriors[xIter][yIter][color_id]);
	      ctr += pair.second;
	    }
	  }
    }

    // 
    Mat test_vis = vis_map(seg,Z,posteriors);

    // extern the detection...
    DetectorResult det = make_shared<Detection>();
    det->resp = .5;
    vector<string> parts = {"wrist","dist_phalan_1","dist_phalan_2","dist_phalan_3","dist_phalan_4","dist_phalan_5"};
    Mat seg_vis = seg.clone(); seg_vis.convertTo(seg_vis,DataType<float>::type); seg_vis = imageeq("",seg_vis,false,false);
    vector<Mat> vis{test_vis,seg_vis};
    for(auto && part_name : parts)
    {
      Vec3b part_color = HandRegion::part_color(part_name);
      if(discrete_colors.find(part_color) == discrete_colors.end())
      {
	ostringstream oss;
	oss << safe_printf("Can't find part's (%) color % in ",part_name,part_color);
	for(auto && stored_name : discrete_colors)
	  oss << stored_name.first << " ";
	log_once(oss.str());
	
	continue; // skip parts we haven't seen yet during training...
      }
      int color_id = discrete_colors.at(part_color);
      Mat probImage(Z.rows,Z.cols,DataType<double>::type,Scalar::all(0));
      for(int yIter = 0; yIter < Z.rows; yIter++)
	for(int xIter = 0; xIter < Z.cols; xIter++)
	{
	  if(seg.at<uint8_t>(yIter,xIter) > 100)
	  {
	    probImage.at<double>(yIter,xIter) = qnan;
	    continue;
	  }

	  double total_weight = 0;
	  for(int colorIter = 0; colorIter < discrete_colors.size(); ++colorIter)
	    total_weight += *posteriors[xIter][yIter][colorIter];

	  double weight = *posteriors[xIter][yIter][color_id]/total_weight;
	  if(not goodNumber(weight))
	    weight = 1.0/static_cast<double>(discrete_colors.size());
	  probImage.at<double>(yIter,xIter) = weight;
	}
      
      auto part = predict_joints_mean_shift(Z,probImage,camera);

      det->emplace_part(part_name,*part);

      Mat vis_prob = imageeq("",probImage,false,false);
      cv::rectangle(vis_prob,part->BB.tl(),part->BB.br(),Scalar(0,255,0));
      vis.push_back(vis_prob);
    }
    //det->tighten_bb();
    log_im("KeskinDets",tileCat(vis));

    return Prediction{test_vis,det};
  }

  ///
  /// SECTION: Xus model
  /// 
  KinectPoseModel::Prediction XusModel::predict(const Mat&seg,const Mat&Z,const CustomCamera&camera) const
  {
    DetectorResult det = make_shared<Detection>();
    det->resp = .5;
    set<string> parts = essential_hand_positives();
    vector<Mat> vis{imageeq("",Z,false,false)};
    TaskBlock predict_parts("predict_parts");
    mutex m;
    for(auto && part_name : parts)
    {
      predict_parts.add_callee([&,this,part_name]()
			       {
				 // generate the PDF
				 Mat probImage = this->hough_forests.at(part_name)
				   .predict_one_part(seg,Z,camera);
				 
				 // from the probability image, form the detection
				 auto part = predict_joints_mean_shift(Z,probImage,camera);
				 
				 // write the part
				 lock_guard<mutex> l(m);
				 det->emplace_part(part_name,*part);
				 if(det->BB == Rect_<double>())
				   det->BB = part->BB;
				 else
				   det->BB |= part->BB;
				 
				 probImage.convertTo(probImage,DataType<float>::type);
				 Mat vis_prob = imageeq("",probImage,false,false);
				 //cv::applyColorMap(vis_prob,vis_prob,COLORMAP_JET);
				 cv::rectangle(vis_prob,part->BB.tl(),part->BB.br(),toScalar(RED));
				 vis.push_back(vis_prob);
			       });
    }
    predict_parts.execute();

    log_im("XuDets",tileCat(vis));

    return Prediction{imageeq("",Z.clone(),false,false),det};
  }

  void XusModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    log_file << "XuModel::train " << training_set.size() << endl;
    // collect the positive examples
    vector<shared_ptr<MetaData> >  positive_set;
    vector<shared_ptr<MetaData> >  negtive_set;    
    split_pos_neg(training_set,positive_set,negtive_set);
    log_once(safe_printf("XuPoseModel pos_set = % neg_set = %",positive_set.size(),negtive_set.size()));
    int npos = fromString<int>(g_params.require("NPOS"));
    vector<shared_ptr<MetaData> > use_set = random_sample<shared_ptr<MetaData> >(positive_set,npos);
    log_once(safe_printf("XuPoseModel use_set = ",use_set.size()));
    
    // init the forests
    set<string> parts = essential_hand_positives();
    for(auto && part_name : parts)
      hough_forests[part_name] = HoughForest();

    // show the training set...
    int training_counter = 0;    
    for(int iter = 0; iter < 1; ++iter)
    {
      progressBars->set_progress("KinectPoseModel::train",iter,10);

      use_set = pseudorandom_shuffle(use_set);
      if(iter == 0)
	is_left = use_set.at(0)->leftP();
      atomic<int> datum_count(0);
      for(auto & datum : use_set)
      {
	assert(datum->leftP() == is_left);
	int datum_number = datum_count++;
	progressBars->set_progress("KinectPoseModel::train datum",datum_number,use_set.size());
	
	// show the training example
	Mat sem = datum->getSemanticSegmentation();
	assert(not sem.empty());
	shared_ptr<ImRGBZ> im = datum->load_im();
	Mat Z = im->Z;
	auto poss = datum->get_positives();
	Mat seg(sem.rows,sem.cols,DataType<uint8_t>::type,Scalar::all(0));
	for(int rIter = 0; rIter < seg.rows; rIter++)
	  for(int cIter = 0; cIter < seg.cols; cIter++)
	    if(HandRegion::is_background(sem.at<Vec3b>(rIter,cIter)))
	      seg.at<uint8_t>(rIter,cIter) = 255;

	// train the trees locally...	
	TaskBlock train_trees("train_trees");
	for(auto && forest : hough_forests)
	{
	  string part_name = forest.first;
	  Point2d part_center = rectCenter(poss.at(part_name));
	  forest.second.train_one(im->Z,part_center,train_trees,seg);
	}
	Point2d hand_center = rectCenter(poss.at("HandBB"));
	Mat skin = skin_dets(*im);
	detection_forest.train_one(im->Z,hand_center,train_trees,skin);
	train_trees.execute();

	// show the test
	Mat hand_map = detection_forest.predict_one_part(skin,im->Z,im->camera);
	DetectorResult det = predict_joints_mean_shift(im->Z,hand_map,im->camera,150);
	Mat vis_det = imageeq("",hand_map,false,false);
	Mat vis_depth = imageeq("",im->Z,false,false);
	cv::rectangle(vis_det,det->BB.tl(),det->BB.br(),toScalar(RED));	
	log_im(safe_printf("HandHough_%_",datum_number),horizCat(vis_depth,horizCat(im->RGB,vis_det)));
	Mat test_vis = predict(seg,Z,im->camera).map;

	// side by side comparison on training data
	log_im(safe_printf("segmentation%",training_counter),horizCat(sem,test_vis));
      } // END Loop over data
      progressBars->set_progress("KinectPoseModel::train datum",1,1);
    }
    progressBars->set_progress("KinectPoseModel::train",1,1);
  }
}
