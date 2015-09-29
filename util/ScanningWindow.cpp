/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ScanningWindow.hpp"
#include "OcclusionReasoning.hpp"
#include "Faces.hpp"
#include "Colors.hpp"
#include "FingerPoser.hpp"

namespace deformable_depth
{
  enum JumpCodes
  {
    GOOD = 0,
    BB_CHEAT = 1,
    BAD_METRIC_BB = 2,
    BAD_SKIN_RATIO = 3,
    BAD_FACE_OVERLAP,
    SMALL_DEPTH,
    NEGATIVE_DEPTH,
    BAD_METRIC_AREA,
    BAD_IMAGE_AREA
  };

  ///
  /// SECTION: Training
  ///  
  typedef boost::multi_array<DetectionSet, 2> DetGrid;
  static double STRIDE = 1;

  Size MetricSize(string video_name)
  {
    constexpr int METRIC_RES = 60;

    const static boost::regex synth_regex(".*synth.*",boost::regex::icase);
    const static boost::regex seq_regex(".*sequence.*");
    const static boost::regex nyu_regex(".*NYU.*",boost::regex::icase);
    
    if(g_params.has_key("METRIC_WIDTH"))
    {
      double width = fromString<double>(g_params.require("METRIC_WIDTH"));
      double height = fromString<double>(g_params.require("METRIC_HEIGHT"));
      return Size(width,height);
    }
    if(boost::regex_match(video_name,seq_regex))
    {
      log_once("returned sequence size");
      return Size(METRIC_RES,METRIC_RES);
    }
    else if(boost::regex_match(video_name,synth_regex))
    {
      log_once(safe_printf("returned synth size for %",video_name));
      return Size(2.0/2.0*METRIC_RES,2.0/2.0*METRIC_RES);
    }
    else if(boost::regex_match(video_name,nyu_regex))
    {
      log_once("returned nyu size");
      return Size(METRIC_RES,METRIC_RES); 
    }
    else
    {
      Size sz(2.0/3.0*METRIC_RES,2.0/3.*METRIC_RES);
      log_once(safe_printf("MetricSize: returned other size for % = %",video_name,sz));
      //return Size(METRIC_RES,METRIC_RES);
      return sz;
    }
  }

  // this is called on positive training examples to extract the positive bounding boxes
  // to use for training.
  vector<AnnotationBoundingBox> metric_positive(
    MetaData&ex, const ImRGBZ&im,Rect train_bb, bool filter = true)
  {
    const Mat&Z = im.Z;
    vector<AnnotationBoundingBox> bbs{};
    if(filter and !rectContains(Z, train_bb))
    {
      log_once(printfpp("warning: Z does not fully contain train_bb %s",im.filename.c_str()));
      Mat vis_of_bad_datum = image_datum(ex,false);
      log_im("badHandBB",vis_of_bad_datum);
      return bbs;
    }
    //vector<float> depths = manifoldFn(im,train_bb);    
    //vector<float> depths = manifoldFn_ordApx(im,train_bb,.01);    
    // median
    vector<float> depths = manifoldFn_boxMedian(im,train_bb);
    if(depths.size() == 0)
    {
      assert(filter);
      return bbs;
    }
    depths = vector<float>{depths.front()};
    // approx min
    //vector<float> depths = manifoldFn_ordApx(im,rectResize(train_bb,.75,.75),.01);

    for(float z : depths)
    {
      if(filter and (z < params::MIN_Z() || z > params::MAX_Z()))
      {
	log_file << "warning: bad depth in training image gt" << endl;
	continue;
      }
      
      // compute metric ROI
      Point2i center = rectCenter(train_bb);
      Size metricSize = MetricSize(im.filename);
      Rect metric_bb = im.camera.bbForDepth(
	z, Z.size(), center.y, center.x, metricSize.width, metricSize.height,false);
      // DEBUG
      //assert(checkArea(*im,z, metric_bb));
      
      if(filter and 
	static_cast<cv::Rect>((metric_bb & static_cast<cv::Rect>(train_bb))) != static_cast<cv::Rect>(train_bb))
      {
	ostringstream oss; 
	oss << static_cast<cv::Rect>(metric_bb) << " vs " << 
	  static_cast<cv::Rect>(train_bb);
	string cam_str = to_string(im.camera);
	log_once(safe_printf(
		   "warning: metric_bb smaller than train_bb// [z = %] % % camera = % metric_size = % zSize = %" ,
			     z,im.filename,oss.str(),cam_str,metricSize,Z.size()));
	log_im_decay_freq("invalid_metric_bb",[&]()
			  {
			    Mat vis = im.RGB.clone();
			    cv::rectangle(vis,train_bb.tl(),train_bb.br(),toScalar(BLUE));
			    cv::rectangle(vis,metric_bb.tl(),metric_bb.br(),toScalar(GREEN));
			    return vertCat(vis,image_text(safe_printf("depth = %",z)));
			  });
	continue;
      }
      AnnotationBoundingBox abb;
      static_cast<Rect_<double>&>(abb) = (metric_bb);
      abb.depth = z;
      bbs.push_back(abb);
    }    
    
    return bbs;
  }

  vector<AnnotationBoundingBox> metric_positive(MetaData&ex)
  {
    shared_ptr<ImRGBZ> im = ex.load_im();
    auto poss = ex.get_positives();
    auto handItr = poss.find("HandBB");
    if(handItr != poss.end())
    {
      Rect train_bb = handItr->second;
      return metric_positive(ex,*im,train_bb,false);
    }
    else
    {
      return vector<AnnotationBoundingBox>{};
    }
  }

  void train_extract_template(const shared_ptr<MetaData>&ex,
			      map<string,string>&sources,
			      map<string,NNTemplateType>&allTemplates,
			      map<string,map<string,AnnotationBoundingBox> >&parts_per_ex)
  {
    if(!ex->use_positives())
    {
      log_file << "warning: use_positives = false" << endl;
      return;
    }
    log_file << printfpp("extracting positives from %s",ex->get_filename().c_str()) << endl;
    
    shared_ptr<const ImRGBZ> im = ex->load_im();
    if(im->Z.size().area() == 0)
    {
      log_file << "warning: no depth for " << ex->get_filename() << endl;
      return;
    }
    if(ex->hasAnnotation("hand in freespace?") and 
       ex->getAnnotation("hand in freespace?").find("y") == string::npos)
    {
      log_file << "warning: positive not in freespace " << ex->get_filename() << endl;
      return;
    }
    
    map<string,AnnotationBoundingBox> pos_bbs = ex->get_positives();
    map<string,AnnotationBoundingBox> train_bbs = params::defaultSelectFn()(pos_bbs);
    log_file << "got " << train_bbs.size() << " train_bbs" << endl;
    for(auto && train_bb_ex : train_bbs)
    {
      log_file << "extracting: " << train_bb_ex.second << " from " << ex->get_filename() << endl;
      if(train_bb_ex.second.area() <= 0)
	continue;
      
      for(auto && metric_pos : metric_positive(*ex, *im,train_bb_ex.second))
      {
	Rect metric_bb = metric_pos;
	RotatedRect metric_rot_bb(rectCenter(metric_pos),metric_pos.size(),0);
	ImRGBZ im_roi = (*im)(metric_pos);
	NNTemplateType T(im_roi,metric_pos.depth,ex,metric_rot_bb);
	if(T.getTIm().empty())
	{
	  log_once(printfpp("warning: unreal template %s",im->filename.c_str()));
	  continue;
	}
	Mat TVis = imVGA(imageeq("pos_template",T.getTIm(),false,false));
	Mat ImVis = imageeq("",im->Z,false,false);
	cv::rectangle(ImVis,metric_pos.tl(),metric_pos.br(),Scalar(255,0,0));
	cv::rectangle(ImVis,train_bb_ex.second.tl(),train_bb_ex.second.br(),Scalar(0,0,255));
	Mat vis = horizCat(TVis,ImVis);
	log_im_decay_freq("FinalTemplateVis",vis);

	static mutex m; lock_guard<mutex> l(m);
	string key = yaml_fix_key(uuid());	
	sources[key] = ex->get_filename();
	allTemplates[key] = T;	
	// store the parts
	parts_per_ex[key] = params::makeSelectFn(".*dist_phalan.*")(
	  ex->get_positives());	
      }
    }
  }

  // collect all the positive examples from the training set...
  ExtractedTemplates extract_templates(
    vector< shared_ptr< MetaData > >& training_set, Model::TrainParams train_params)
  {
    // collect the positive examples    
    // loop over the training set in parallel
    TaskBlock extract_templates("extract_templates");
    ExtractedTemplates allTemplates;
    for(auto & ex : training_set)
    {
      extract_templates.add_callee([&,ex]()
				   {
				     train_extract_template(ex,
							    allTemplates.sources,
							    allTemplates.allTemplates,
							    allTemplates.parts_per_ex);
				   });
    }
    extract_templates.execute();
    log_once(safe_printf("extracted % of % templates",allTemplates.allTemplates.size(),training_set.size()));

    return allTemplates;
  }

  /// 
  /// SECTION: Testing
  ///
  bool checkArea(const ImRGBZ&im,float z, Rect metric_bb)
  {
    assert(!im.camera.is_orhographic());
    double lenExp = std::sqrt<double>(MetricSize(im.filename).area()).real();
    double lenAct = std::sqrt<double>(im.camera.worldAreaForImageArea(z, metric_bb)).real();
    //cout << "checkArea: " << lenExp << " " << lenAct << endl;
    double err = lenAct/lenExp;
    if(err < 3.0/4.0  || err > 3.0/2.0)
    {
      return false;
    }    
    return true;
  }  

  void set_det_positions2
  (RotatedRect gtBB, shared_ptr<MetaData>&datum,string pose,
   RotatedRect orig_bb, // X BB
   map<string,AnnotationBoundingBox> & parts,
   DetectorResult&det,float depth)
  {
    //Rect gtHandBB = Tex.getMetadata()->get_positives().at("HandBB");
    //RotatedRect gtBB(rectCenter(gtHandBB),gtHandBB.size(),0);
    
    Mat AfT = affine_transform_rr(gtBB,orig_bb);
    det->BB = rect_Affine(datum->get_positives()["HandBB"],AfT);
    for(auto && part : parts)
    {
      part.second.depth = depth;
      part.second.write(rect_Affine(part.second,AfT));
    }
    det->set_parts(parts);
    det->exemplar = datum;
    det->pose = pose;	  
    det->in_plane_rotation = orig_bb.angle;
  }
  
  // Use the affine transform from the TBB to the XBB
  // to regress the pose.
  void set_det_positions
  (const NNTemplateType&Tex,string pose,
   RotatedRect orig_bb, // X BB
   map<string,AnnotationBoundingBox> & parts,
   DetectorResult&det,float depth)
  {    
    RotatedRect gtBB = Tex.getExtractedFrom(); // template BB
    auto datum = Tex.getMetadata();
    set_det_positions2(gtBB,datum,pose,orig_bb,parts,det,depth);
  }
    
  DetectionSet detect_for_window
    (
     const ImRGBZ& im,
     Rect orig_bb,
     const vector<Rect>&bbs_faces,
     Mat&respIm,Mat&invalid,
     EPM_Statistics&stats,DetectionFilter&filter) 
  { 
    DetectionSet dets;

    int jump_code = 0;
    // skip metric bbs not fully contained in the image
    Rect core_bb = rectResize(orig_bb,.5,.5);
    if(!rectContains(im.Z,core_bb))
    {
      stats.count("Bad MetricBB");
      jump_code = JumpCodes::BAD_METRIC_BB;
    }
    // apply the skin filter
    if(g_params.require("SKIN_FILTER") == "TRUE" && im.skin_ratio(rectResize(core_bb,.5,.5)) < .05)
    {
      stats.count("Bad Skin Ratio");
      jump_code = JumpCodes::BAD_SKIN_RATIO;
    }
    // check face overlap
    bool bad_face_overlap = false;
    for(Rect face_bb : bbs_faces)
      if(rectIntersect(face_bb,rectResize(core_bb,.5,.5)) > .15)
      {
	bad_face_overlap = true;
      }
    if(bad_face_overlap)
    {
      stats.count("Bad Face Overlap");
      jump_code = JumpCodes::BAD_FACE_OVERLAP;
    }

    if(jump_code == 0)
    {
      vector<float> depths = filter.manifoldFn(im,orig_bb);    
      for(float depth : depths)
      {    
	if(depth <= params::MIN_Z())
	{
	  stats.count("Small Depth");
	  jump_code = JumpCodes::SMALL_DEPTH;
	}
	if(depth <= 0)
	{
	  stats.count("Negative Depth");
	  jump_code = JumpCodes::NEGATIVE_DEPTH;
	}
	
	if(!checkArea(im,depth, orig_bb))
	{
	  stats.count("Bad Metric Area");
	  jump_code = JumpCodes::BAD_METRIC_AREA;
	}
	if(orig_bb.size().area() < params::min_image_area())
	{
	  stats.count("Bad Image Area");
	  jump_code = JumpCodes::BAD_IMAGE_AREA;
	}
	
	// if metric size checks out, compute the score
	auto det = make_shared<Detection>();
	det->BB = orig_bb;
	det->depth = depth;
	det->jump_code = jump_code;
	dets.push_back(det);
      }    
    }
    else
    {
      auto det = make_shared<Detection>();
      det->BB = orig_bb;
      det->depth = qnan;
      det->jump_code = jump_code;
      dets.push_back(det);
    }

    return dets;
  }

  void set_parent(int pyr_level,DetectorResult&child,DetectionSet&parent_cell,DetectionSet&all_parents) 
  {
    // try to find a parent
    for(auto && parent_candidate : parent_cell)
      if(std::abs<float>(child->depth - parent_candidate->depth) < params::MIN_Z_STEP)
      {
	child->pyramid_parent_windows.push_back(parent_candidate);
	return;
      }

    // otherwise, we need to allcoate a new parent
    DetectorResult parent = make_shared<Detection>();
    parent->BB = child->BB;
    parent->supressed = true;
    parent->resp = qnan;
    parent->depth = child->depth;
    parent->down_pyramid_count = pyr_level;
    all_parents.push_back(parent);
    parent_cell.push_back(parent);
    
    child->pyramid_parent_windows.push_back(parent);
  }

  DetectionSet build_resolution_pyramid(DetGrid&grid) 
  {
    DetectionSet all_parents;
    Size sz = SpearTemplSize;
    int pyr_level = 1;
    double ps = params::pyramid_sharpness();
    for(; sz.area() > 0; ++pyr_level)
    {
      int width  = grid.shape()[0];
      int height = grid.shape()[1];
      int newWidth = std::ceil(ps*width);
      int newHeight = std::ceil(ps*height);
      DetGrid newGrid(boost::extents[newWidth][newHeight]);
      if(newWidth == width and newHeight == height)
	break;

      // set the child pointers twoard the parents
      for(int xIter = 0; xIter < width; ++xIter)
	for(int yIter = 0; yIter < height; ++yIter)
	  for(auto && det : grid[xIter][yIter])
	  {
	    // calc the new coordinates
	    int nearest_parent_x = clamp<int>(0,std::round(ps*xIter),newWidth-1);
	    int nearest_parent_y = clamp<int>(0,std::round(ps*yIter),newHeight-1);
	    set_parent(pyr_level,det,newGrid[nearest_parent_x][nearest_parent_y],all_parents);	  

	    assert(det->pyramid_parent_windows.size() > 0);
	  }

      sz = Size(ps*sz.width, ps*sz.height);
      grid.resize(boost::extents[newWidth][newHeight]);
      grid = newGrid;
    }
    
    log_once(safe_printf("build_resolution_pyramid max_pyr_level = %",pyr_level-1));
    return all_parents;
  }

  DetectionSet enumerate_pyramid(
    const ImRGBZ&im,DetectionFilter filter,vector<Rect>&bbs_faces,
    EPM_Statistics&stats,Mat&respIm,Mat&invalid,Point2d cheat_center,Rect cheat_bb,double cheat_thresh)
  {
    double sf_base = 1.1;
    vector<double> sfs = getScaleFactors(SpearTemplSize.area(),
					 16*im.RGB.size().area(),
					 SpearTemplSize.area(),sf_base);

    DetectionSet dets;
    // here we search over the image Pyramid
    for(double sf : sfs)
    {
      Mat Zsc; cv::resize(im.Z,Zsc,Size(0,0),sf,sf, params::DEPTH_INTER_STRATEGY);
      typedef boost::multi_array<DetectionSet, 2> DetGrid;
      DetGrid detGrid(boost::extents[Zsc.cols/STRIDE][Zsc.rows/STRIDE]);

      for(double yPos = 0; yPos < Zsc.rows; yPos += STRIDE)
	for(double xPos = 0; xPos < Zsc.cols; xPos += STRIDE)
	{
	  Rect_<double> bb_sc = rectFromCenter(Point2d(xPos,yPos),SpearTemplSize);
	  Rect_<double> metric_cheat_bb = rectFromCenter(
	    Point2d(cheat_center.x,cheat_center.y),
	    Size_<double>(SpearTemplSize.width*1.0/sf,SpearTemplSize.height*1.0/sf));
	  Rect orig_bb = rectScale(bb_sc,1.0/sf);
	  
	  DetectionSet new_dets;
	  new_dets = detect_for_window
	    (im,orig_bb,bbs_faces,respIm,invalid,stats,filter);	  
	  if((cheat_bb == Rect() or rectIntersect(metric_cheat_bb,orig_bb) >= cheat_thresh))
	  {	   	    
	    // we are accepting the detection
	    detGrid[xPos][yPos] = new_dets;
	  }         
	  else
	  {
	    // we are rejecting the detection
	    for(auto && det : new_dets)
	    {
	      det->jump_code = JumpCodes::BB_CHEAT;
	    }
	  }	  
	  dets.insert(dets.end(),new_dets.begin(),new_dets.end());	    
	}

      DetectionSet pyr_parents = build_resolution_pyramid(detGrid);
      dets.insert(dets.end(),pyr_parents.begin(),pyr_parents.end());
    }
    return dets;
  }


  DetectionSet enumerate_windows(const ImRGBZ&im,DetectionFilter filter)
  {
    log_file << "++enumerate_windows" << endl;
    DetectionSet dets;
    EPM_Statistics stats(im.filename);
    Mat respIm(im.rows(),im.cols(),DataType<float>::type,Scalar::all(-inf));
    Mat invalid(im.rows(),im.cols(),DataType<uchar>::type,Scalar::all(true));
        
    // handle CHEAT_HAND_BB
    shared_ptr<MetaData> cheat = filter.cheat_code.lock();
    Rect cheat_bb;
    Point2d cheat_center;
    double cheat_thresh = .75;
    if(g_params.has_key("CHEAT_HAND_BB_THRESH"))
      cheat_thresh = fromString<double>(g_params.get_value("CHEAT_HAND_BB_THRESH"));
    if(cheat and g_params.option_is_set("CHEAT_HAND_BB"))
    {
      auto poss = cheat->get_positives();
      bool has_hand_bb = (poss.find("HandBB") != poss.end());
      if(!has_hand_bb)
      {
	log_once("warning: enumerate_windows failed to cheat");
	return dets;
      }      
      cheat_bb = poss["HandBB"];
      if(g_params.option_is_set("CHEAT_ONLY_FREESPACE") and !DumbPoser().is_freespace(cheat_bb,im) and thread_rand()%2 == 0)
      {
	int iter = 0;
	vector<float> cheat_depths;
	do
	{
	  int x1 = thread_rand()%im.cols();
	  int y1 = thread_rand()%im.rows();
	  int x2 = thread_rand()%im.cols();
	  int y2 = thread_rand()%im.rows();	 
	  cheat_bb = Rect(Point(x1,y1),Point(x2,y2));

	  cheat_depths = manifoldFn_boxMedian(im,cheat_bb);
	  iter++;
	} while(cheat_depths.empty() and iter < 10);
	if(iter >= 10)
	{
	  log_file << "Unable to place cheat BB: " << im.filename << endl;
	  return dets;
	}
      }      

      cheat_center = rectCenter(cheat_bb);      
      if(cheat_bb == Rect())
      {
	log_once("warning: enumerate_windows failed to cheat");
	return dets;
      }

      // add perfectly aigned window
      for(auto metric_cheat : metric_positive(*cheat))
      {
	auto cheat_dets = detect_for_window
	  (im,metric_cheat,vector<Rect>{},respIm,invalid,stats,filter);
	log_once(safe_printf("enumerate_windows inserted % perfect fits",cheat_dets.size()));
	dets.insert(dets.end(),cheat_dets.begin(),cheat_dets.end()); 	  
      }
    }

    // detect faces
    vector<Rect> bbs_faces;
    if(not g_params.has_key("CHEAT_HAND_BB_THRESH"))
      bbs_faces = SimpleFaceDetector().detect(im);

    // pyramid
    if(cheat_thresh <= 1)
    {
      DetectionSet pyr_dets = enumerate_pyramid(
	im,filter,bbs_faces,stats,respIm,invalid,cheat_center,cheat_bb,cheat_thresh);
      dets.insert(dets.end(),pyr_dets.begin(),pyr_dets.end());
    }
        
    // why was the closest skiped?
    if(cheat && g_params.option_is_set("CHEAT_HAND_BB"))
    {
      // find the metric BB closest to the annotation
      vector<AnnotationBoundingBox> metric_pos = metric_positive(*cheat, im,cheat_bb,false);
      AnnotationBoundingBox best_abb = metric_pos.front();
      Rect_<double> metric_pos_bb = static_cast<Rect_<double> >(best_abb);
      for(auto && abb : metric_pos)
	if(rectIntersect(best_abb,cheat_bb) < rectIntersect(abb,cheat_bb))
	  best_abb = abb;

      // now find the detection closest to the best metric bb and report why it was skipped...
      DetectorResult closest_det;
      for(auto det : dets)	
      {	
	if(closest_det == nullptr or 
	   rectIntersect(closest_det->BB,metric_pos_bb) < rectIntersect(det->BB,metric_pos_bb))
	  closest_det = det;
      }
      double ol = rectIntersect(closest_det->BB,metric_pos_bb);
      assert(closest_det);
      log_once(safe_printf("note(%): closest_det has jump code = % and overlap = %",im.filename,closest_det->jump_code,ol));
      if(closest_det->jump_code == 1)
      {
	Mat vis = im.RGB.clone();
	rectangle(vis,closest_det->BB.tl(),closest_det->BB.br(),Scalar(255,0,0));
	log_im("jumped_best_det_bb",vis);
      }
    }
					    

    // only take those which don't have a jump code.
    Mat viz = imageeq("",im.Z,false,false);
    Mat window_accum(viz.rows,viz.cols,DataType<float>::type,Scalar::all(0));
    DetectionSet takenDets;
    for(auto && det : dets)
      if(det->jump_code == 0)
      {
	for(int yIter = det->BB.tl().y; yIter < det->BB.br().y; ++yIter)
	  for(int xIter = det->BB.br().x; xIter < det->BB.br().x; ++xIter)
	    if(0 <= xIter && 0 <= yIter && xIter < window_accum.cols && yIter < window_accum.rows)
	      window_accum.at<float>(yIter,xIter)++;
	takenDets.push_back(det);
      }
    Mat accum_viz = imageeq("",window_accum,false,false);
    for(int rIter = 0; rIter < viz.rows; ++rIter)
      for(int cIter = 0; cIter < viz.cols; ++cIter)
      {
	uint8_t&c = viz.at<Vec3b>(rIter,cIter)[0];
	c = accum_viz.at<Vec3b>(rIter,cIter)[0];
      }    
    log_im("scanned_windows",viz);

    stats.print();
    log_file << "--enumerate_windows" << endl;
    return takenDets;
  }
}
