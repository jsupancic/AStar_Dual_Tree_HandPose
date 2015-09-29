/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "VolumetricNN.hpp"
#include "Orthography.hpp"
#include "SphericalVolumes.hpp"
#include "ScanningWindow.hpp"
#include "Colors.hpp"
#include "LibHandRenderer.hpp"
#include "Probability.hpp"

namespace deformable_depth
{
  TemplateUniScalar::TemplateUniScalar(shared_ptr<Mat> T, shared_ptr<MetaData> datum, float theta) :
    T(T), datum(datum), theta(theta)
  {
  }
  
  VolumetricNNModel::VolumetricNNModel()
  {
  }

  static void filterImages_one_depth(
    FilterResults&result,SphericalOccupancyMap&SOM,Visualization&vis,
    multimap<double,shared_ptr<TemplateUniScalar> >&templates_by_depth,
    double zIter)
  {
    Mat zSlice = SOM.slice_z(zIter,zIter + params::obj_depth());

    // collect templates
    float z_min = zIter - params::obj_depth()/2;
    //float z_min = zIter;
    auto it_begin = templates_by_depth.lower_bound(z_min);
    float z_max = zIter + params::obj_depth()/2;
    auto it_end = templates_by_depth.upper_bound(z_max);
    //if(it_end != templates_by_depth.end())
    //++it_end;
    vector<Mat> TsAtZ;
    vector<double> TsZs;
    vector<double> TsMaxs;
    for(auto iter = it_begin; iter != it_end; ++iter)
    {       
      Mat TVis = iter->second->T->clone(); //.vis_high_res();//iter->second.getTIm();	
      for(int rIter = 0; rIter < TVis.rows; ++rIter)
	for(int cIter = 0; cIter < TVis.cols; ++cIter)
	  if(!goodNumber(TVis.at<float>(rIter,cIter)))
	    TVis.at<float>(rIter,cIter) = params::obj_depth();
      log_im_decay_freq("activeT",[&](){return imageeq("",TVis,false,false);});

      TsAtZ.push_back(TVis);
      double max_resp = ((params::obj_depth())*TVis.size().area());
      TsMaxs.push_back(max_resp);
      TsZs.push_back(iter->first);
    }

    // 
    assert(false);
    vector<Mat> respss;// = matchTemplatesL1(zSlice,TsAtZ,30*30,params::obj_depth());      
    for(int iter = 0; iter < TsAtZ.size(); ++iter)
    {
      respss.push_back(
	matchTemplateL1(zSlice,TsAtZ.at(iter),30*30,params::obj_depth()));
    }    

    // find the best response map
    int index = 0;
    for(auto iter = it_begin; iter != it_end; ++iter, ++index)
    {
      float max_resp = TsMaxs.at(index);
      Mat TVis = TsAtZ.at(index);
      Mat resps = (max_resp - respss.at(index))/max_resp;
      float tz = TsZs.at(index);
      
      FilterResult cur_result;
      cur_result.depth = zIter;
      cur_result.template_depth = tz;
      cur_result.filter = iter->second;
      cur_result.extrema = extrema(resps);
      Rect detBB = rectFromCenter(cur_result.extrema.maxLoc,TVis.size());
      if(result.filters_by_resps.empty() ||
	 result.filters_by_resps.rbegin()->second.extrema.max < cur_result.extrema.max)
      {
	result.top_resps = resps;
	result.top_evidence = zSlice;
      }
      result.filters_by_resps.insert(std::pair<double,FilterResult >({cur_result.extrema.max,cur_result}));
    }      
    
    log_once(safe_printf("note: z = % templates = %",zIter,TsAtZ.size()));    
  }

  FilterResults VolumetricNNModel::filterImages(
    DetectionFilter&filter,
    const ImRGBZ&im, SphericalOccupancyMap&SOM,Visualization&vis,
    multimap<double,shared_ptr<TemplateUniScalar> >&templates_by_depth) const
  {
    FilterResults result;
    if(g_params.option_is_set("CHEAT_DEPTH"))
    {
      // get ground truth depth
      shared_ptr<MetaData> cheat = filter.cheat_code.lock();
      assert(cheat);
      VolTrainingExample gt = do_training_extract(cheat);

      // run filters at said depth
      double zIter = gt.z_min;
      log_file << "test template depth = " << zIter << endl;
      filterImages_one_depth(result,SOM,vis,templates_by_depth,zIter);
    }
    else
    {      
      ProgressBar zBar(string("zBar") + im.filename,params::MAX_Z() - params::obj_depth());
      for(double zIter = params::MIN_Z(); zIter < params::MAX_Z() - params::obj_depth(); zIter += 1)
      {
	zBar.set_progress(zIter);
	filterImages_one_depth(result,SOM,vis,templates_by_depth,zIter);
      }
    }

    return result;
  }

  static void visualize(const ImRGBZ&im,const FilterResults&fr,Visualization&vis)
  {       
    // visualize the results
    auto fIter = fr.filters_by_resps.rbegin(); 
    for(int iter = 1; fIter != fr.filters_by_resps.rend() ; iter *= 2 )
    {
      TemplateUniScalar&TUS = *fIter->second.filter;
      Mat TVis = imageeq("",*TUS.T,false,false);      
      TVis = vertCat(imVGA(TVis),image_text(safe_printf("resp = %",fIter->first)));
      vis.insert(imVGA(TVis),safe_printf("filter_%_",to_string(iter,10)));
      
      Rect detBB = rectFromCenter(fIter->second.extrema.maxLoc,TVis.size());
      Mat DVis = imageeq("",im.Z,false,false);
      cv::rectangle(DVis,detBB.tl(),detBB.br(),toScalar(BLUE),5);
      string prefix = safe_printf("Detection_%_",to_string(iter,10));
      vis.insert(DVis,prefix);    
      vis.insert(
	image_format("% det_depth = % templ_depth = %",prefix,fIter->second.depth,fIter->second.template_depth),
	"depths");
      
      for(int i = 0; i < iter ; ++i)
      {
	++fIter;
	if(fIter == fr.filters_by_resps.rend())
	  goto END_VIS;
      }
    }
    END_VIS:
    vis.insert(imageeq("",fr.top_resps,false,false),"topResps");
    vis.insert(imageeq("",fr.top_evidence,false,false),"top_evidence");
  }

  static DetectorResult report_det(FilterResults&fr)
  {
    //
    
    // set the HandBB
    auto det = make_shared<Detection>();
    Extrema max_extrema = extrema(fr.top_resps);
    FilterResult&top_result = fr.filters_by_resps.rbegin()->second;
    Rect detBB = rectFromCenter(max_extrema.maxLoc,top_result.filter->T->size());    
    det->resp = max_extrema.max;
    det->BB = detBB;

    // set the fingers
    auto datum = top_result.filter->datum;
    Rect handBB = datum->get_positives().at("HandBB");
    RotatedRect r_detBB(max_extrema.maxLoc,handBB.size(),-rad2deg(top_result.filter->theta));
    RotatedRect gt_bb(rectCenter(handBB),handBB.size(),0);
    auto poss = datum->get_positives();
    set_det_positions2(gt_bb, datum,datum->get_filename(),
		      r_detBB, poss,// X BB
		      det,top_result.depth);

    // set the blob?
    Mat Z = datum->load_im()->Z;
    if(!Z.empty())
    {
      Mat IAT = affine_transform_rr(r_detBB,gt_bb);    
      auto im = datum->load_im();
      det->blob = Mat(im->rows(),im->cols(),DataType<float>::type,Scalar::all(qnan));
      for(int yIter = 0; yIter < im->rows(); yIter++)
	for(int xIter = 0; xIter < im->cols(); xIter++)
	{
	  Point p = Point(xIter,yIter);
	  Point q = point_affine(p,IAT);
	  float z = at(Z,clamp<int>(0,q.y,im->rows()-1),clamp<int>(0,q.x,im->cols()-1));
	  if(goodNumber(z))
	  {
	    det->blob.at<float>(yIter,xIter) = z;
	  }
	}
    }
    else
      log_once(safe_printf("no seg for %",datum->get_filename()));
    
    return det;
  }

  // model virtuals
  DetectionSet VolumetricNNModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    DetectionSet result;
    
    //OrthographicOccupancyMap omap(im);
    //omap.vis().write(im.filename);

    SphericalOccupancyMap SOM(im);
    Visualization vis = SOM.vis();

    // get the  handBB (for debugging)
    shared_ptr<MetaData> cheat = filter.cheat_code.lock();
    if(cheat and g_params.option_is_set("CHEAT_HAND_BB"))
    {
      vector<AnnotationBoundingBox> poss = metric_positive(*cheat);
      double handZ = poss.begin()->depth;
      string message = safe_printf("handZ = ",handZ);
      vis.insert(image_text(message),"message");
    }
       
    // 
    auto active_filters = templates_by_depth;
    FilterResults fr = filterImages(filter,im, SOM,vis,active_filters);

    // report the detection
    if(!fr.filters_by_resps.empty())
    {
      auto res = report_det(fr);
      if(res)
      {
	result.push_back(res);
	Mat blob = imageeq("",res->blob,false,false);
	Mat evidence = imageeq("",fr.top_evidence,false,false);
	Mat match(blob.rows,blob.cols,DataType<Vec3b>::type,toScalar(BLACK));
	for(int yIter = 0; yIter < blob.rows; yIter++)
	  for(int xIter = 0; xIter < blob.cols; ++xIter)
	  {
	    if(blob.at<Vec3b>(yIter,xIter) != INVALID_COLOR)
	      match.at<Vec3b>(yIter,xIter)[0] = blob.at<Vec3b>(yIter,xIter)[0];
	    if(evidence.at<Vec3b>(yIter,xIter) != INVALID_COLOR)
	      match.at<Vec3b>(yIter,xIter)[1] = evidence.at<Vec3b>(yIter,xIter)[1];
	  }
	vis.insert(match,"match");    
      }
    }
    
    // filter the image to find the detections    
    visualize(im,fr,vis);
    Mat zVis = imageeq("",im.Z,false,false);
    vis.insert(visualize_result(im,zVis,result),"ModelViz");
    vis.write("VolNN_SOM");
    
    return result;
  }

  ///
  /// TRAINING
  ///
  
  void VolumetricNNModel::do_training_write(VolTrainingExample&t,const shared_ptr<MetaData>&datum)
  {
    lock_guard<mutex> l(monitor);
    
    if(g_params.option_is_set("IMPLICIT_IN_PLANE_ROTATION"))
    {
      shared_ptr<ImRGBZ> im = datum->load_im();
      float z = t.z_min + params::obj_depth()/2;
      auto metric_size = MetricSize(datum->get_filename());
      // Rect metric_bb = im->camera.bbForDepth(z,im->Z.size(),
      // 					     im->rows()/2,im->cols()/2,
      // 					     metric_size.height,
      // 					     metric_size.width,
      // 					     false);
      Size size = im->camera.imageSizeForMetricSize(z,metric_size);
      log_once(safe_printf("do_training_write got metric_size = % image_size = %",metric_size,size));
      Mat tPad = imtake(*t.t,size,BORDER_CONSTANT,Scalar::all(qnan));
      for(double theta = 0; theta < 2*params::PI; theta += 2*params::PI/20)
	//for(int iter = 0; iter < 20; ++iter)
      {
	//double theta = sample_in_range(0,2*params::PI);
	// imrotate_tight(*t.t,theta)
	Mat zRot = imrotate(tPad,theta);
	auto tRot = make_shared<Mat>(zRot);
	shared_ptr<TemplateUniScalar> TUS = make_shared<TemplateUniScalar>(tRot,datum,theta);
	templates_by_depth.insert({t.z_min,TUS});
	log_im_decay_freq("templ_out_rot_1_",[&](){return imageeq("",*tRot,false,false);});
      }
    }
    else
    {
      shared_ptr<TemplateUniScalar> TUS = make_shared<TemplateUniScalar>(t.t,datum,0);
      templates_by_depth.insert({t.z_min,TUS});
      log_im_decay_freq("templ_out_1_",[&](){return imageeq("",*t.t,false,false);});
    }
    //log_im_decay_freq("templ_in2",[&](){return imageeq("",*templates_by_depth.at(z_min),false,false);});    
  }
  
  VolTrainingExample VolumetricNNModel::do_training_extract(const shared_ptr<MetaData>&datum) const
  {
    Rect handBB = datum->get_positives().at("HandBB");
    shared_ptr<ImRGBZ> im = datum->load_im();
    const ImRGBZ cim = *im;
    vector<float> depths = manifoldFn_discrete_sparse(cim, handBB, 1);

    vector<VolTrainingExample> exs;
    ostringstream depth_message;
    for(auto z_min : depths)
    {
      shared_ptr<Mat> t = make_shared<Mat>(im->Z.clone());	
      *t = imroi(*t,handBB);
      if(!goodNumber(z_min))
      {
	ostringstream message;
	message << "warning reject z_min: " << z_min << " " << *t;
	log_file << message << endl;
	log_im_decay_freq("rej_zmin",[&](){return image_datum(*datum);});
	return{nullptr,static_cast<float>(qnan)};
      }
      *t = *t - z_min;
      medianBlur(*t,*t,5);
      double area = 0;
      vector<double> depth_counts(params::obj_depth() + 2,0);
      for(int yIter = 0; yIter < t->rows; yIter++)
	for(int xIter = 0; xIter < t->cols; xIter++)
	{
	  float&tz = t->at<float>(yIter,xIter);
	  //float tn = clamp<float>(0,tz,params::obj_depth());
	  //if(tz == tn)
	  //area++;
	  //tz = tn;

	  if(tz <= 0)
	  {
	    depth_counts.at(0)++;
	    tz = 0;
	  }
	  else if(tz >= params::obj_depth())
	  {
	    depth_counts.back()++;
	    tz = params::obj_depth();
	  }
	  else
	  {
	    depth_counts.at(std::floor(tz + 1))++;
	    area++;
	  }
	}      
      log_im_decay_freq("templ_in1",[&](){return imageeq("",*t,false,false);});
      depth_message << safe_printf(" [z_min = % area = %] ",z_min,area);

      exs.push_back({t,z_min,area,shannon_entropy(depth_counts)});
    }

    std::sort(exs.begin(),exs.end());
    if(exs.size() == 0)
      return {nullptr,static_cast<float>(qnan), static_cast<float>(qnan)};
    else
    {
      depth_message << " chose z_min = " << exs.back().z_min << " with area = " << exs.back().area <<
	" with entropy = " << exs.back().entropy;
      log_file << depth_message.str() << endl;
      return exs.back();
    }
  }
  
  void VolumetricNNModel::do_training(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    static mutex m; lock_guard<mutex> l(m);
    // load some templates?
    //ExtractedTemplates extracted_templates = extract_templates(training_set,train_params);
    //log_file << "extracted templates count = " << extracted_templates.allTemplates.size() << 
    //" from " << training_set.size() << endl;

    TaskBlock do_training_block("do_training");
    for(auto && datum : training_set)
    {
      do_training_block.add_callee([&,this,datum]()
				   {
				     Rect handBB = datum->get_positives().at("HandBB");
				     if(handBB.size().area() > 0 ) 
				     {
				       auto t = this->do_training_extract(datum);
				       if(t.t)
				       {					 
					 this->do_training_write(t,datum);
				       }
				     }
				     else
				     {
				       shared_ptr<ImRGBZ> im = datum->load_im();
				       log_file << "warning reject template (bb) " << im->filename << endl;
				       log_im_decay_freq("rej_templ_bb",[&](){return image_datum(*datum);});
				     }
				   });
    }
    do_training_block.execute();
    
    log_file << "loaded templates: " << templates_by_depth.size() << endl;
    for(auto && T : templates_by_depth)
    {
      if(thread_rand() % templates_by_depth.size() < 100)
      {
	auto TUS = T.second;
	shared_ptr<ImRGBZ> im = TUS->datum->load_im();
	vector<Mat> viss{eq(*TUS->T),eq(im->Z),image_text(safe_printf("z_min = %",T.first))};
	log_im_decay_freq("templ_out_2_",[&](){return tileCat(viss);});
	log_file << " train template depth = " << T.first << endl;
      }
    }
  }

  
  
  void VolumetricNNModel::train_on_test(
    vector<shared_ptr<MetaData>>&training_set,
    TrainParams train_params)
  {
    // compute distance to the gt set.
    //VolumetricNNModel gt_model;
    //gt_model.do_training(training_set,train_params);
    //do_training(training_set,train_params);
  }

  void VolumetricNNModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    //do_training(training_set,train_params);
  }
  
  Mat VolumetricNNModel::show(const string&title)
  {
    return image_text("w00p");
  }

  void VolumetricNNModel::load_templates()
  {
    
  }
}
