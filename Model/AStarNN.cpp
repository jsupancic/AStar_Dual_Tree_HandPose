/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "AStarNN.hpp"
#include "HeuristicTemplates.hpp"
#include <queue>
#include <map>
#include <unordered_map>
#include "ApxNN.hpp"
#include "LibHandMetaData.hpp"
#include "Annotation.hpp"
// requires OpenCV be be built with Intel's TBB?
#include "tbb/concurrent_unordered_set.h"
#include "tbb/concurrent_unordered_map.h"
#include <functional>
#include "ScanningWindow.hpp"

namespace std 
{
  size_t hash < pair<deformable_depth::TemplateId/*X*/,const deformable_depth::SearchTree*/*T*/> >::operator()(const pair<deformable_depth::TemplateId/*X*/,const deformable_depth::SearchTree*/*T*/> &x ) const
  {
    return std::hash<int>()(static_cast<int>(x.first)) ^ hash<const deformable_depth::SearchTree*>()(x.second);
  }

  size_t hash < deformable_depth::RotatedWindow >::operator()(const deformable_depth::RotatedWindow &x ) const
  {
    return std::hash<deformable_depth::DetectorResult>()(x.window) ^ std::hash<double>()(x.theta);
  }
}

namespace deformable_depth
{
  using namespace cv;
  using namespace std;

  ///
  /// represents and active A* Search   
  ///
  // volumetric_penalty
  static double volumetric_penalty(RotatedWindow parent_window,RotatedWindow current_window)
  {
    double parent_depth = parent_window.window->depth;
    RectAtDepth r1(parent_window.window->BB, parent_depth, parent_depth+params::obj_depth());    

    double current_depth = current_window.window->depth;
    RectAtDepth r2(current_window.window->BB,current_depth,current_depth+params::obj_depth());

    double penalty = 1 - r2.intersection(r1).volume()/(.5*r1.volume() + .5*r2.volume());
    return penalty;
  }

  // calc_correlation_parents
  double InformedSearch::calc_correlation_parents(long x0, long x_index, const SearchTree*T)
  {
    double best_cor = inf;
    volatile int parents_checked = 0;

    // get parents	
    auto eq_range = template_to_parent_window.equal_range(x_index);
    for(auto iter = eq_range.first; iter != eq_range.second; ++iter)
    {
      parents_checked++;
      auto && parent_window = iter->second;
      long parent_template_id = window_to_template_id.at(parent_window);
      // get the parent's correlation? 
      double cor = calc_correlation(parent_template_id,T);
      
      // add a volumetric penality to ensure optimality.
      RotatedWindow current_window = template_id_to_window.at(x0);
      cor += volumetric_penalty(parent_window,current_window);
      if(cor < best_cor and goodNumber(cor))
	best_cor = cor;

      double recur_cor = calc_correlation_parents(x0,parent_template_id,T);
      if(goodNumber(recur_cor))
	best_cor = std::min(best_cor,recur_cor);
    }

    if(!goodNumber(best_cor))
      return inf;
    else
      return best_cor;
  }

  double InformedSearch::calc_correlation_pyramid(long x_index, const SearchTree*T)
  {
    {   
      double max_cor;
      shared_ptr<NNTemplateType> XT = (*XS[x_index])();
      if(XT->resolution() == T->Templ.resolution())
      {
	template_evals++;    
	// get the XT from the cache...      
	// calculate the correlation
	//max_cor = T->Templ.cor(*XT,admissibility);
	max_cor = calc_correlation_simple(x_index,T);
      }
      else if(true)
      {
	max_cor = std::max<double>(max_cor,calc_correlation_parents(x_index,x_index,T));
      }
      else
      {
	max_cor = inf;
      }
      
      assert(goodNumber(max_cor));      
      return max_cor;
    }
  }

  double InformedSearch::bubble_correlation(long x0, long x_index,double raw_cor)
  {
    // update our cor
    RotatedWindow current_window = template_id_to_window.at(x_index);
    RotatedWindow window_0 = template_id_to_window.at(x0);
    double cor = raw_cor + volumetric_penalty(current_window,window_0);

    float&resp = template_id_to_window.at(x_index).window->resp;
    if(cor < resp or not goodNumber(resp))
    {
      resp = cor;
    
      // recursively update the parents;      
      auto eq_range = template_to_parent_window.equal_range(x_index);
      for(auto iter = eq_range.first; iter != eq_range.second; ++iter)
      {
	auto && parent_window = iter->second;
	long parent_template_id = window_to_template_id.at(parent_window);
	bubble_correlation(x_index,parent_template_id,raw_cor);
      }
    }
    return resp;
  }

  InformedSearch::Correlation InformedSearch::calc_correlation_simple(long x_index, const SearchTree*T)
  {
    // compute the leaf node's score
    shared_ptr<NNTemplateType> XT = (*XS[x_index])();
    assert(XT->resolution() == T->Templ.resolution());
    template_evals++;    
    double cor0 = T->Templ.cor(*XT,admissibility);

    // use any trained hs
    for(auto && h : T->heuristics)
    {
      auto new_cor = (*h)(*T,*XT);
      if(new_cor > cor0)
	cor0 = new_cor;
    }

    return Correlation{cor0,cor0};
  }

  InformedSearch::Correlation InformedSearch::calc_correlation_aobb(long x_index, const SearchTree*T)
  {
    RotatedWindow current_window = template_id_to_window.at(x_index);
    if(not current_window.window->supressed)
    {
      double cor0 = calc_correlation_simple(x_index,T);
      double parent_cor = calc_correlation_parents(x_index,x_index,T);      
      double cor = cor0;
      if(goodNumber(parent_cor) && parent_cor < cor)
	cor = parent_cor;

      // perculate the score up the parents
      bubble_correlation(x_index,x_index,cor);

      assert(goodNumber(cor));
      return Correlation{cor0,cor};
    }
    else
    {
      // deal with parent windows which can only come from the cache under this modality.
      double cache_loopup = template_id_to_window.at(x_index).window->resp;
      if(goodNumber(cache_loopup))
      {     
	return Correlation{inf,cache_loopup};
      }      
      else
	return Correlation{inf,inf};
    }
  }

  InformedSearch::Correlation InformedSearch::calc_correlation(long x_index, const SearchTree*T)
  {   
    //auto  max_cor = calc_correlation_aobb(x_index,T);
    auto max_cor = calc_correlation_simple(x_index,T);

    return max_cor;
  }

  static atomic<TemplateId> cache_id(0);  

  double InformedSearch::extract_image_template_random(const ImRGBZ& im,DetectorResult&window)
  {
    string alloc_id = std::to_string(cache_id++);
    double theta = sample_in_range(0,2*params::PI); 
    function<shared_ptr<NNTemplateType> ()> getFn = [this,&im,window,theta]()
	    {
              // generate the template		      
	      RotatedRect bb(rectCenter(window->BB),window->BB.size(),rad2deg(theta));
	      ImRGBZ im_roi = im.roi_light(bb);
	      return make_shared<NNTemplateType>(
		im_roi,window->depth,nullptr,bb);	     
	    };
    XFn xfn = [getFn,this,alloc_id]()
	    {
		    return x_template_cache.get(alloc_id,getFn);
	    };
    shared_ptr<XFn> xfn_ptr = make_shared<XFn>(xfn);
    XS.push_back(xfn_ptr);

    return theta;
  }

  double InformedSearch::extract_image_template_directed(const ImRGBZ& im,DetectorResult&window,size_t theta_index)
  {        
    double theta = interpolate_linear(theta_index,0,ROTS_TOTAL,0,2*params::PI);
    
    // make sure the rotation is in the cache.
    if(rotations.find(theta_index) == rotations.end())
    {
      // resize the thing
      Size doubleSize(4*im.cols(),4*im.rows());
      Rect big_rect(Point(-im.cols(),-im.rows()),doubleSize);
      ImRGBZ big = im(big_rect);
      
      // rotate about *new* center	    
      big_rect = Rect(Point(0,0),doubleSize);
      Point rot_about = rectCenter(big_rect);	    
      RotatedRect rr(rot_about,doubleSize,rad2deg(theta));
      rotations.insert({theta_index,big.roi_light(rr)});
    }

    // return a closure to extract template from the pre-rotated image.
    string alloc_id = std::to_string(cache_id++);
    function<shared_ptr<NNTemplateType> ()> getFn = [this,&im,window,theta_index,theta]()
	    {		    
	      Size doubleSize(4*im.cols(),4*im.rows());
	      Rect big_rect(Point(0,0),doubleSize);

	      // extract the ROI using the pre-rotated image...	      
	      ImRGBZ&pre_rotated = rotations.at(theta_index);
	      Mat rotMat = cv::getRotationMatrix2D(rectCenter(big_rect),rad2deg(theta),1); 
	      Point2d cropped_point = rectCenter(window->BB) + Point2d(im.cols(),im.rows());
	      Point2d new_center = point_affine(cropped_point,rotMat);
	      RotatedRect raw_bb(new_center,window->BB.size(),0);
	      ImRGBZ im_roi = pre_rotated.roi_light(raw_bb);
	      	      
	      // generate the template
	      RotatedRect rot_bb(rectCenter(window->BB),window->BB.size(),rad2deg(theta));
	      auto templ =  make_shared<NNTemplateType>(
		im_roi,window->depth,nullptr,rot_bb);
	      for(int iter = 0; iter < window->down_pyramid_count; ++iter)
		*templ = templ->pyramid_down();

	      if(window->down_pyramid_count == 0)
		log_im_decay_freq("extract_template",[&](){
		    Mat ImVis = imageeq("",pre_rotated.Z.clone(),false,false);
		    cv::rectangle(ImVis,window->BB.tl(),window->BB.br(),Scalar(0,0,255));
		    cv::circle(ImVis,cropped_point,5,Scalar(0,255,0));
		    cv::circle(ImVis,new_center,5,Scalar(255,0,0));
		    ImRGBZ t_old = im.roi_light(rot_bb);
		    vector<Mat> tvis = {imageeq("",t_old.Z),imageeq("",im_roi.Z),
					ImVis};
		    return tileCat(tvis);
		  });


	      return templ;
	    };
    XFn xfn = [getFn,this,alloc_id]()
	    {
		    return x_template_cache.get(alloc_id,getFn);
	    };
    shared_ptr<XFn> xfn_ptr = make_shared<XFn>(xfn);
    XS.push_back(xfn_ptr);

    return theta;
  }

  void InformedSearch::implement_image_template(DetectorResult&window,double theta)
  {
    size_t x_index = XS.size()-1;
    template_id_to_window.insert({x_index,RotatedWindow{window,theta}});
    // add all the parent windows to the multimap.    
    for(auto && parent_window : window->pyramid_parent_windows)
    {
      template_to_parent_window.insert({x_index,RotatedWindow{parent_window,theta}});
    }

    if(not window->supressed)
    {
      //assert(window->pyramid_parent_windows.size() > 0);
      shared_ptr<NNTemplateType> XT = (*XS.back())();
      if(XT->is_valid())
      {	
	double cost = -calc_correlation(x_index,&model.search_tree_root);
	AStarSearchNode nodeTop{cost,&model.search_tree_root,window->depth,x_index};
	init_frontier(nodeTop);
      }
    }
    else
    {
      window_to_template_id.insert({RotatedWindow{window,theta},x_index});
    }    
  }
  
  void InformedSearch::extract_image_template(const ImRGBZ& im,DetectorResult&window)
  {
    if(g_params.option_is_set("IMPLICIT_IN_PLANE_ROTATION"))
      {
      	for(int iter = 0; iter < ROTS_PER_WINDOW; ++iter)
	{
	  int theta_index = thread_rand() % ROTS_TOTAL;
	  double theta = interpolate_linear(theta_index,0,ROTS_TOTAL,0,2*params::PI);
	  theta = extract_image_template_directed(im,window,theta_index);
      	  //theta = extract_image_template_random(im,window);
	  implement_image_template(window,theta);
	}
      }
    else
      implement_image_template(window,extract_image_template_directed(im,window,0));
  }

  shared_ptr<NNTemplateType> InformedSearch::getX(TemplateId id)
  {
    return (*XS.at(id))();
  }

  size_t InformedSearch::numberXS()
  {
    return XS.size();
  }

  bool InformedSearch::suppressedX(TemplateId id)
  {
    return template_id_to_window.at(id).window->supressed;
  }

  void InformedSearch::init(DetectionFilter&filter)
  {
    // fill the frontier.
    NNSearchParameters params;
    params.stop_at_window = true;
    windows = model.detect_linear(im,filter,params);
    log_file << "Got " << windows.size() << " windows" << endl;
    for(auto && window : windows)
      if(window->supressed)
	extract_image_template(im,window);
    for(auto && window : windows)
      if(not window->supressed)
	extract_image_template(im,window);
    log_once(printfpp("InformedSearch::init DONE extracted %d templates (XS) from %s ... starting A* Search",
		      (int)XS.size(),im.filename.c_str()));
    
    // compute gt up
    hint = (filter.cheat_code.lock());
    gt_up = Vec3d(qnan,qnan,qnan);
    gt_normal = Vec3d(qnan,qnan,qnan);
    if(hint)
      fliplr = hint->leftP();
    if(hint && hasFullHandPose(*hint))
      {	
	Rect handBB = (hint->get_positives()["HandBB"]);	

	if(handBB != Rect())
	  {
	    map<string,Vec3d> set_keypoints, all_keypoints;
	    libhand::HandCameraSpec cam_spec;
	    libhand::FullHandPose hand_pose;
	    getFullHandPose(*hint,set_keypoints,all_keypoints,fliplr,cam_spec,hand_pose);    
	    if(!(all_keypoints.find("carpals") == all_keypoints.end()) and !(all_keypoints.find("Bone_002") == all_keypoints.end()))
	      {
		Vec3d carpal = all_keypoints.at("carpals");
		gt_center = all_keypoints.at("Bone_002");    
		Vec3d finger1joint1 = all_keypoints.at("finger1joint1");
		Vec3d lr_dir = finger1joint1 - gt_center;
		gt_up = gt_center - carpal; gt_up = gt_up / std::sqrt(gt_up.ddot(gt_up));
		gt_normal = lr_dir.cross(gt_up);
		gt_normal = gt_normal / std::sqrt(gt_normal.ddot(gt_normal));
		if(!fliplr) //??
		  {
		    gt_up[2] = -gt_up[2];
		    gt_normal[2] = -gt_normal[2];
		    gt_center[2] = -gt_center[2];
		  }
	      }
	  }
      }      
  }
  
  InformedSearch::InformedSearch(const ImRGBZ& im,const ApxNN_Model&model,DetectionFilter&filter) : 
    parts_per_ex(model.parts_per_ex),
    model(model),
    template_evals(0),
    DEBUG_invalid_Xs_from_frontier(0),
    im(im),
    x_template_cache(fromString<int>(g_params.require("NN_XTEMPL_CACHE_SIZE"))),
    admissibility(fromString<double>(g_params.require("NN_ADMISSIBILITY"))),
    upper_bound(inf),
    boundings(0),
    rejected_dets(0),
    iterations(0),
    last_frame_made(0)
  {
    log_once(safe_printf("InformedSearch::InformedSearch admissibility = %",admissibility));
    progress_last_reported = std::chrono::system_clock::now();
  }

  DetectorResult InformedSearch::find_best(const AStarSearchNode&active,RotatedRect&rawWindow)
  {
    // get the detection
    shared_ptr<NNTemplateType> XTempl = (*XS[active.X_index])(); 

    // emitt the detection
    auto det = make_shared<Detection>();	    
    rawWindow = XTempl->getExtractedFrom();
    det->rawBB = rawWindow;
    log_file << "iteration emit angle = " << XTempl->getExtractedFrom().angle << endl;
    auto rotWindow = rawWindow; rotWindow.angle = 0;
    Rect window = rotWindow.boundingRect();
    det->BB = window;
    det->depth = active.depth;
    det->resp = -active.cost;
    // generate parents
    auto eq_range = template_to_parent_window.equal_range(active.X_index);
    for(auto iter = eq_range.first; iter != eq_range.second; ++iter)
    {
      det->pyramid_parent_windows.push_back(iter->second.window);
    }        
    // generate parts
    string tree_id = active.tree->get_uuid();
    auto part_iter = parts_per_ex.find(tree_id);
    if(part_iter != parts_per_ex.end())
    {
      auto parts = part_iter->second;
      set_det_positions(active.tree->Templ,
			active.tree->pose_uuid,
			XTempl->getExtractedFrom(),parts,det,active.depth);
      return det;
    }
    else
    {
      string err_mesg = safe_printf("error: couldn't lookup %",tree_id);
      log_file << err_mesg << endl;
      throw std::runtime_error(err_mesg);
    }
  }

  // returns true if we are done.
  bool InformedSearch::emit_best(const AStarSearchNode&active,DetectionSet&dets)
  {
    RotatedRect rawWindow;
    auto det = find_best(active,rawWindow);
    // compute the affine transform
    Mat aff = cv::getRotationMatrix2D(Point2f(0,0),(rawWindow.angle),1);
    aff.convertTo(aff,DataType<float>::type);
    if(model.accept_det(*hint,*det,vec_affine(gt_up,aff),vec_affine(gt_normal,aff),fliplr))
    {	  
      dets.push_back(det);
      return false;
    }
    else
    {
      rejected_dets++;      
      log_file << (safe_printf("emit_best % immediate parents",det->pyramid_parent_windows.size())) << endl;
      log_once(safe_printf("InformedSearch::emit_best warning rejected one for %",im.filename));
      return true;    
    }
  }

  void AStarSearch::init_frontier(const AStarSearchNode&node)
  {
    frontier.push(node);
  }
    
  AStarSearch::~AStarSearch()
  {
  }

  AStarSearch::AStarSearch(const ImRGBZ& im,const ApxNN_Model&model,DetectionFilter&filter) : 
    InformedSearch(im,model,filter)
  {
  }

  void AStarSearch::iteration_admissible(shared_ptr<NNTemplateType>&XTempl,const AStarSearchNode&active)
  {
    // pre-filter?
    if(active.cost <= upper_bound)	  
    {
      // update boundings?
      update_random_upper_bound(active.tree,*XTempl);
	
      iteration_inadmissible(XTempl,active);
    }
  }

  void AStarSearch::iteration_inadmissible(shared_ptr<NNTemplateType>&XTempl,const AStarSearchNode&active)
  {
    // expand the node
    for(auto & child : active.tree->children)
    {
      Correlation cor = calc_correlation(active.X_index, child.get());
      //double cost = -cor.exact;//
      double cost = std::max(-cor.exact,active.cost);
      if(cost <= upper_bound)
	frontier.push(AStarSearchNode{
	    cost,child.get(),active.depth,active.X_index});
      else
	boundings++;
    }
  }

  void AStarSearch::make_frame(const AStarSearchNode&active_ref)
  {
    if(not g_params.option_is_set("DEBUG_ASTAR_VIDEO"))
      return;

    AStarSearchNode active = active_ref;
    auto boundings = this->boundings;
    auto template_evals = this->template_evals;
    auto rejected_dets = this->rejected_dets;
    FrameFn frameFn = [this,active,boundings,template_evals,rejected_dets]()
    {
      // visualize the matched data 
      shared_ptr<NNTemplateType> XTempl = (*this->XS[active.X_index])(); 
      RotatedRect rawWindow = XTempl->getExtractedFrom();
      Rect bb = rectFromCenter(rawWindow.center,Size(rawWindow.size.width,rawWindow.size.height));
      Mat frame = im.RGB.clone();
      cv::rectangle(frame,bb.tl(),bb.br(),Scalar(0,255,0));
      
      // visualize the template
      auto&T = active.tree->Templ;
      Vec3i tRes = T.resolution();
      Mat rotMat = cv::getRotationMatrix2D(Point(tRes[0]/2,tRes[1]/2),-rawWindow.angle,1);    
      Mat mean = T.getTIm().clone();  
      Size tVisSize(2*mean.cols,2*mean.rows);
      cv::warpAffine(mean,mean,rotMat,mean.size()); cv::resize(mean,mean,tVisSize);
      Mat near = T.getTNear().clone();
      cv::warpAffine(near,near,rotMat,near.size()); cv::resize(near,near,tVisSize);
      Mat far  = T.getTFar().clone(); 
      cv::warpAffine(far ,far ,rotMat,far .size()); cv::resize(far ,far ,tVisSize);
      Mat VisTs = horizCat(imageeq("",mean,false,false),
			   horizCat(imageeq("",near,false,false),imageeq("",far,false,false)));
      
      // visualize a message
      long possible_evals = ROTS_PER_WINDOW*windows.size()*model.allTemplates.size();
      string message = safe_printf(
	"(boundings % ) (template_evals %) (possible evals %) (rejected dets = %)",
	boundings,template_evals,possible_evals,rejected_dets);
      Mat txt = vertCat(image_text(im.filename),image_text(message));
      cv::resize(txt,txt,Size(),.5,.5);
      
      // update the video    
      frame = vertCat(frame,vertCat(VisTs,txt));
      return frame;
    };
    debug_video.push_back(frameFn);
  }

  static bool astar_add_bounding()
  {
    static int bounded = -1;
    if(bounded == -1)
    {
      bounded = g_params.option_is_set("ASTAR_ADD_BOUNDING");
    }
    return bounded;
  }

  // exapnd the min-cost node from the frontier and add its
  // children back into the frontier or terminate the search
  // if it is a terminal node.
  // RETURNS: If we should continue to another iteration. False means we're done.
  bool AStarSearch::iteration(DetectionSet&dets)
  {   
    bool done = frontier.empty();
    iterations++;
    if(not done)
    {
      const AStarSearchNode active = frontier.top();
      assert(0 <= active.X_index and active.X_index < XS.size());
      shared_ptr<NNTemplateType> XTempl = (*XS[active.X_index])(); 
      frontier.pop();
      
      if(!XTempl->is_valid())
      {
	++DEBUG_invalid_Xs_from_frontier;
      }
      else if(active.tree->children.size() == 0)
      {
	done = not emit_best(active,dets);
	assert(not done or frontier.empty() or dets.size() > 0);
      }
      else
      {
	if(admissibility >= 1 and astar_add_bounding())
	  iteration_admissible(XTempl,active);
	else
	  iteration_inadmissible(XTempl,active);
      }      
      done |= frontier.empty();

      if(done or last_frame_made < iterations)
      {
	 make_frame(active);
	 last_frame_made = iterations + 1000;
      }
    }

    // log our progress
    if(done or (std::chrono::system_clock::now() > std::chrono::seconds(15) + progress_last_reported))
    {
      if(done)
	assert(frontier.empty() or dets.size() > 0);
      write_progress(done); 
    } 
    return not done;
  }    

  void InformedSearch::update_random_upper_bound(const SearchTree*tree,NNTemplateType&XTempl)
  {
    int num_children = tree->children.size();
    if(num_children == 0)
    {
      double candidate_ub = -tree->Templ.cor(XTempl,admissibility);
      if(candidate_ub < upper_bound)
      {
	log_once(printfpp("%s ub %f => %f",im.filename.c_str(),upper_bound,candidate_ub));
	upper_bound = candidate_ub;
      }
    }
    else
    {      
      update_random_upper_bound(tree->children[thread_rand()%num_children].get(),XTempl);
    }
  }

  void AStarSearch::write_progress(bool done) 
  {
    // write it out.
    log_file << "++AStarSearch::write_progress" << endl;
    log_file << safe_printf("DONE=% % (frontier.size() = %)",done,im.filename,frontier.size()) << endl;
    InformedSearch::write_progress(done);
  }

  void InformedSearch::write_progress(bool done)
  {
    // write some textural information about the progress of the search
    long possible_evals = ROTS_PER_WINDOW*windows.size()*model.allTemplates.size();
    log_file << "DONE=" << done << " templ evals = " << template_evals << " of " << 
      possible_evals  << endl;

    log_once(safe_printf("DONE=% % failed frontier features: %",done,im.filename,DEBUG_invalid_Xs_from_frontier));
    log_once(safe_printf("DONE=% % (boundings % ) (template_evals %) (possible evals %) (rejected dets = %)",done,im.filename,boundings,template_evals,possible_evals,rejected_dets));
    progress_last_reported = std::chrono::system_clock::now();

    // generate some visual information (will slow things down, but look pretty)
    if(done and g_params.option_is_set("DEBUG_ASTAR_VIDEO"))
    {
      int frame_targets = 1000;
      double step_size = static_cast<double>(debug_video.size())/frame_targets;
      log_file << safe_printf("% debug_video.size = % step_size = %",im.filename,debug_video.size(),step_size) << endl;
      Size outSize(640,480);
      string video_file = params::out_dir() + "/" + uuid() + ".avi";
      VideoWriter video_out(video_file,CV_FOURCC('F','M','P','4'),15,outSize,true);
      for(double frameIter = 0; frameIter < debug_video.size(); frameIter += step_size)
      {
	int frame_index = clamp<int>(0,frameIter,debug_video.size());
	auto frame = debug_video.at(frame_index)();
	cv::resize(frame,frame,outSize);
	bool last_frame = (frameIter + 2 >= debug_video.size());
	for(int iter = 0; iter < (last_frame?30:2); ++iter)
	{
	  video_out.write(frame);
	  log_file << safe_printf("% wrote frame %",im.filename,frameIter) << endl;
	}
      }
      log_file << "InformedSearch::write_progress frame count = " << 30 * debug_video.size() << endl; 
      video_out.release();
    }
  }

  InformedSearch::~InformedSearch()
  {
    write_progress(true);
  }
}

