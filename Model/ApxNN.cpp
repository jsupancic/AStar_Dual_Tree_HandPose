/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ApxNN.hpp"
#include "Detection.hpp"
#include <opencv2/opencv.hpp>
#include "OcclusionReasoning.hpp"
#include "RigidTemplate.hpp"
#include "Faces.hpp"
#include "Log.hpp"
#include "util.hpp"
#include "Cache.hpp"
#include "ColorModel.hpp"
#include "ThreadPoolCXX11.hpp"
#include <boost/filesystem.hpp>
#include <queue>
#include "LibHandMetaData.hpp"
#include "Annotation.hpp"
#include "AStarNN.hpp"
#include "ScanningWindow.hpp"
#include "Colors.hpp" 

namespace deformable_depth
{
  using namespace cv;   
        
  ApxNN_Model::ApxNN_Model(Size gtSize, Size ISize)
  {
  }
  
  ApxNN_Model::ApxNN_Model()
  {
  }

  DetectorResult ApxNN_Model::detect_at_position
    (NNTemplateType&X,Rect orig_bb,float depth,Mat&respIm,
     Mat&invalid,EPM_Statistics&stats) const
  {
    //static mutex m; lock_guard<mutex> l(m);
    //cout << "rootTemplate " << rootTemplate << endl;
    //cout << "fT " << fT << endl;
    //cout << "prod = " << prod << endl;
    stats.count("Good");
    DetectorResult det(new Detection());
    det->resp = -inf;
    det->depth = depth;
        
    // optimize responce over exemplars
    for(auto && Tex : allTemplates)
    {
      auto parts = parts_per_ex.at(Tex.second.getMetadata()->get_filename());
      
      double prod = Tex.second.cor(X);
      if(prod > det->resp)
      {
	RotatedRect rotOrigBB(rectCenter(orig_bb),orig_bb.size(),0);
	set_det_positions(Tex.second,Tex.first,rotOrigBB,parts,det,depth);
	det->resp = prod;
      }
    }
      
    // finalize
    Point2i center = rectCenter(orig_bb);
    assert(goodNumber(det->resp));
    respIm.at<float>(center.y,center.x) = 
      std::max(det->resp,respIm.at<float>(center.y,center.x));
    invalid.at<uchar>(center.y,center.x) = false;	      
    return det;    
  }

 
  // Even when using smarter search strategies (eg A*) this produces the BB candidates
  DetectionSet ApxNN_Model::detect_linear(const ImRGBZ& im, DetectionFilter filter,NNSearchParameters params) const
  {
    // for linear search, do the following at each position
	// RotatedRect bb(rectCenter(orig_bb),orig_bb.size(),0);
	// NNTemplateType X = NNTemplateType(im.roi_light(bb),depth,nullptr,bb);
	// if(X.getTIm().empty())
	// {
	//   stats.count("spearman template extraction failed");
	//   continue;
	// }      
	// log_im_decay_freq("SubImageTemplate",X.getTIm());

	// dets.push_back(detect_at_position(X,orig_bb,depth,respIm,invalid,stats));

    assert(params.stop_at_window);
    filter.manifoldFn = this->manifoldFn;
    auto windows = enumerate_windows(im,filter);
    if(windows.size() == 0)
      log_file << "warning: no valid windows for frame " << im.filename << endl;

    lock_guard<mutex> l(monitor);
    respImages[im.filename] = Mat(im.rows(),im.cols(),DataType<Vec3b>::type,Scalar::all(0));
    // here we log the responce image...
    //respIm = fillHoles<float>(respIm,invalid,5);
    //Mat vis_resp = imageeq("",respIm,false,false);
    //Mat vis_depth = imageeq("",im.Z,false,false);
    //log_im("resp_im",vertCat(vis_depth,vis_resp));

    
    return windows;
  }

  bool ApxNN_Model::accept_det(MetaData&hint,Detection&det,Vec3d gt_up, Vec3d gt_norm, bool flip_lr) const
  {
    // ensure normalization
    gt_up = gt_up/std::sqrt(gt_up.ddot(gt_up));
    gt_norm = gt_norm/std::sqrt(gt_norm.ddot(gt_norm));

    // lock the hint
    shared_ptr<MetaData>&exemplar = (det.exemplar);

    //Cheat with angles
    // we need gt pose for orientation cheats
    if(g_params.option_is_set("CHEAT_HAND_ORI") and exemplar and goodNumber(gt_up[0]))
    {
      map<string,AnnotationBoundingBox> poss = exemplar->get_positives();
      string meta_name = exemplar->get_filename();
      Vec3d v2 = poss["HandBB"].up; v2 = v2 / std::sqrt(v2.ddot(v2));
      double dot = gt_up.ddot(v2);
      double theta = rad2deg(std::acos(dot));
      double phi = rad2deg(std::acos(poss["HandBB"].normal.ddot(gt_norm)));
      log_file << "theta1(dot) = " << theta << "(" << dot << ")" << " from " << meta_name 
    	       << "phi = " << phi << endl;
      //return theta < 45 and phi < 45;
      //return (180 - phi) < 45 and theta < 45;
      if(phi < 45 or theta > 45)
	return false;
    }

    // Cheat w/ left vs. right hand
    //shared_ptr<MetaData>&exemplar = (det.exemplar);
    if(g_params.option_is_set("CHEAT_HAND_LR") and exemplar)
    {
      log_file << safe_printf
	("% flip_lr[%] == exemplar->leftP()[%]",hint.get_filename().c_str(),(int)flip_lr,(int)exemplar->leftP()) << endl;
      if(flip_lr != exemplar->leftP())
	return false;
    }
    return true;
  }

  DetectionSet ApxNN_Model::detect_Astar(const ImRGBZ& im, DetectionFilter filter) const
  {   
    DefaultSearchAlgorithm search(im,*this,filter);
    search.init(filter);
    DetectionSet result;

    // pop nodes and expand the frontier until we find a goal state (full hand pose)
    auto search_start = std::chrono::system_clock::now();
    while(search.iteration(result))
      ;
    
    //assert(false);
    if(result.size() == 0)
      log_file << "warning: no valid detections for frame " << im.filename << endl;
    auto search_stop = std::chrono::system_clock::now();
    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(search_stop - search_start);
    log_file << "AStarSearch " <<  im.filename << " took " << duration.count() << " milliseconds" << endl;

    // log the max resp
    result = sort(result);
    double best_resp = 0;
    if(result.size() > 0)
      best_resp = result.front()->resp;
    log_file << "AStarSearch " << im.filename << " best_resp = " << best_resp << endl;

    return result;
  }
  
  DetectionSet ApxNN_Model::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    log_once(printfpp("+ApxNN_Model::detect %s",im.filename.c_str()));   

    DetectionSet dets = detect_Astar(im,filter);
    //return detect_linear(im,filter);
    
    dets = sort(dets);    
    log_once(safe_printf("-ApxNN_Model::detect % %",im.filename.c_str(),dets.size()));
    return dets;    
  }

  Mat ApxNN_Model::show(const string& title)
  {
    return Mat();
  }
  
  // collect all the positive examples from the training set...
  map<string,NNTemplateType> ApxNN_Model::train_extract_templates(
    vector< shared_ptr< MetaData > >& training_set, TrainParams train_params)
  {
    // setup the training set
    vector<shared_ptr<MetaData> >  positive_set;
    vector<shared_ptr<MetaData> >  negtive_set;    
    split_pos_neg(training_set,positive_set,negtive_set);
    //this->training_set = training_set;
    int npos = fromString<int>(g_params.require("NPOS"));
    this->training_set = random_sample<shared_ptr<MetaData> >(positive_set,npos,19860);
    for(auto && datum : training_set)
      log_once(safe_printf("selected positive example: %",datum->get_filename()));
    Mat Tmax;

    // extract templates from the training set 
    auto extracted = deformable_depth::extract_templates(this->training_set,train_params);
    allTemplates = extracted.allTemplates;
    parts_per_ex = extracted.parts_per_ex;
    sources      = extracted.sources;
    return allTemplates;
  }
  
  void ApxNN_Model::train(
    vector< shared_ptr< MetaData > >& training_set, TrainParams train_params) 
  {
    // configure the manifold function
    // manifoldFn_telescope manifoldFn_discrete_sparse, manifoldFn_kmax
    manifoldFn = manifoldFn_boxMedian;
    
    if(g_params.has_key("SAVED_MODEL"))
    {
      // load a saved model
      FileStorage saved_model(g_params.require("SAVED_MODEL"),FileStorage::READ);
      assert(saved_model.isOpened());
      saved_model["model"]["search_tree_root"] >> search_tree_root;
      read(saved_model["model"]["parts_per_ex"],parts_per_ex);
      read(saved_model["model"]["allTemplates"],allTemplates);
      read(saved_model["model"]["sources"],sources);
      saved_model.release();
    }
    else
    {
      // train anew
      train_extract_templates(training_set,train_params);
      search_tree_root = train_search_tree(allTemplates);
    }
  }

  bool ApxNN_Model::write(FileStorage& fs)
  {
    fs << "{";

    fs << "search_tree_root" << search_tree_root;
    fs << "parts_per_ex" << parts_per_ex;
    fs << "allTemplates" << allTemplates;
    string s;
    fs << "sources"; deformable_depth::write(fs,s,sources);

    fs << "}";

    return true;
  }
  
  ApxNN_Builder::ApxNN_Builder()
  {
  }

  Model* ApxNN_Builder::build(Size gtSize, Size imSize) const
  {
    return new ApxNN_Model(gtSize,imSize);
  }

  string ApxNN_Builder::name() const
  {
    return "ApxNN_Builder";
  }

  Mat ApxNN_Model::vis_result_agreement(const ImRGBZ&im,Mat& background, DetectionSet& dets) const
  {
    // get the template
    DetectorResult best_det = dets[0];
    const NNTemplateType&Tmatched = allTemplates.at(best_det->pose);    

    // get X image
    RotatedRect best_det_bb(rectCenter(best_det->BB),best_det->BB.size(),best_det->in_plane_rotation);
    NNTemplateType XImage(im(best_det->BB),best_det->depth,nullptr,best_det_bb);
    if(XImage.getTIm().empty())
      return image_text("Error Extracting Image Template");

    // compute the difference
    Mat Tdiff = (Tmatched.getTIm().size() == XImage.getTIm().size())?
      Tmatched.getTIm()-XImage.getTIm() : image_text("bad sizes");
    Tdiff = cv::abs(Tdiff);
    Mat vis_match = imageeq("",Tdiff,false,false);
    cv::resize(vis_match,vis_match,best_det->BB.size());
    Mat template_vis = background.clone();
    //vis_match.copyTo(template_vis(best_det->BB));
    for(int rIter = best_det->BB.tl().y; rIter < best_det->BB.br().y; rIter++)
      for(int cIter = best_det->BB.tl().x; cIter < best_det->BB.br().x; cIter++)
      {
	if(rIter < 0 or cIter < 0 or template_vis.rows <= rIter or template_vis.cols <= cIter)
	  continue;
	double match_rate = 1.0 - vis_match.at<Vec3b>(rIter-best_det->BB.tl().y,
						      cIter-best_det->BB.tl().x)[0]/255.0;
	Vec3b&pixel = template_vis.at<Vec3b>(rIter,cIter);
	pixel = match_rate*Vec3b(0,0,255) + (1-match_rate)*pixel;
      }
    return template_vis;
  }
  
  Rect placeAdjust(const Mat&im,Rect metric_bb)
  {
    Rect vis_bb = metric_bb;
    // try right
    vis_bb.x += vis_bb.width;
    if(!rectContains(im,vis_bb))
    {
      vis_bb.x -= 2*vis_bb.width;
      if(!rectContains(im,vis_bb))
      {  
	vis_bb = metric_bb;
	vis_bb.y -= vis_bb.height;
	if(!rectContains(im,vis_bb))
	{
	  vis_bb = metric_bb;
	  vis_bb.y += vis_bb.height;
	  if(!rectContains(im,vis_bb))
	    throw std::runtime_error("couldn't place visualization");
	}
      }
    }    
    return vis_bb;
  }

  // visualize the extracted template to the right of the bounding box.
  Mat ApxNN_Model::vis_adjacent(const ImRGBZ&im,Mat&background,DetectionSet&dets) const
  {
    Mat vis = imageeq("",im.Z.clone(),false,false);
    vis = monochrome(vis,BLUE);
    // get a bb
    DetectorResult best_det = dets[0];
    float z = best_det->getDepth(im);
    Point2d center = rectCenter(best_det->BB);
    //Size metricSize = MetricSize(im.filename);
    //Rect metric_bb = im.camera.bbForDepth(
    //z, im.Z.size(), center.y, center.x, metricSize.width, metricSize.height,false);
    RotatedRect best_det_bb = best_det->rawBB;
    Rect metric_bb = rectFromCenter(best_det_bb.center,best_det_bb.size);
    rectangle(vis,metric_bb.tl(),metric_bb.br(),Scalar(0,255,255));

    // place the visualization
    //Rect vis_bb = placeAdjust(im.RGB,metric_bb);
    Rect vis_bb = metric_bb;

    //rectangle(vis,vis_bb.tl(),vis_bb.br(),Scalar(0,255,255));

    // draw the visualization
    const NNTemplateType&Tmatched = allTemplates.at(best_det->pose);    
    Mat vis_rot = imrotate(Tmatched.vis_high_res(),deg2rad(-best_det->in_plane_rotation));
    Mat vis_match = imageeq("",vis_rot,false,false);
    cv::resize(vis_match,vis_match,metric_bb.size());
    for(int yIter = 0; yIter < vis_match.rows; ++yIter)
      for(int xIter = 0; xIter < vis_match.cols; ++xIter)
      {
	Vec3b src_pix = vis_match.at<Vec3b>(yIter,xIter);
	int dst_x = clamp<int>(0,xIter + vis_bb.x,vis.cols-1);
	int dst_y = clamp<int>(0,yIter + vis_bb.y,vis.rows-1);
	Vec3b&dst_pix = vis.at<Vec3b>(dst_y,dst_x);
	if((src_pix[0] < 255 or src_pix[1] < 255 or src_pix[2] < 255) and src_pix != toVec3b(Colors::invalid()))
	{
	  double w = static_cast<double>(src_pix[0])/static_cast<double>(255);
	  dst_pix += w * GREEN;
	}
      }

    return vis;
  }

  // called from TestModel.cpp to visualize a detection.
  Mat ApxNN_Model::vis_result(const ImRGBZ&im,Mat& background, DetectionSet& dets) const
  {
    return visualize_result(im,background,dets).image();
  }  

  Visualization ApxNN_Model::visualize_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const
  {
    if(dets.size() <= 0)
      return Visualization(image_text("no dets to show"),"no_dets_to_show");
    
    // get the matched template
    DetectorResult best_det = dets[0];
    RotatedRect best_det_bb(rectCenter(best_det->BB),best_det->BB.size(),best_det->in_plane_rotation);
    const NNTemplateType&Tmatched = allTemplates.at(best_det->pose);    

    // get the image evidence
    NNTemplateType XImage(im(best_det->BB),best_det->depth,nullptr,best_det_bb);

    // visualize the matchiness
    Mat template_vis = vis_result_agreement(im,background, dets);

    // visualize the rotated template
    Mat TVis = imVGA(imageeq("",imrotate(Tmatched.getTIm(),deg2rad(-best_det->in_plane_rotation)),false,false));
    TVis = vertCat(TVis,image_text(printfpp("rotation = %f",best_det->in_plane_rotation)));

    //
    Mat vis_adj = vis_adjacent(im,background, dets);

    // draw the skeleton
    //Mat vis_skeleton = background.clone();
    //Tmatched.getMetadata()->drawSkeletons(vis_skeleton,best_det_bb.boundingRect());

    lock_guard<mutex> l(monitor);
    Visualization viz;
    viz.insert(imVGA(template_vis),"template_vis");
    viz.insert(imVGA(vis_adj),"vis_adj");
    viz.insert(TVis,"TVis");
    viz.insert(imVGA(imageeq("",XImage.getTIm(),false,false)),"XVis");
    viz.insert(imVGA(respImages[im.filename]),"resp");
    viz.insert(image_text(printfpp("resp = %f",best_det->resp)),"resp_text");
    return viz;
  }

  void ApxNN_Model::train_on_test(vector<shared_ptr<MetaData>>&training_set,
				  TrainParams train_params)
  {
    //train_hord_heuristics(*this,search_tree_root,training_set);
  }
}

