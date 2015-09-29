/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "OcclusionReasoning.hpp"
#include "Log.hpp"
#include "GlobalFeat.hpp"
#include "FauxLearner.hpp"

namespace deformable_depth
{  
  ///
  /// SECTION: OccAwareLinearModel
  ///
  static atomic<long> dot_product_count;
  
  OccAwareLinearModel::OccAwareLinearModel(Size gtSize, Size ISize, double C, double area, shared_ptr< IHOGComputer_Factory > comp_factory, shared_ptr< LDA > learner) : 
    SparseOneFeatureModel(gtSize,ISize, C, area, comp_factory, learner),
    occ_factory(shared_ptr<IHOGComputer_Factory>(new AdapterFactory<OccSliceFeature>(comp_factory)))
  {
    init();
  }

  OccAwareLinearModel::OccAwareLinearModel(shared_ptr< IHOGComputer_Factory > comp_factory) :
    SparseOneFeatureModel(comp_factory),
    occ_factory(shared_ptr<IHOGComputer_Factory>(new AdapterFactory<OccSliceFeature>(comp_factory)))
  {
    //init();
  }

  void OccAwareLinearModel::init_post_construct()
  {
    deformable_depth::OneFeatureModel::init_post_construct();
    init();
  }
  
  void OccAwareLinearModel::init()
  {
    assert(ISize.area() > 0);
    assert(TSize.area() > 0);
    occ_hog.reset(dynamic_cast<OccSliceFeature*>(occ_factory->build(ISize,s_cell)));
    occ_hog1.reset(dynamic_cast<OccSliceFeature*>(occ_factory->build(TSize,s_cell)));
    obj_depth = fromString<float>(g_params.require("OBJ_DEPTH"));
    times_logged = 0;
    prime_learner();
  }
  
  void OccAwareLinearModel::do_dot_extract_one_feat(
    float&xValue, bool&occluded,bool&background,
    int x0, int xIter, int y0, int yIter,
    int bin,float cell_depth, float min_depth, float max_depth,
    shared_ptr< FeatIm >&im_feats,
    DepthFeatComputer& hog_for_scale,
    const vector<double>&depths_histogram) const
  {     
    if(bin < CellDepths::METRIC_DEPTH_BINS)
    {
      xValue = depths_histogram[bin];
    }
    else
    {
      // the occlusion reasoning should be moved to the pairwise model.
      if(!occluded)
	xValue = (*im_feats)[hog_for_scale.getIndex(x0+xIter,y0+yIter,0,bin-OccSliceFeature::DECORATION_LENGTH)];
      else
	xValue = 0;
    }    
  }
  
  float OccAwareLinearModel::do_dot(int y0, int x0, double manifold_z, 
				    int blocks_x, int blocks_y, int nbins, 
				    const ImRGBZ& im, vector< float >& wf, 
				    shared_ptr< FeatIm > im_feats, 
				    DepthFeatComputer& hog_for_scale, 
				    const CellDepths& cell_depths,
				    const DetectionFilter&filter,
				    float&occ_percent, float&real_perecnt) const
  {
    double min_depth = manifold_z;
    double max_depth = manifold_z + obj_depth;
    double num_real = 0;
    int occluded_cell_count = 0;
    int background_cell_count = 0;
    
    float resp = 0;
    int wIdx = 0; // order of iterations important for this to work
    for(int xIter = 0; xIter < blocks_x; xIter++)
      for(int yIter = 0; yIter < blocks_y; yIter++)
      {
	float cell_depth = cell_depths.upper_manifold.at<float>(y0+yIter,x0+xIter);
	bool occluded = (cell_depth < min_depth);
	bool background = (cell_depth >= max_depth);
	
	// extract the histogram of depths
	vector<double> depths_histogram = 
	  cell_depths.depth_histogram(y0+yIter,x0+xIter,min_depth,max_depth);
	
	// loop over the bins
	for(int bin = 0; bin < nbins + OccSliceFeature::DECORATION_LENGTH; bin++)
	{
	  assert(bin < occ_hog1->getNBins());
	  
	  // get the w component
	  float wValue = wf[wIdx]; wIdx++;
	  
	  // get the x component
	  float xValue;
	  do_dot_extract_one_feat
	    (xValue, occluded,background,x0, xIter, y0, yIter,
	     bin,cell_depth, min_depth, max_depth,
	      im_feats,hog_for_scale,depths_histogram);
	  
	  // dot product
	  resp += wValue*xValue;
	} // finish looping over the bins
	
	if(!(occluded || background))
	  num_real++;
	
	if(occluded)
	  occluded_cell_count++;
	if(background)
	  background_cell_count++;
      }
      
    occ_percent = 
      static_cast<double>(occluded_cell_count)
      /static_cast<double>(blocks_x*blocks_y);
    real_perecnt = num_real/(blocks_x*blocks_y);
    double background_percent =   
      static_cast<double>(background_cell_count)
      /static_cast<double>(blocks_x*blocks_y);    
    
    // DEBUG: check may be temporalily disabled for debugging
    //if(true)
    //if(background_percent <= .60 && (occ_percent == 1 || occ_percent == 0))
    if(filter.is_root_template && filter.testing_mode 
      && (background_percent > .9 || background_percent < .1))
    {
      //cout << "Reject BG % = " << background_percent << endl;
      return -inf;
    }
    else
    {
      dot_product_count++;
      return resp;
    }
  }
  
  DetectionSet OccAwareLinearModel::do_extern(
    vector< Intern_Det >& interns, string filename, 
    shared_ptr< FeatIm > im_feats, Size im_blocks, 
    DetectionFilter filter, shared_ptr<CellDepths> cell_depths,
    const vector<float>&occ_percent, const vector<float>&real_percent,
    double scale) const
  {
    DetectionSet detections = 
      externDets(interns, filename, im_feats, im_blocks, filter,scale);
    int idx = 0;
    for(DetectorResult & detection : detections)
    {
      auto raw_feat = detection->feature;
      float min_depth = detection->depth;
      //float max_depth = detection->depth + obj_depth;
      float max_depth = detection->depth + obj_depth;
      int block_x1 = interns[idx].col;
      int block_y1 = interns[idx].row;
      int block_x2 = block_x1 + nx;
      int block_y2 = block_y1 + ny;
      Rect roi(Point(block_x1,block_y1),Point(block_x2,block_y2));
      
      detection->feature = 
      [this,cell_depths,raw_feat,min_depth,max_depth,roi]()
      {  
	assert(roi.area() == occ_hog1->blocks_x()*occ_hog1->blocks_y());
	SparseVector raw = raw_feat();
	vector<float> fraw = raw;
	vector<float> feat = occ_hog1->decorate_feat(
	  fraw,(*cell_depths)(roi),min_depth,max_depth);
	assert(feat.size() == occ_hog1->getDescriptorSize());
	//log_once(printfpp("OccAwareLinearModel::do_extern raw = %d dec = %d",
			  //(int)raw.size(),(int)feat.size()));
	return feat;
      };
      
      detection->occlusion = occ_percent[idx];
      detection->real = real_percent[idx];
      detection->z_size = obj_depth;
      
      idx++;
    }
    return detections; 
  }
  
  DetectionSet OccAwareLinearModel::detect_at_scale(const ImRGBZ& im_raw, DetectionFilter filter, FeatPyr_Key  scale) const
  {
    // avoid pesky rounding errors.
    if(im_raw.RGB.size().area() <= 1)
      return DetectionSet{};
    shared_ptr<const ImRGBZ> im_crop = cropToCells(im_raw,occ_hog->getCellSize().width,
						    occ_hog->getCellSize().height);
    const ImRGBZ&im = *im_crop;
    
    unique_ptr<OccSliceFeature> occ_feat_for_scale(
      dynamic_cast<OccSliceFeature*>(occ_factory->build(im.Z.size(),s_cell)));
    if(occ_feat_for_scale->blocks_x() < nx || occ_feat_for_scale->blocks_y() < ny)
      return DetectionSet{};
    shared_ptr<CellDepths> cell_depths(new CellDepths(occ_feat_for_scale->mkCellDepths(im)));
    
    // we need to add two virtual features to the dot product.
    vector<float> occ_perc;
    vector<float> real_perc;
    auto dotFn = [&](int y0, int x0, double manifold_z,
		      int blocks_x, int blocks_y, int nbins, 
		      const ImRGBZ&im,
		      std::vector< float >& wf, 
		      shared_ptr< FeatIm > im_feats, 
		      DepthFeatComputer& hog_for_scale)
      {
	occ_perc.push_back(0);
	real_perc.push_back(0);
	return do_dot(y0, x0, manifold_z, blocks_x, blocks_y, nbins, im, wf, im_feats, 
		      hog_for_scale, *cell_depths,filter,occ_perc.back(),real_perc.back());
	
      };
    
    //  here we add the occ and bg features
    auto externFn = [&](vector< Intern_Det >& interns, string filename, 
			shared_ptr< FeatIm > im_feats, Size im_blocks, DetectionFilter filter)
    {
      return do_extern(
	interns,filename,im_feats,im_blocks,filter,cell_depths,occ_perc,real_perc,scale.scale);
    };      
      
    auto manifoldFn = [&](int x0, int y0)
    {
      vector<float> flat_depths = flatten_cell_depths(cell_depths->upper_manifold,x0,y0,x0+nx,y0+ny);
      if(flat_depths.size() == 0)
	return vector<float>();
      float min_depth = flat_depths[0];
      
      if(filter.allow_occlusion)
      {
	return vector<float>{
	  order(flat_depths,0.0),
	  order(flat_depths,0.5),
	  order(flat_depths,1.0)+.1f};
      }
      else
	return vector<float>{order(flat_depths,0)};
	
      //return vector<float>{min_depth,min_depth+obj_depth/2,min_depth+obj_depth};
    };
    
    auto result = deformable_depth::SparseOneFeatureModel::detect_at_scale(
      im, filter, scale,dotFn,externFn,manifoldFn);
    //log_once(printfpp("Dot Product count = %ld",static_cast<long>(dot_product_count)));
    return result;
  }
  
  vector< float > OccAwareLinearModel::flatten_cell_depths(
    const Mat& cell_depths, int xmin, int ymin, int xsup, int ysup) const
  {
    // PASS 1: compute min depth
    float min_depth = inf;
    for(int x = xmin; x < xsup; x++)
      for(int y = ymin; y < ysup; y++)
	min_depth = std::min<float>(min_depth,cell_depths.at<float>(y,x));
    
    // PASS 2: collect depths which are not in the background.
    vector<float> flat_depths;
    for(int x = xmin; x < xsup; x++)
      for(int y = ymin; y < ysup; y++)
      {
	float depth = cell_depths.at<float>(y,x);
	if(depth < min_depth + obj_depth && depth < params::MAX_Z())
	  flat_depths.push_back(depth);
      }
    std::sort(flat_depths.begin(),flat_depths.end());
    return flat_depths;
  }
  
  void OccAwareLinearModel::chose_pos_depth(
    const AnnotationBoundingBox&bb, const Mat&cell_depths,float&min_depth,float&max_depth) const
  {
    // convert Mat to vector<float>
    vector<float> flat_depths = flatten_cell_depths(
      cell_depths,1,1,cell_depths.cols-1,cell_depths.rows-1);
    
    if(flat_depths.size() == 0)
    {
      log_once(printfpp("warning: a positive bb which is in background"));
      min_depth = params::MAX_Z();
    }
    // switch on the visiblity percentage
    else if(bb.visible > 2.0/3.0)
    {
      // the object is fully visible
      min_depth = order(flat_depths,0.0);
    }
    else if(bb.visible > 1.0/3.0)
    {
      // the object is partially occluded
      min_depth = order(flat_depths,0.5);
    }
    else
    {
      // the object is fully occluded
      min_depth = order(flat_depths,1.0)+.1f;
    }
    
    // set the max depth trivially
    max_depth = min_depth + obj_depth;
  }
  
  SparseVector OccAwareLinearModel::extractPos(MetaData&metadata, AnnotationBoundingBox bb) const
  {
    // validate size
    bool good_size = true;
    if(bb.area() <= 0)
      good_size = false;
    if(bb.area() <= min_area())
      good_size = false;    
    if(!good_size)
      log_once(printfpp("Bad size in %s",metadata.get_filename().c_str()));
    
    // get the image
    //log_file << "++OccAwareLinearModel::extractPos" << endl;
    shared_ptr<const ImRGBZ> im = metadata.load_im();
    return extract(*im,bb,true).sp_vec;
  }
  
  OccAwareLinearModel::PosInfo OccAwareLinearModel::extract(const ImRGBZ&im, 
					       AnnotationBoundingBox bb,bool is_pos_bb) const
  {
    bool good_bb = 
      0 <= bb.x && 0 <= bb.width && 
      bb.x + bb.width <= im.Z.cols && 
      0 <= bb.y && 0 <= bb.height && 
      bb.y + bb.height <= im.Z.rows;
    if(!good_bb)
    {
      log_once(printfpp(
	"OccAwareLinearModel::extractPos: [%s] BB has issues [%s] tl: (%d %d) size: [%d %d]",
	im.filename.c_str(),
	(!good_bb)?"bb range problem":"bb area problem",
	(int)bb.x,(int)bb.y,(int)bb.width,(int)bb.height));
      return PosInfo{vector<float>(),qnan,qnan,qnan};
    }
    
    auto depthFn_auto = [&]
      (const CellDepths&cell_depths,float&min_depth,float&max_depth)
      {
	//bb.visible = true; // force no-occlusion in positives for now.
	chose_pos_depth(bb,cell_depths.upper_manifold,min_depth,max_depth);
      };
    
    auto depthFn_provided = [&]
      (const CellDepths&cell_depths,float&min_depth,float&max_depth)
      {
	min_depth = bb.depth;
	max_depth = bb.depth + OccAwareLinearModel::obj_depth;
      };
         
      
    // ideally, we'll pad occluded cells with a "1"
    SparseVector pos = OneFeatureModel::extractPos_w_border(im,bb,&*occ_factory,
			[&](DepthFeatComputer* comp)
			{
			    OccSliceFeature* occ_comp = dynamic_cast<OccSliceFeature*>(comp);
			    assert(occ_comp != nullptr);
			    if(std::isnan(bb.depth))
			      occ_comp->setDepthFn(depthFn_auto);
			    else
			      occ_comp->setDepthFn(depthFn_provided);
			});
    if(pos.size() == 0)
    {
      log_once(printfpp("OccAwareLinearModel::extractPos: subordinate problem %s",
	im.filename.c_str()));
      return PosInfo{pos,qnan,qnan,qnan};
    }
    else if(occ_hog1->occlusion(pos) > .1)
    {
      //log_once(printfpp("Occluded Positive"));
      //return PosInfo{vector<double>{},qnan,qnan,qnan};
    }
    
    static atomic<int> global_times_logged(0);
    if(times_logged < 2 || global_times_logged < 10)
    {
      times_logged++;
      global_times_logged++;
      
      // DEBUG: Show the decorated features
      Mat feat_vis = occ_hog1->show("",pos);
      log_im(printfpp("feat_vis_%s",occ_hog1->toString().c_str()),feat_vis); 
    }
    assert(pos.size() == occ_hog1->getDescriptorSize());
    
    // add the global feature
    SparseVector gbl_feat = GlobalFeature::calculate(im,bb,nx,ny);
    assert(gbl_feat.size() == GlobalFeature::length);
    if(GlobalFeature::length > 0)
      log_im("GlobalFeature",GlobalFeature::show(gbl_feat,nx,ny));
    pos.push_back(gbl_feat); 
    
    // format the output
    return PosInfo{pos,occ_hog1->real(pos),occ_hog1->occlusion(pos),occ_hog1->background(pos)};
  }
  
  void OccAwareLinearModel::prime_learner()
  {
    if(!learner)
      learner.reset(new FauxLearner());
      
    learner->prime(occ_hog1->getDescriptorSize() + GlobalFeature::length);
  }

  Mat OccAwareLinearModel::show(const string& title)
  {
    assert(occ_hog1 != nullptr);
    assert(learner != nullptr);
    assert(learner->getW().size() > 0);
    return occ_hog1->show(title,learner->getW()); waitKey_safe(1);
  }
  
  float OccAwareLinearModel::getObjDepth() const
  {
    return obj_depth;
  }
  
  Mat OccAwareLinearModel::vis_result(Mat&background, DetectionSet& dets) const
  {
    //log_file << "OneFeatureModel::vis_result: begin" << endl;
    if(dets.size() <= 0)
    {
      log_file << "OneFeatureModel::vis_result: no dets" << endl;
      return Mat();
    }
    
    // (1) extract feature from the detection
    if(!dets[0]->feature)
    {
      log_file << "OneFeatureModel::vis_result: feature() undefined" << endl;
      return Mat();
    }
    vector<double> feat = vec_f2d(dets[0]->feature());
    if(feat.size() != learner->getFeatureLength())
    {
      cout << "feat.size() = " << feat.size() << endl;
      cout << "learner->getFeatureLength() = " << learner->getFeatureLength() << endl;
      assert(false);
    }
    vector<double> w = learner->getW();
    //log_file << "OneFeatureModel::vis_result: feat extracted" << endl;
    
    // (2) visulalize feature using HoG1
    vector<FeatVis> planes = occ_hog1->show_planes(w);
    Mat feat_vis = planes[0].getPos();
    //log_file << "OneFeatureModel::vis_result: w visulized" << endl;
    
    // (3) mark contextual information from the feature into the template
    vector<double> local_feat(feat.begin(),feat.end()-GlobalFeature::length);
    occ_hog1->mark_context(feat_vis,local_feat,dets[0]->lr_flips % 2 == 1);
    
    // (4) put visulization into the background and return
    Rect bb = clamp(background,dets[0]->BB);
    if(bb.tl().x < 0 || bb.tl().y < 0 || bb.br().x >= background.cols || bb.br().y >= background.rows)
    {
      log_file << "OneFeatureModel::vis_result: bad size" << endl;
      return Mat();
    }
    resize(feat_vis,feat_vis,bb.size());
    Mat vis = background.clone();
    //image_safe("feat_vis",feat_vis);
    feat_vis.copyTo(vis(bb));
    rectangle(vis,bb.tl(),bb.br(),Scalar(0,255,0));
    //log_file << "OneFeatureModel::vis_result: done" << endl;
    return vis;
  }
  
  bool valid_configuration(bool part_is_visible, bool overlaps_visible)
  {
    bool permissive = false;
    if(permissive)
    {
      // permissive
      if(part_is_visible && !overlaps_visible) 
	return true;
      else if(!part_is_visible && !overlaps_visible)
	return false;
      else if(part_is_visible && overlaps_visible)
	return true; // this is difference vs. strict
      else // if !part_is_visible && overlaps_visible
	return true;
    }
    else
    {
      if(part_is_visible != overlaps_visible) 
	return true;
      else
	return false;
    }
  }
  
  void OccAwareLinearModel::debug_incorrect_resp(SparseVector& feat, Detection& det)
  {
    log_file << "Bad Det's BB: " << det.BB << endl;
    log_im("Bad Feature",occ_hog1->show("Bad Feature",feat));
    assert(false);
  }
  
  bool occlusion_intersection(const Rect_< double >& r1, const Rect_< double >& r2)
  {
    return rectIntersect(r1,r2) >= .3;
  }
  
  bool is_visible(float occ)
  {
    return occ < .5;
  }
}
