/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "GeneralizedHoughTransform.hpp"
#include "params.hpp"
#include <boost/multi_array.hpp>
#include <boost/math/special_functions/round.hpp>
#include "Skin.hpp"
#include "Orthography.hpp"
#include "Probability.hpp"
#include "util_vis.hpp"
#include "Quaternion.hpp"
#include "util.hpp"
#include "Detection.hpp"
 
namespace deformable_depth
{
  using params::RESP_ORTHO_X_RES;
  using params::RESP_ORTHO_Y_RES;  
  using params::RESP_ORTHO_Z_RES;
  
  ///
  /// SECTION: static
  ///
  
  static Size ortho_res(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES);
 
  auto houghMatrixLayer(HoughOutputSpace::Matrix3D&matrix, int z_dim) -> 
    Mat&
  {
    return matrix[ z_dim ];
  }     
  
  double vote_gaussian(
    Mat & output, 
    Point3d centroid, 
    double conf,
    const Mat&voting_kernel)
  {
    int yMin = centroid.y - voting_kernel.rows/2;
    int xMin = centroid.x - voting_kernel.cols/2;
    double max = -inf;
    
    // vote into the window     
    for(int yIter = yMin; yIter < yMin + voting_kernel.rows;  yIter++)
      for(int xIter = xMin; xIter < xMin + voting_kernel.cols; xIter++)
      {
	if(yIter < 0 || xIter < 0 || xIter >= output.cols || yIter >= output.rows)
	  continue;
	
	int kernel_y = yIter - yMin; 
	int kernel_x = xIter - xMin;
	double kernel_here = conf*voting_kernel.at<double>(kernel_y,kernel_x);
	output.at<double>(yIter,xIter) += kernel_here;
	max = std::max(max,kernel_here);
	//output.negative[xIter][yIter][0] += (1-conf)*voting_kernel.at<double>(kernel_y,kernel_x);
      }
  
    return max;
  }
    
  void vote_flat(HoughOutputSpace&output, Rect_<double> roi, double conf)
  {
    // vote into the window     
    for(int yIter = roi.tl().y; yIter < roi.br().y;  yIter++)
      for(int xIter = roi.tl().x; xIter < roi.br().x; xIter++)
      {
	if(yIter < 0 || xIter < 0 || xIter >= output.xSize() || yIter >= output.ySize())
	  continue;
	
	output.negative.at(xIter,yIter,0) += conf;
      }
  }  
  
  double latent_q2bin(double latent_q)
  {
    return interpolate_linear(
	latent_q,
	0.0,PCAPose::PROJ_Q_MAX,
	0,HoughOutputSpace::LATENT_SPACE_DIM);
  }
  
  double bin2latent_q(double latent_bin)
  {
    return interpolate_linear(
	latent_bin,
	0,HoughOutputSpace::LATENT_SPACE_DIM,
	0.0,PCAPose::PROJ_Q_MAX);        
  }
  
  // vote into a Hough space w/ 1 latent dimension
  void vote_gaussian_latent1(
    LatentHoughOutputSpace& output, 
    int latent_var_index,
    const Gaussian&latent_dist,
    const PCAPose&pose,
    Point3d pos_center, 
    Point3d neg_center,
    double correct_ratio,
    const Mat&voting_kernel)
  {
    // this has to call vote_gaussian once per layer in 
    // the latent space.
    if(latent_var_index >= output.lht_positives.size())
    {
      cout << "latent_var_index: " << latent_var_index << endl;
      cout << "output.lht_positives.size(): " << output.lht_positives.size() << endl;
    }
    HoughOutputSpace::Matrix3D&latent_positive = output.lht_positives[latent_var_index];
    HoughOutputSpace::Matrix3D&latent_negative = output.lht_negatives[latent_var_index];
    int latent_dimensions = latent_positive.zSize();
    
    // compute the latent weights
    vector<double> latent_weights;
    for(int latent_bin = 0; latent_bin < latent_dimensions; ++latent_bin)
    {
      // we need to map the latent_bin into the domain of the latent variable 
      // and evalaute the PDF to re-weight the correct_ratio for this bin. 
      // map [0,LATENT_DIMS] to [lMin,lMax]
      double latent_superposition = bin2latent_q(latent_bin);
      double latent_weight = latent_dist.pdf(latent_superposition);
      if(std::isnan(latent_weight)) latent_weight = 0;
//       cout << printfpp("latent min/max = %f/%f",
// 		       pose.latentMin(latent_var_index),
// 		       pose.latentMax(latent_var_index)) << endl;
//       cout << printfpp("latent_bin = %d superposition %f weight %f",
// 		       latent_bin,latent_superposition,latent_weight) << endl;
      assert(latent_weight >= 0);   
      latent_weights.push_back(latent_weight);
    }
    latent_weights = latent_weights / sum(latent_weights); // normalize
    
    for(int latent_bin = 0; latent_bin < latent_dimensions; ++latent_bin)
    {
      // extract the views
      auto negative_level_view = houghMatrixLayer(latent_negative,latent_bin);
      auto positive_level_view = houghMatrixLayer(latent_positive,latent_bin);
      
      double latent_weight = latent_weights[latent_bin];
      
      // negative
      assert(!voting_kernel.empty());      
      // vote background on this level
      // 1.0/latent_weights.size() or latent_weight?
      vote_gaussian(negative_level_view,neg_center,
		    (1-latent_weight)*(1-correct_ratio),voting_kernel);
      
      // positive
      if(latent_weight > 0)
      {
	// vote foreground on this level
	vote_gaussian(positive_level_view,pos_center,
		      latent_weight*(correct_ratio),voting_kernel);        
      }
    }
  }
  
  ///
  /// SECTION: HoughOutputSpace
  ///    
  
  ///
  /// SECTION: Matrix3D
  ///
  double& HoughOutputSpace::Matrix3D::at(int x, int y, int z)
  {
    return (*this)[z].at<double>(y,x);
  }

  double HoughOutputSpace::Matrix3D::at(int x, int y, int z) const
  {
    return (*this)[z].at<double>(y,x);
  }
  
  HoughOutputSpace::Matrix3D::Matrix3D()
  {
  }

  HoughOutputSpace::Matrix3D::Matrix3D(const vector< Mat >& copy) : 
    vector<cv::Mat>(copy)
  {
  }

  HoughOutputSpace::Matrix3D::Matrix3D(size_t N, Mat init) : 
    vector<cv::Mat>(N,init)
  {
  }

  int HoughOutputSpace::Matrix3D::xSize() const
  {
    return (*this)[0].cols;
  }

  int HoughOutputSpace::Matrix3D::ySize() const
  {
    return (*this)[0].rows;
  }

  int HoughOutputSpace::Matrix3D::zSize() const
  {
    return size();
  }
      
  double HoughOutputSpace::vote(
    bool prediction,
    double correct_ratio,
    Point3d center_pos, 
    Point3d center_neg,
    Mat&voting_kernel,
    const PCAPose&pose,
    const HoughVote&vote_pose
 			      )
  {
    // vote background
    auto neg_layer0 = houghMatrixLayer(negative,0);
    vote_gaussian(neg_layer0,center_neg,1-correct_ratio,voting_kernel);
    // vote foreground
    auto pos_layer0 = houghMatrixLayer(positive,0);    
    double result_vote_count = 
      vote_gaussian(pos_layer0,center_pos,correct_ratio,voting_kernel);
    return result_vote_count;
  }
  
  // functions for dense responce maps
  Mat max_z(HoughOutputSpace::Matrix3D& space,Mat&indexes,Mat&lconf)
  {
    int xsize = space[0].cols;
    int ysize = space[0].rows;
    Mat flat_space(ysize,xsize,DataType<float>::type,Scalar::all(-inf));
    indexes = cv::Mat(ysize,xsize,DataType<float>::type,Scalar::all(qnan));
    lconf = cv::Mat(ysize,xsize,DataType<float>::type,Scalar::all(qnan));
    
    for(cv::Mat & mat : space)
      assert(mat.type() == DataType<double>::type);
    
    for(int yIter = 0; yIter < ysize; yIter++)
      for(int xIter = 0; xIter < xsize; xIter++)
      {
	map<double/*resp*/,int/*idx*/> values_at_position;
	vector<double> weights, values;
	for(int zIter = 0; zIter < space.size(); zIter++)
	{
	  double value = space[zIter].at<double>(yIter,xIter);
	  values_at_position[value] = zIter;
	  values.push_back(zIter);
	  weights.push_back(value);
	  //indexes.at<float>(yIter,xIter) = zIter;
	}
	
	double top_conf = values_at_position.rbegin()->first;
	double next_conf = (--values_at_position.rbegin())->first; 
	flat_space.at<float>(yIter,xIter) = top_conf;
	lconf.at<float>(yIter,xIter) = top_conf/(top_conf+next_conf);
	
	// update indexes
	weights = weights / sum(weights);
	double i_out = dot(weights,values);
	indexes.at<float>(yIter,xIter) = i_out;
      }
    
    return flat_space;
  }  
  
  static const Mat blank(RESP_ORTHO_Y_RES,RESP_ORTHO_X_RES,
		 DataType<double>::type,Scalar::all(0));
  
  HoughOutputSpace::HoughOutputSpace() 
  {
    positive = vector<cv::Mat>{blank.clone()}; 
    negative = vector<cv::Mat>{blank.clone()};
  }
  
  HoughOutputSpace::Matrix3D 
    HoughOutputSpace::likelihood_ratio(const Matrix3D& active_positive, const Matrix3D& active_negative) const
  {
    // this is a bit strange,
    // Pr(Y = 1 | X = x) = Pr(X = x | Y = 1)*Pr(Y = 1)/Pr(X = x)
    // Pr(X = x | Y = 1) ~= Votes here / Votes that would arrive here if we are positive
    
    Matrix3D likelihood;
    for(int zIter = 0; zIter < active_positive.size(); zIter++)
    {
      likelihood.push_back(active_positive[zIter].clone());
      for(int yIter = 0; yIter < RESP_ORTHO_Y_RES; yIter++)
	for(int xIter = 0; xIter < RESP_ORTHO_X_RES; xIter++)
	{
	  double   neg = active_negative[zIter].at<double>(yIter,xIter);
	  double   pos = active_positive[zIter].at<double>(yIter,xIter);
	  double & lik = likelihood[zIter].at<double>(yIter,xIter);
	  
	  if(neg + pos > 0)
	    lik = pos / (pos + neg);
	    //lik = pos / neg;
	  else
	    lik = -inf;
	}
    }
      
    return likelihood;
  }
  
  Mat visualize_hough_resutls(
    int latent_var,
    HoughOutputSpace::Matrix3D&likelihood,
    const ImRGBZ& im,
    Mat&indices)
  {
    Mat lconf;
    //Mat vis_pos = imageeq("",max_z(hand_space.positive,indices,lconf));
    //Mat vis_neg = imageeq("",max_z(hand_space.negative,indices,lconf));
//     for(int layerIter = 0; layerIter < likelihood.zSize(); ++layerIter)
//     {
//       Mat layer; likelihood[layerIter].convertTo(layer,DataType<float>::type);
//       Mat layerPos; hand_space.lht_positives[latent_var][layerIter].
// 	convertTo(layerPos,DataType<float>::type);
//       Mat layerNeg; hand_space.lht_negatives[latent_var][layerIter].
// 	convertTo(layerNeg,DataType<float>::type);
      //log_im("DEBUG_likelihood_layer",tileCat(vector<Mat>{
	//imageeq("",layer),imageeq("",layerPos),imageeq("",layerNeg)}));
//     }
    Mat flat_like = max_z(likelihood,indices,lconf);
    Mat vis_lik = imageeq("",flat_like);
    cout << "max idxs: " << indices << endl;
    Mat vis_max_idx = imageeq("",indices);
    Mat vis_lconf = imageeq("",lconf);
    Mat resps = tileCat(vector<Mat>{/*vis_pos,vis_neg,*/vis_lik,vis_max_idx,vis_lconf});
    log_im(printfpp("SRF_RESPs_%d_",latent_var),horizCat(im.RGB,resps));    
    
    return flat_like;    
  }  
  
  shared_ptr<HoughLikelihoods> HoughOutputSpace::likelihood_ratio(const ImRGBZ& im) const
  { 
    Matrix3D liki3d = likelihood_ratio(positive,negative);
    Mat indexes;
    Mat flat_like = visualize_hough_resutls(-1,liki3d,im,indexes);
    
    shared_ptr<HoughLikelihoods> result(new HoughLikelihoods());
    result->observed_likelhoods = liki3d;
    result->flat_resps = flat_like;
    return result;
  }
  
  int HoughOutputSpace::xSize() const
  {
    return positive.xSize();
  }

  int HoughOutputSpace::ySize() const
  {
    return positive.ySize();
  }
  
  double HoughOutputSpace::spearman_binary_delta(
    const Matrix3D&ground_truth,const Matrix3D&delta_pos, const Matrix3D&delta_neg)
  {
    double sbd = 0;
    
    for(int xIter = 0; xIter < ground_truth.xSize(); xIter++)
      for(int yIter = 0; yIter < ground_truth.ySize(); yIter++)
	for(int zIter = 0; zIter < ground_truth.zSize(); zIter++)
	{
	  double p = delta_pos.at(xIter,yIter,zIter);
	  double n = delta_neg.at(xIter,yIter,zIter);
	  double g = ground_truth.at(xIter,yIter,zIter);
	  if(g > .5)
	  {
	    sbd += p - n;
	  }
	  else
	  {
	    sbd += n - p;
	  }
	}
	
    return sbd;
  }
  
  shared_ptr<HoughLikelihoods> LatentHoughOutputSpace::likelihood_ratio(const ImRGBZ& im) const
  { 
    // compute the indpendent likelhood ratios
    Matrix3D liki3d_0 = HoughOutputSpace::likelihood_ratio(lht_positives[0],lht_negatives[0]);
    Matrix3D liki3d_1 = HoughOutputSpace::likelihood_ratio(lht_positives[1],lht_negatives[1]);    
    
    Mat indexes0, indexes1;
    Mat flat_like = visualize_hough_resutls(0,liki3d_0,im,indexes0);
    //flat_like = deformable_depth::min(flat_like,visualize_hough_resutls(0,hand_space,im));
    flat_like = deformable_depth::min(flat_like,visualize_hough_resutls(1,liki3d_1,im,indexes1));
    //flat_like = flat_like + visualize_hough_resutls(1,hand_space,im);
    //flat_like = flat_like + visualize_hough_resutls(1,hand_space,im);
    //flat_like = visualize_hough_resutls(-1,hand_space,im); 
    
    shared_ptr<LatentHoughLikelihoods> result(new LatentHoughLikelihoods());
    result->flat_resps = flat_like;
    result->latent_estimates.push_back(indexes0);
    result->latent_estimates.push_back(indexes1);
    result->pcaPose = pcaPose;
    return result;      
  }
  
  double LatentHoughOutputSpace::vote(
    bool prediction, double correct_ratio, 
    Point3d center_pos, Point3d center_neg, 
    Mat& voting_kernel, const PCAPose& pose, const HoughVote& vote_pose)
  {
    double result_vote_count = HoughOutputSpace::vote(
      prediction,correct_ratio,center_pos,center_neg,voting_kernel,pose,vote_pose);
    
    // vote into our latent spaces
    for(int latent_dim = 0; latent_dim < PCAPose::N_DIM; ++latent_dim)
    {
      Gaussian pose_dist = vote_pose.marginalize(prediction,latent_dim);
//       cout << "pose_dist: " << pose_dist << endl;
      vote_gaussian_latent1(*this, latent_dim,
			    pose_dist,
			    pose,center_pos, center_neg,correct_ratio,voting_kernel);
    }    
    
    return result_vote_count;    
  }
  
  LatentHoughOutputSpace::LatentHoughOutputSpace(PCAPose pcaPose) : 
    pcaPose(pcaPose)
  {
    for(int iter = 0; iter < PCAPose::N_DIM; ++iter)
    {
      lht_negatives.push_back(vector<cv::Mat>());
      for(int jter = 0; jter < LATENT_SPACE_DIM; jter++)
	lht_negatives[iter].push_back(blank.clone());
      
      lht_positives.push_back(vector<cv::Mat>());
      for(int jter = 0; jter < LATENT_SPACE_DIM; jter++)
	lht_positives[iter].push_back(blank.clone());
    }
  }
    
  ///
  /// SECTION: HoughVote
  ///
  void HoughVote::update_votes(double cov_add_eye_weight)
  {
    cov_false = cov_false + 
      cov_add_eye_weight*Mat::eye(cov_false.rows,cov_false.cols,cov_false.type());
    if(!cov_false.empty() && cv::determinant(cov_false) != 0)
    {
      sigmaInv_false = cov_false.inv();
      voting_kernel_false = mv_gaussian_kernel(cov_false,sigmaInv_false);
    }
    
    cov_true = cov_true + 
      cov_add_eye_weight*Mat::eye(cov_true.rows,cov_true.cols,cov_true.type());    
    if(!cov_true.empty() && cv::determinant(cov_true) != 0)
    {
      sigmaInv_true  = cov_true.inv();
      voting_kernel_true = mv_gaussian_kernel(cov_true,sigmaInv_true);
    }
  }
  
  const Mat& HoughVote::cov(bool prediction) const
  {
    if(prediction)
      return cov_true;
    else
      return cov_false;
  }

  const Mat& HoughVote::mu(bool prediction) const
  {
    if(prediction)
      return mu_true;
    else
      return mu_false;
  }
  
  Gaussian HoughVote::marginalize(bool prediction, int dim) const
  {
    if(dim >= mu(prediction).size().area())
      return Gaussian(qnan,qnan);
    
    double sigma = std::sqrt(cov(prediction).at<double>(dim,dim));
    double d_mu = mu(prediction).at<double>(dim);
    
    return Gaussian(d_mu,sigma);
  }
  
  Mat HoughVote::log_kernels() const
  {    
    // generate the kernel image
    Mat vis_false = voting_kernel_false;
    Mat vis_true  = voting_kernel_true;
    Mat sidebyside = horizCat(vis_true,vis_false);
    Mat vis_sidebyside = imageeq("",sidebyside,false,false);
    
    log_file << "HoughVote::log_kernels(): " << message << endl;
    
    return vis_sidebyside;
  } 
  
  ///
  /// SECTION: HoughLikelihoods
  ///
  Point3d HoughLikelihoods::center(DetectorResult& win, std::shared_ptr< const ImRGBZ >& im) const
  {
    StructuredWindow swin;
    swin.detection = win;
    swin.im = im;    
    
    return swin.ortho_detBBCenter();
  }
  
  void HoughLikelihoods::read_detection(
    DetectorResult& win, std::shared_ptr< const ImRGBZ >& im)
  {
    // check that the BB is in the image
    Rect_<double> bb = win->BB;
    if(!(0 <= bb.x && 0 <= bb.width && bb.x + bb.width <= im->Z.cols && 
	0 <= bb.y && 0 <= bb.height && bb.y + bb.height <= im->Z.rows))
    {
      win->resp = -inf;
      return;
    }
    
    // compute info about win
    Point3d center = HoughLikelihoods::center(win,im);
    float resp = flat_resps.at<float>(center.y,center.x);
    double z_min = extrema(im->Z(win->BB)).min;
    
    // update BB
    // 	only detect extruding objects as hands
    if(std::abs(z_min - center.z) > 10)
    {
      double x = win->BB.tl().x+extrema(im->Z(win->BB)).minLoc.x;
      double y = win->BB.tl().y+extrema(im->Z(win->BB)).minLoc.y;
      win->BB = rectFromCenter(Point2d(x,y),win->BB.size());
    }
    
    // update resps
    win->resp = resp;
  }
  
  ///
  /// SECTION: Latent Hough Likelihoods
  ///
  void LatentHoughLikelihoods::read_detection(DetectorResult& win, 
					      std::shared_ptr< const ImRGBZ >& im)
  {
    // get the basic bb detection
    HoughLikelihoods::read_detection(win,im);
    if(win->resp == -inf)
      return;
    
    // now try to get the pose
    Point3d center = HoughLikelihoods::center(win,im);
    assert(PCAPose::N_DIM == 2);
    assert(latent_estimates[0].type() == DataType<float>::type);
    Mat latent_pose(1,2,DataType<double>::type);
    latent_pose.at<double>(0) = bin2latent_q(latent_estimates[0].at<float>(center.y,center.x));
    latent_pose.at<double>(1) = bin2latent_q(latent_estimates[1].at<float>(center.y,center.x));
    for(auto & part : pcaPose.unproject_q(latent_pose,win,*im))
      win->emplace_part(part.first,part.second);
    win->latent_space_position.push_back(latent_pose.at<double>(0));
    win->latent_space_position.push_back(latent_pose.at<double>(1));
  }
}
