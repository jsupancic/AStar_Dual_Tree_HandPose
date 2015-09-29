/**
 * Copyright 2014: James Steven Supancic III
 **/ 

#include "HoughForest.hpp"
#include "RandomHoughFeature.hpp"

namespace std
{
  size_t hash < cv::Vec3b >::operator()(const cv::Vec3b &x ) const
  {
    return hash<uint8_t>()(x[0]) ^ hash<uint8_t>()(x[1]) ^ hash<uint8_t>()(x[2]);
  }
}

namespace deformable_depth
{
  using namespace cv;
  using namespace std;

  ///
  /// SECTION: PredictionStatistics
  ///
  PredictionStatistics::PredictionStatistics() : 
    X(0,0,0),
    X2(3,3,DataType<double>::type,Scalar::all(0)),
    N(0)
  {
  }

  void PredictionStatistics::update(Vec3d pix)
  {
    // update the regression
    X += pix;
    Mat pix_vec(3,1,DataType<double>::type,Scalar::all(0));
    for(int iter = 0; iter < 3; ++iter)
      pix_vec.at<double>(iter) = pix[iter];
    X2 += pix_vec*pix_vec.t();
    N ++ ;    

    // update for classification
    volatile Vec3b bpix = pix;
    Vec3b rbpix = pix;
    frequencies[rbpix]++;
  }

  unordered_map<Vec3b,long> PredictionStatistics::posterior() const
  {
    // produce a classification density estimate
    return frequencies;
  }

  Vec3d PredictionStatistics::predict() const
  {
    return X/N;
  }

  bool PredictionStatistics::splitable() const
  {
    static int TREE_SPLIT_SIZE = -1;
    if(TREE_SPLIT_SIZE == -1)
    {
      TREE_SPLIT_SIZE = fromString<int>(g_params.require("TREE_SPLIT_SIZE"));
    }

    return N > TREE_SPLIT_SIZE;
  }

  double PredictionStatistics::shannon_entropy() const
  {
    vector<double> ps;
    for(auto && pair : frequencies)
      ps.push_back(static_cast<double>(pair.second)/N);

    return deformable_depth::shannon_entropy(ps);
  }

  double PredictionStatistics::differential_entropy() const
  {
    // compute sigma
    Mat pix_vec(3,1,DataType<double>::type,Scalar::all(0));
    for(int iter = 0; iter < 3; ++iter)
      pix_vec.at<double>(iter) = X[iter];
    Mat SIGMA = X2/N - pix_vec*pix_vec.t()/(N*N);
    SIGMA += cv::Mat::eye(SIGMA.rows,SIGMA.cols,SIGMA.type());

    //cout << "SIGMA: " << SIGMA << endl;
    
    //return .5 * 
    volatile double det = cv::determinant(SIGMA);
    return std::log(det);
  }

  double PredictionStatistics::entropy() const
  {
    enum EntropyType
    {
      UNKNOWN = -1,
      SHANNON = 0,
      DIFFERENTIAL = 1
    };
    static EntropyType entropyType = UNKNOWN;
    if(entropyType == UNKNOWN)
    {
      string entropy_string = g_params.require("ENTROPY_TYPE");
      if(entropy_string == "shannon")
	entropyType = SHANNON;
      else if(entropy_string == "differential")
	entropyType = DIFFERENTIAL;
      else
	assert(false);
    }

    if(entropyType == SHANNON)
      return shannon_entropy();
    else if(entropyType == DIFFERENTIAL)
      return differential_entropy();
    throw std::logic_error("Bad Entropy Type");
  }

  double PredictionStatistics::samples() const
  {
    return N;
  }

  ///
  /// SplitFunction
  ///
  bool SplitFunction::split_kinect_feat(const Mat&Z,Point pt) const
  {
    double zu = Z.at<float>(pt.y,pt.x);
    double y1 = pt.y + d1.y/zu;
    double x1 = pt.x + d1.x/zu;
    double x2 = pt.y + d2.y/zu;
    double y2 = pt.x + d2.x/zu;
    if(!goodNumber(y1) or !goodNumber(y2) or !goodNumber(x1) or !goodNumber(x2))
      return false;
    double z1 = Z.at<float>(cv::borderInterpolate(y1,Z.rows,BORDER_REFLECT_101), 
			    cv::borderInterpolate(x1,Z.cols,BORDER_REFLECT_101));
    double z2 = Z.at<float>(cv::borderInterpolate(y2,Z.rows,BORDER_REFLECT_101), 
			    cv::borderInterpolate(x2,Z.cols,BORDER_REFLECT_101));
    double feat = z1 - z2;
    return (feat > thresh);
  }

  bool SplitFunction::split_absolute_x(const Mat&Z,Point pt) const
  {
    return pt.x < absolute.x;
  }

  bool SplitFunction::split_absolute_y(const Mat&Z,Point pt) const
  {
    return pt.y < absolute.y;
  }

  bool SplitFunction::split(const Mat&Z,Point pt) const
  {
    if(function_code == 0)
      return split_kinect_feat(Z,pt);
    // these might be useful for egocentric?
    else if(function_code == 1)
      return split_absolute_x(Z,pt);
    else
      return split_absolute_y(Z,pt);
  }

  SplitFunction::SplitFunction()
  {
    function_code = thread_rand()%3;
    d1 = Vec2d(sample_in_range(0,20*100),sample_in_range(0,20*100));
    d2 = Vec2d(sample_in_range(0,20*100),sample_in_range(0,20*100));
    absolute = Point(sample_in_range(0,320-1),sample_in_range(0,240-1));
    thresh = sample_in_range(-10,10);
  }

  void SplitFunction::train(const Mat&sem,const Mat&Z,Point pt)
  {
    Vec3d pix = sem.at<Vec3b>(pt.y,pt.x);
    if(split(Z,pt))
      stats_above.update(pix);
    else
      stats_below.update(pix);
  }

  double SplitFunction::info_gain() const
  {
    double N1 = stats_above.samples();
    double N2 = stats_below.samples();    
    double N  = N1 + N2;
    volatile double H1 = stats_above.entropy();
    volatile double H2 = stats_below.entropy();
    volatile double ig = - N1/N * H1 - N2/N * H2;
    if(not goodNumber(ig))
      return -inf;
    else
      return ig;
  }

  ///
  /// StochasticExtremelyRandomTree
  ///
  StochasticExtremelyRandomTree::StochasticExtremelyRandomTree(int depth)  : depth(depth)
  {
    for(int iter = 0; iter < 50; ++iter)
      split_candidates.push_back(make_shared<SplitFunction>());
  }

  unordered_map<Vec3b,long> StochasticExtremelyRandomTree::posterior(const Mat&Z,Point pt) const
  {
    if(splitFn)
      if(splitFn->split(Z,pt))
	return branch_above->posterior(Z,pt);
      else
	return branch_below->posterior(Z,pt);
    
    return local_stats.posterior();    
  }

  size_t StochasticExtremelyRandomTree::total_samples() const
  {
    return local_stats.samples() + 
      ((branch_above != nullptr)?(branch_above->total_samples()):0) + 
      ((branch_below != nullptr)?(branch_below->total_samples()):0);
  }

  Vec3d StochasticExtremelyRandomTree::predict(const Mat&Z,Point pt) const
  {
    if(splitFn)
      if(splitFn->split(Z,pt))
	return branch_above->predict(Z,pt);
      else
	return branch_below->predict(Z,pt);
    
    return local_stats.predict();
  }

  void StochasticExtremelyRandomTree::train(const Mat&sem,const Mat&Z,Point pt)
  {
    if(splitFn)
    {
      if(splitFn->split(Z,pt))
	branch_above->train(sem,Z,pt);
      else
	branch_below->train(sem,Z,pt);
    }
    else if(local_stats.splitable() and depth < MAX_DEPTH) // consider a split
    {      
      double best_info_gain = -inf;
      vector<double> gains;
      for(auto && candidate : split_candidates)
      {
	double gain = candidate->info_gain();	
	gains.push_back(gain);
	if(gain > best_info_gain)
	{
	  best_info_gain = gain;
	  splitFn = candidate;
	}
      }
      branch_above = (make_shared<StochasticExtremelyRandomTree>(depth+1));
      branch_below = (make_shared<StochasticExtremelyRandomTree>(depth+1));

      log_file << safe_printf("StochasticExtremelyRandomTree::train splits! ig = % of ",best_info_gain) << gains << endl;
    }
    else
    {      
      // update the weights
      Vec3b pix = sem.at<Vec3b>(pt.y,pt.x);
      local_stats.update(pix);

      // update the split candidates
      if(depth < MAX_DEPTH)
      {
	for(auto && candidate : split_candidates)
	  candidate->train(sem,Z,pt);
      }
    }
  }

  ///
  /// SECTION: Simple 2D hough forest
  /// 
  HoughForest::HoughForest()
  {
    // init the forest
    for(int iter = 0; iter < 7; ++iter)
      trees.push_back(StochasticExtremelyRandomTree());
  }

  void HoughForest::train_one(Mat&Z,Point2d part_center,TaskBlock&train_trees,Mat seg)
  {
    assert(seg.empty() or seg.type() == DataType<uint8_t>::type);

    // (1) generate
    shared_ptr<Mat> target_image = make_shared<Mat>(Z.rows,Z.cols,DataType<Vec3b>::type);
    for(int yIter = 0; yIter < Z.rows; yIter++)
      for(int xIter = 0; xIter < Z.cols; xIter++)
      {
	float z = Z.at<float>(yIter,xIter);
	int xerror = z*(xIter - part_center.x)/16 + 128;
	int yerror = z*(yIter - part_center.y)/16 + 128;
	target_image->at<Vec3b>(yIter,xIter) = Vec3b(
	  saturate_cast<uint8_t>(xerror),
	  saturate_cast<uint8_t>(yerror),0);
      }

    // (2) train trees to predict the target image
    for(int iter = 0; iter < trees.size(); ++iter)
    {	    
      train_trees.add_callee([&,this,iter,target_image]() 
			     {
			       auto&tree = this->trees.at(iter);
			       for(int yIter = 0; yIter < Z.rows; yIter++)
				 for(int xIter = 0; xIter < Z.cols; xIter++)
				 {
				   int row = thread_rand()%Z.rows;
				   int col = thread_rand()%Z.cols;
					 
				   if(seg.empty() or seg.at<uint8_t>(row,col) <= 100)
				   {					 
				     tree.train(*target_image,Z,Point(col,row));
				   }
				 }
			     });
	  
    }
  }

  Mat HoughForest::predict_one_part(const Mat&seg,const Mat&Z,const CustomCamera&camera) const
  {
    assert(seg.empty() or seg.type() == DataType<uint8_t>::type);
    // (1) form the probability image
    Mat probImage(Z.rows,Z.cols,DataType<double>::type,Scalar::all(0));
    for(int yIter = 0; yIter < Z.rows; yIter++)
      for(int xIter = 0; xIter < Z.cols; xIter++)
      {
	float z = Z.at<float>(yIter,xIter);
	if(seg.empty() or seg.at<uint8_t>(yIter,xIter) <= 100)
	{
	  for(auto && tree : trees)
	  {
	    Vec3d vote = tree.predict(Z,Point(xIter,yIter));
	    int offset_x = 16*(vote[0] - 128)/z;
	    int offset_y = 16*(vote[1] - 128)/z;
	    // pred = xIter - part_center.x;
	    // part_center = xIter - pred
	    if(seg.empty() and 
	       ((xIter - offset_x) < 0 or Z.cols <= (xIter - offset_x) 
		or (yIter - offset_y) < 0 or Z.rows <= (yIter - offset_y)))
	      continue;
	    auto vote_x = [&](){return clamp<int>(0,xIter - offset_x,Z.cols-1);};
	    auto vote_y = [&](){return clamp<int>(0,yIter - offset_y,Z.rows-1);};
	    while(!seg.empty() and seg.at<uint8_t>(vote_y(),vote_x()) > 100)
	    {
	      if((offset_x == 0 and offset_y == 0))
	      {
		cout << safe_printf("% % % % % %",vote_x(),vote_y(),offset_x,offset_y,xIter,yIter) << endl;
		assert(false);
	      }
	      if(offset_x > 0)
		offset_x/=2;
	      if(offset_y > 0)
		offset_y/=2;
	      if(offset_x < 0)
	      {
		offset_x/=2;
		offset_x++;
	      }
	      if(offset_y < 0)
	      {
		offset_y/=2;
		offset_y++;
	      }
	    }
	    probImage.at<double>(vote_y(),vote_x())++;
	  }	  
	}
      }
    // remove sections outside the segmentation
    for(int yIter = 0; yIter < Z.rows; yIter++)
      for(int xIter = 0; xIter < Z.cols; xIter++)
      {
	if(!seg.empty() and seg.at<uint8_t>(yIter,xIter) > 100)
	{
	  probImage.at<double>(yIter,xIter) = -inf;
	}
      }
    return probImage;
  }
}
