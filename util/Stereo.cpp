/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifdef DD_HAVE_MRF_LIBS
 
#include "Stereo.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include "MetaDataKITTI.hpp"
//#include "mrf.h"
#include "ICM.h"
#include "GCoptimization.h"
#include <set>
#include <boost/graph/graph_traits.hpp>
#include "TRW-S.h"
#include "BP-S.h"
#include "MetaDataKITTI.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  static Mat formDepth_stereoSGBM
    (KITTI_Calibration&calib,VelodyneData&point_cloud,Mat&RGB1,Mat&RGB2)
  {
    int P1 = 1; // penality on small disparity changes
    int P2 = 2; // penality on large disparity changes
    StereoSGBM stereo_alg(0,16*16,11,P1,P2);
    Mat disp; stereo_alg(RGB1,RGB2,disp);
    disp.convertTo(disp,DataType<float>::type);
    for(int yIter = 0; yIter < disp.rows; yIter++)
      for(int xIter = 0; xIter < disp.cols; xIter++)
	if(disp.at<float>(yIter,xIter) < 0)
	  disp.at<float>(yIter,xIter) = qnan;
    return disp;
  }
  
  static vector<MRF::CostVal> aggregateDSI(const std::vector<MRF::CostVal>&dsi,int rows, int cols,int numDisps)
  {
    // initialize the aggregation structures
    std::vector<MRF::CostVal> agg_dsi(dsi.size(),0);
    vector<double> Ns(dsi.size(),0);
    
    // apply a box filter to remove some noise
    int window_side_length = 2;
    for(int yIter = 0; yIter < rows; yIter++)
      for(int xIter = 0; xIter < cols; xIter++)
	for (int disparity = 0; disparity < numDisps; disparity++)    
	  for(int y2 = yIter - window_side_length/2; y2 <= yIter + window_side_length/2; y2++)
	    for(int x2 = xIter - window_side_length/2; x2 <= xIter + window_side_length/2; x2++)
	      if(0 <= y2 && y2 < rows && 0 <= x2 && x2 < cols)
	      {
		int outIdx = disparity + xIter * numDisps + yIter * numDisps * cols;
		int inIdx  = disparity + x2 * numDisps + y2 * numDisps * cols;
		agg_dsi[outIdx] += dsi[inIdx];
		Ns[outIdx] ++;
	      }
	  
    // renormalize
    for(int yIter = 0; yIter < rows; yIter++)
      for(int xIter = 0; xIter < cols; xIter++)
	for (int disparity = 0; disparity < numDisps; disparity++)    
	{
	  int index = disparity + xIter * numDisps + yIter * numDisps * cols;
	  agg_dsi[index] /= Ns[index];
	}
	
    // implement shiftable windows
    int shitable_window_side_len = 2;
    for(int yIter = 0; yIter < rows; yIter++)
      for(int xIter = 0; xIter < cols; xIter++)
	for (int disparity = 0; disparity < numDisps; disparity++)    
	  for(int y2 = yIter - shitable_window_side_len/2; y2 <= yIter + shitable_window_side_len/2; y2++)
	    for(int x2 = xIter - shitable_window_side_len/2; x2 <= xIter + shitable_window_side_len/2; x2++)
	      if(0 <= y2 && y2 < rows && 0 <= x2 && x2 < cols)
	      {
		int outIdx = disparity + xIter * numDisps + yIter * numDisps * cols;
		int inIdx  = disparity + x2 * numDisps + y2 * numDisps * cols;
		agg_dsi[outIdx] = std::min(agg_dsi[outIdx],dsi[inIdx]);
	      }
	
    return agg_dsi;
  }
  
  static std::vector<MRF::CostVal> calcDSI(
    const Mat&rgb1, const Mat&rgb2, int numDisp)
  {
    std::vector<MRF::CostVal> dsi(rgb1.size().area() * numDisp);
    bool birchfield = false;
    bool squaredDiffs = 0;
    int truncDiffs  = 255;
    int nColors = 3;
    
    // worst value for sumdiff below 
    int worst_match = 3 * (squaredDiffs ? 255 * 255 : 255);
    // truncation threshold - NOTE: if squared, don't multiply by nColors (Eucl. dist.)
    int maxsumdiff = squaredDiffs ? truncDiffs * truncDiffs : nColors * abs(truncDiffs);
    // value for out-of-bounds matches
    int badcost = std::min(worst_match, maxsumdiff);    

    // compute the DSI
    int dsiIndex = 0;
    for(int yIter = 0; yIter < rgb1.rows; yIter++)
      for(int xIter = 0; xIter < rgb1.cols; xIter++)
      {
	// adopted from the MSR MRF library
	Vec3b pix1 = rgb1.at<Vec3b>(yIter,xIter);
	for (int disparity = 0; disparity < numDisp; disparity++) {
	    int x2 = xIter - disparity;
	    int dsiValue;
	    
	    if (x2 >= 0 && disparity < numDisp) 
	    { // in bounds
		Vec3b pix2 = rgb2.at<Vec3b>(yIter,x2);
		int sumdiff = 0;
		for (int b = 0; b < 3; b++) {
		    int diff = 0;
		    if (birchfield) {
			// Birchfield/Tomasi cost
			int im1c = pix1[b];
			int im1l = xIter == 0?   im1c : (im1c + pix1[b - 3]) / 2;
			int im1r = xIter == rgb1.cols-1? im1c : (im1c + pix1[b + 3]) / 2;
			int im2c = pix2[b];
			int im2l = x2 == 0?   im2c : (im2c + pix2[b - 3]) / 2;
			int im2r = x2 == rgb1.cols-1? im2c : (im2c + pix2[b + 3]) / 2;
			int min1 = std::min(im1c, std::min(im1l, im1r));
			int max1 = std::max(im1c, std::max(im1l, im1r));
			int min2 = std::min(im2c, std::min(im2l, im2r));
			int max2 = std::max(im2c, std::max(im2l, im2r));
			int di1 = std::max(0, std::max(im1c - max2, min2 - im1c));
			int di2 = std::max(0, std::max(im2c - max1, min1 - im2c));
			diff = std::min(di1, di2);
		    } else {
			// simple absolute difference
			int di = static_cast<int>(pix1[b]) - static_cast<int>(pix2[b]);
			diff = abs(di);
		    }
		    // square diffs if requested (Birchfield too...)
		    sumdiff += (squaredDiffs ? diff * diff : diff);
		}
		// truncate diffs
		dsiValue = std::min(sumdiff, maxsumdiff);
	    } else { // out of bounds: use maximum truncated cost
		dsiValue = badcost;
	    }
	    //int x0=-140, y0=-150;
	    //if (x==x0 && y==y0)
	    //    printf("dsi(%d,%d,%2d)=%3d\n", x, y, d, dsiValue); 

	    // The cost of pixel p and label l is stored at dsi[p*nLabels+l]
	    dsi[dsiIndex++] = dsiValue+1;
	}	
      }
    require_equal<int>(dsi.size(),dsiIndex);
    
    return dsi;
    std::vector<MRF::CostVal> agg_dsi = aggregateDSI(dsi,rgb1.rows, rgb1.cols,numDisp);
    return agg_dsi;
  }
  
  static constexpr int numDisp = 256;
  
  static Mat formDepth_stereoWTA
      (const KITTI_Calibration&calib,Mat&RGB1,Mat&RGB2)
  {
    std::vector<MRF::CostVal> dsi = calcDSI(RGB1, RGB2,numDisp);
    
    Mat depthImage(RGB1.rows,RGB1.cols,DataType<float>::type,Scalar::all(qnan));
    for(int yIter =0; yIter < RGB1.rows; yIter++)
      for(int xIter = 0; xIter < RGB1.cols; xIter++)
      {
	float min_cost = inf;
	
	for(int disp = 0; disp < numDisp; disp++)
	{
	  int dis_cost = dsi[disp + xIter * numDisp + yIter * RGB1.cols * numDisp];
	  if(dis_cost < min_cost)
	  {
	    min_cost = dis_cost;
	    depthImage.at<float>(yIter,xIter) = calib.disp2depth(disp);
	  }
	}
      }
      
    //Mat depthImage; reprojectImageTo3D(dispImage,depthImage,calib.Q,true);  
    return depthImage;    
  }
  
  static Mat formDepth_stereoMRF
    (const KITTI_Calibration&calib,Mat&RGB1,Mat&RGB2,Mat fixed_depths = Mat())
  {
    // STEP1A : Setup the energy functions
    std::vector<MRF::CostVal> dsi = calcDSI(RGB1, RGB2,numDisp);
    //dsi = vector<MRF::CostVal>(dsi.size(),0);
    if(!fixed_depths.empty())
    {
      set<int> targets;
      for(int yIter = 0; yIter < RGB1.rows; yIter++)
	for(int xIter = 0; xIter < RGB1.cols; xIter++)
	{
	  float& fixed_depth = fixed_depths.at<float>(yIter,xIter);
	  if(!goodNumber(fixed_depth))
	    continue;
	  int target_disparity = clamp<int>(0,calib.depth2disp(fixed_depth),numDisp-1);
	  if(target_disparity < 0 || target_disparity >= numDisp)
	    continue;
	  
	  targets.insert(target_disparity);
	  //cout << printfpp("fixed %d %d %d",yIter,xIter,target_disparity) << endl;
	  
	  for (int disparity = 0; disparity < numDisp; disparity++)
	  {
	    MRF::CostVal&cost_here = dsi[disparity + xIter * numDisp + yIter * RGB1.cols * numDisp] ;
	    if(disparity == target_disparity)
	    {
	      cost_here = 0;
	    }
	    else
	      cost_here = 255 * 250;
	  }
	}
      cout << "targeted disparties: " << vector<int>(targets.begin(),targets.end()) << endl;
    }
    unique_ptr<DataCost> dcost(new DataCost(&dsi[0]));
    
    // STEP1B : Set up the smoothness terms
    int smoothexp = 1; // default 1
    int lambda = 75; // default 20?
    int smoothmax = 16; // default 2
    unique_ptr<SmoothnessCost> scost(new SmoothnessCost(smoothexp, smoothmax, lambda));
    unique_ptr<EnergyFunction> energy(new EnergyFunction(dcost.get(), scost.get()));
    
    // STEP2 : Invoke the optimization algorithm
    // Expansion, ICM, Swap, TRWS, MaxProdBP
    unique_ptr<MRF> mrf(
      new Swap(RGB1.cols, RGB1.rows, numDisp, energy.get()));
    mrf->initialize();
    mrf->clearAnswer();
    for(int iter = 0; iter < 2; ++iter)
    {
      float time; mrf->optimize(1,time);
      MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
      MRF::EnergyVal E_data   = mrf->dataEnergy();
      printf("Total Energy = %d (Smoothness energy %d, Data Energy %d)\n", E_smooth+E_data,E_smooth,E_data);
    }
    
    // STEP3: Convert the results to a disparity image
    Mat depthImage(RGB1.rows,RGB1.cols,DataType<float>::type,Scalar::all(qnan));
    int n = 0;
    for(int yIter =0; yIter < RGB1.rows; yIter++)
      for(int xIter = 0; xIter < RGB1.cols; xIter++)
      {
	int label = mrf->getLabel(n++);
	depthImage.at<float>(yIter,xIter) = calib.disp2depth(label);
      }
    return depthImage;
  }
  
  std::function<double (double)> findDepthMap_lr(
    const Mat&m1, const Mat&m2)
  {
    // find the mapping using linear regression
    Mat X(0,1,DataType<double>::type,Scalar::all(0));
    Mat Y(0,1,DataType<double>::type,Scalar::all(0));
    for(int rIter = 0; rIter < m1.rows; rIter++)
      for(int cIter = 0; cIter < m1.cols; cIter++)
      {
	double n1 = m1.at<float>(rIter,cIter);
	double n2 = m2.at<float>(rIter,cIter);
	if(goodNumber(n1) && goodNumber(n2))
	{
	  Mat y(vector<double>{n1},true);
	  Mat x(vector<double>{n2},true);
	  X.push_back<double>(x.t());
	  Y.push_back<double>(y);
	  //cout << printfpp("x = %f y = %f",n1,n2) << endl;
	}
      }
    Mat tmp; cv::invert(X.t()*X,tmp);
    //cout << "sizes = " << tmp.size() << " " << X.size() << " " << Y.size() << endl;
    Mat beta = tmp * X.t() * Y;
    //cout << "beta.size = " << beta.size() << endl;
    //cout << "beta = " << beta << endl;
    require_equal<int>(beta.size().area(),1);
    //require_equal<int>(beta.rows,1);
    //require_equal<int>(beta.cols,2);
    double beta_x = beta.at<double>(0);
    log_file << "findDepthMap_lr: " << "y = " << beta_x << "*x" << endl;

    return [beta_x](double n2){return beta_x*(n2);};
  }
  
  // Uses RANSAC
  std::function<double (double)> findDepthMap(
    const Mat&m1, const Mat&m2) 
  {
    auto mapFn = findDepthMap_lr(m1,m2);
    
    for(int iter = 0; iter < 25; ++iter)
    {
      // STEP1: find the residuals
      vector<double> residuals;
      for(int rIter = 0; rIter < m1.rows; rIter++)
	for(int cIter = 0; cIter < m1.cols; cIter++)
	{
	  double n1 = m1.at<float>(rIter,cIter);
	  double n2 = m2.at<float>(rIter,cIter);	
	  if(goodNumber(n1) && goodNumber(n2))
	  {
	    double err = n1-mapFn(n2);
	    residuals.push_back(err*err);
	  }
	}
      auto nth_iter = (residuals.begin()+residuals.size()*1.0/25);
      std::nth_element(residuals.begin(),nth_iter,residuals.end());
      double median_residual = *(nth_iter);
      cout << "median_residual = " << median_residual << endl;
      
      // STEP2: mask out the badly fit points
      Mat m1p = m1.clone();
      for(int rIter = 0; rIter < m1.rows; rIter++)
	for(int cIter = 0; cIter < m1.cols; cIter++)
	{
	  double n1 = m1.at<float>(rIter,cIter);
	  double n2 = m2.at<float>(rIter,cIter);	
	  
	  if(goodNumber(n1) && goodNumber(n2))
	  {
	    double err = n1-mapFn(n2);
	    if(err*err > median_residual)
	      m1p.at<float>(rIter,cIter) = qnan;
	  }
	}
	
      // STEP3: Update the map function
      mapFn = findDepthMap_lr(m1p,m2);
    }
      
    return mapFn;
  }
  
  static Mat merge_depth_maps(const Mat&m1, const Mat&m2)
  {
    assert(m1.size() == m2.size());
    assert(m1.type() == DataType<float>::type);
    assert(m2.type() == DataType<float>::type);
    Mat result = m1.clone();
    Mat isInvalid(result.rows,result.cols,DataType<float>::type,Scalar::all(0));
    
    // 1: Find points with depth in both images to learn a mapping
    // function (use linear regression)
    //auto mapfn = findDepthMap(m1, m2);
    
    // 2: Use the mapping function to merge points into a single image
    for(int rIter = 0; rIter < m1.rows; rIter++)
      for(int cIter = 0; cIter < m1.cols; cIter++)
      {
	double n1 = m1.at<float>(rIter,cIter);
	double n2 = m2.at<float>(rIter,cIter);	
	if(goodNumber(n1))
	  result.at<float>(rIter,cIter) = n1;
	else if(goodNumber(n2))
	  result.at<float>(rIter,cIter) = n2;//mapfn(n2);
	else
	  isInvalid.at<float>(rIter,cIter) = 1;
      }
    
    return fillDepthHoles(result,isInvalid);
  }
  
  static Mat formDepth_joint
    (KITTI_Calibration&calib,VelodyneData&point_cloud,Mat&RGB1,Mat&RGB2)
  {
    // the LiDAR map is fixed
    Mat fixed_depths;
    const Mat depth_lidar = formDepth_lidar(calib,point_cloud,RGB1,RGB2);
    Mat depth_stereo;// = formDepth_stereoMRF(calib,RGB1,RGB2,fixed_depths);
    
    for(int iter = 0; iter < 1; ++iter)
    {
      // map lidar depths to stereo
      // the mapfn should be the same for all functions???
      //auto mapfn = findDepthMap(depth_stereo,depth_lidar);
     /* Extrema ex_lidar  = extrema(depth_lidar);
      cout << "LiDAR range: " << ex_lidar.min << " to " << ex_lidar.max << endl;
      auto mapfn = [ex_lidar](double v)
      {
	return interpolate_linear(v,ex_lidar.min,ex_lidar.max,0,numDisp-1) - numDisp;
      }; */     
      
      fixed_depths = depth_lidar.clone();
//       for(int rIter = 0; rIter < fixed_depths.rows; rIter++)
// 	for(int cIter = 0; cIter < fixed_depths.cols; cIter++)
// 	{
// 	  float&curf = fixed_depths.at<float>(rIter,cIter);
// 	  if(goodNumber(curf))
// 	    curf = mapfn(curf);
// 	}      
      
      // compute the initial maps with no constraints
      depth_stereo = formDepth_stereoMRF(calib,RGB1,RGB2,fixed_depths);
    }
    
    return depth_stereo;
  }
  
}

#endif
