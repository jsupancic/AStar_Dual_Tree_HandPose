/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "Skin.hpp"
#include "params.hpp"
#include <boost/filesystem.hpp>
#include <mutex>
#include "util_real.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  using boost::filesystem::path;
  
  /// SECTION Histogram computation
  void skin_make_hist()
  {
    path cwd = boost::filesystem::current_path();
    cout << "computing a histogram" << endl;
    
    // initialize the histograms.
    MatND fg_hist, bg_hist;
    
    for(int iter = 0; iter <= 554; iter++)
    {
      char cur_image_path[512];
      char cur_mask_path[512];
      snprintf(cur_image_path,512,"../ibtd/%04d.jpg",iter);
      snprintf(cur_mask_path,512,"../ibtd/Mask/%04d.bmp",iter);
      cout << cur_image_path << endl;
      cout << cur_mask_path << endl;
      
      Mat image = imread(cur_image_path);
      Mat mask, inv_mask = imread(cur_mask_path,0);
      threshold(inv_mask,mask,254,255,cv::THRESH_BINARY_INV);
      threshold(mask,inv_mask,254,255,cv::THRESH_BINARY_INV);
      //cout << mask.channels() << endl;
      //cout << mask.rows << " " << mask.cols << endl;
      //cout << image.rows << " " << image.cols << endl;
      //imshow("test",image); cvWaitKey(0);
      //cout << mask;/*imshow("Making hist for Image",mask); */ cvWaitKey(0);
      //cout << inv_mask; /* imshow("Making hist for Image",inv_mask); */ cvWaitKey(0);
      //cvWaitKey(1);
      
      // add to the histogram...
      calcHist(&image,1,params::channels,mask,fg_hist,3,params::histSz,params::ranges,true,true);
      calcHist(&image,1,params::channels,inv_mask,bg_hist,3,params::histSz,params::ranges,true,true);
    }
    
    // finally, save the histogram.
    normalize(fg_hist,fg_hist,1,0,cv::NORM_L1);
    normalize(bg_hist,bg_hist,1,0,cv::NORM_L1);
    FileStorage skin_file("skin_hist.yml", FileStorage::WRITE);
    skin_file << "fg_hist" << fg_hist;
    skin_file <<  "bg_hist" << bg_hist;
    skin_file.release();
  }  
  
  static Mat skin_hist, bg_hist;
  
  static void load_skin_hists()
  {
    // access the histograms in a CRITICAL SECTION
    if(!skin_hist.empty() && !bg_hist.empty())
      return;
      
    static mutex m; unique_lock<mutex> l(m);
    if(skin_hist.empty() || bg_hist.empty())
    {
      FileStorage skinHists("skin_hist.yml", FileStorage::READ);
      assert(skinHists.isOpened());
      skinHists["fg_hist"] >> skin_hist;
      skinHists["bg_hist"] >> bg_hist;
      skinHists.release();    
    }
  }
  
  double skin_likelihood(Vec3b color)
  {
    load_skin_hists();
    
    int num_bins = params::histSz_1;
    int bin_size = 256/num_bins;
    
    double pSkin = skin_hist.at<float>(color[0]/bin_size,color[1]/bin_size,color[2]/bin_size);
    double pBg   =   bg_hist.at<float>(color[0]/bin_size,color[1]/bin_size,color[2]/bin_size);
    pSkin += 1e-6;
    pBg   += 1e-6;
    return pSkin/pBg;
  }
  
  Mat skin_detect(const Mat&RGB,Mat&pSkin, Mat&pBG)
  {
    load_skin_hists();
      
    // back project the histogram onto the image
    Mat fRGB; RGB.convertTo(fRGB,DataType<Vec3f>::type);
    calcBackProject(&fRGB,1,params::channels,skin_hist,pSkin,params::ranges);
    calcBackProject(&fRGB,1,params::channels,bg_hist,pBG,params::ranges);
    pSkin.convertTo(pSkin,DataType<double>::type);
    pBG.convertTo(pBG,DataType<double>::type);
    pSkin += 1e-6;
    pBG += 1e-6;
    //return pBG;
    
    // Pr(skin | color) = Pr(color | skin) * Pr(Skin)/Pr(Color)
    // compute the likelihood ratio assumming equal likelihood
    Mat likelihood = pSkin/pBG;
    
    // debug the above
//     for(int rIter = 0; rIter < likelihood.rows; rIter++)
//       for(int cIter = 0; cIter < likelihood.cols; cIter++)
//       {
// 	Vec3b color = RGB.at<Vec3b>(rIter,cIter);
// 	likelihood.at<double>(rIter,cIter) = skin_likelihood(color);
//       }
      
    return likelihood;
    
    //imageeq("pSkin",pSkin);
    //imageeq("pBG",pBG);
    //imageeq("Skin Likelihood",likelihood);
    //waitKey_safe(0);  
  }
  
  ///
  /// SECTION: SkinFeatureComputer
  /// 
  int SkinFeatureComputer::cellsPerBlock()
  {
    return 1;
  }

  void SkinFeatureComputer::compute(ImRGBZ& im, vector< float >& feats)
  {
    // get the FG and BG probabilities
    Mat pFG, pBG;
    skin_detect(im.RGB,pFG,pBG);
    
    // generate the feature
    feats = vector<float>(getDescriptorSize(),0);
    for(int block_y = 0; block_y < blocks_y(); block_y++)
      for(int block_x = 0; block_x < blocks_x(); block_x++)
	for(int im_y = block_y*cell_size.height; im_y < (block_y+1)*cell_size.height; im_y++)
	  for(int im_x = block_x*cell_size.width; im_x < (block_x+1)*cell_size.width; im_x++)
	  {
	    // compute the bin indexes
	    int fg_bin = clamp<int>(0,FG_BINS*pFG.at<float>(im_y,im_x),FG_BINS-1);
	    int bg_bin = clamp<int>(0,BG_BINS*pBG.at<float>(im_y,im_x),BG_BINS-1);
	    
	    // write the feature
	    feats[getIndex(block_x,block_y,0,fg_bin)] = weight;
	    feats[getIndex(block_x,block_y,0,FG_BINS + bg_bin)] = weight;
	  }
  }

  Size SkinFeatureComputer::getBlockSize()
  {
    return getCellSize();
  }
  
  Size SkinFeatureComputer::getBlockStride()
  {
    return getCellSize();
  }

  Size SkinFeatureComputer::getCellSize()
  {
    return cell_size;
  }

  int SkinFeatureComputer::getNBins()
  {
    return BG_BINS + FG_BINS;
  }
  
  size_t SkinFeatureComputer::getDescriptorSize()
  {
    return getNBins()*blocks_x()*blocks_y();
  }

  Size SkinFeatureComputer::getWinSize()
  {
    return win_size;
  }

  SkinFeatureComputer::SkinFeatureComputer(
    Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    win_size(win_size), cell_size(cell_size)
  {
    //assert(block_size == cell_size);
    //assert(block_stride == cell_size);
  }
  
  SkinFeatureComputer::SkinFeatureComputer(SkinFeatureComputer&other) : 
    win_size(other.win_size), cell_size(other.cell_size)
  {
  }

  Mat SkinFeatureComputer::show(const string& title, vector< double > feat)
  {
    vector<FeatVis> vis = show_planes(feat);
    return vis[0].getPos();
  }

  vector< FeatVis > SkinFeatureComputer::show_planes(vector< double > feat)
  {
    // extract the feature
    Mat mode_positive = Mat(blocks_y(),blocks_x(),DataType<Vec3b>::type,Scalar::all(0));
    Mat mode_negative = Mat(blocks_y(),blocks_x(),DataType<Vec3b>::type,Scalar::all(0));
    for(int block_y = 0; block_y < blocks_y(); block_y++)
      for(int block_x = 0; block_x < blocks_x(); block_x++)
      {
	// get write the pos bin
	double max_bin_weight = -inf;
	int mode_fg_bin = 0;
	for(int fg_bin = 0; fg_bin < FG_BINS; ++fg_bin)
	{
	  double bin_weight = feat[getIndex(block_x,block_y,0,fg_bin)];
	  if(bin_weight > max_bin_weight)
	  {
	    max_bin_weight = bin_weight;
	    mode_fg_bin = fg_bin;
	  }
	}
	double i = mode_fg_bin*255.0/(FG_BINS-1);
	mode_positive.at<Vec3b>(block_y,block_x) = Vec3b(i,i,i);
	
	// get write the negative bin
	max_bin_weight = -inf;
	int mode_bg_bin = 0;
	for(int bg_bin = 0; bg_bin < BG_BINS; ++bg_bin)
	{
	  double bin_weight = feat[getIndex(block_x,block_y,0,FG_BINS + bg_bin)];
	  if(bin_weight > max_bin_weight)
	  {
	    max_bin_weight = bin_weight;
	    mode_bg_bin = bg_bin;
	  }
	}
	i = mode_bg_bin*255.0/(BG_BINS-1);
	mode_negative.at<Vec3b>(block_y,block_x) = Vec3b(i,i,i);	
      }    
    
    FeatVis vis("SkinFeatureComputer");
    vis.setPos(mode_positive);
    vis.setNeg(mode_negative);
    return vector<FeatVis>{vis};
  }
  
  string SkinFeatureComputer::toString() const
  {
    return "SkinFeatureComputer";
  }
}

