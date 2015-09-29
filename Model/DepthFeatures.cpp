/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "DepthFeatures.hpp"
#include "params.hpp"
#include "util.hpp"
#include "vec.hpp"
#include "Probability.hpp"
#include <cmath>
#include <math.h>
#include <boost/graph/graph_concepts.hpp>
#include "Log.hpp"

namespace deformable_depth
{    
  /// SECTION: HOG General concrete methods  
  int DepthFeatComputer::blocks_x()
  {
    return 1 + (getWinSize().width-getBlockSize().width)/getBlockStride().width;
  }

  int DepthFeatComputer::blocks_y()
  {
    return 1 + (getWinSize().height-getBlockSize().height)/getBlockStride().height;
  }
  
  string DepthFeatComputer::toString() const
  {
    return "Unknown";
  }
  
  vector< FeatVis > DepthFeatComputer::show_planes(std::vector< double > feat)
  {
    assert(false); // this is so bad it should never be called.
    // this is a filler operation which does something extremly basic...
    // it will most likely need to be replaced (overriden).
    FeatVis vis("DepthFeatComputer(BAD)");
    vis.setPos(show("",feat));
    vector<FeatVis> result;
    result.push_back(vis);
    return result;
  }
  
  shared_ptr<const ImRGBZ> cropToCells(const ImRGBZ& im,int cellx, int celly)
  {
    // enforce pre-conditions
    require_equal<int>(im.RGB.type(),DataType<Vec3b>::type);
    
    // exploit C's round towards 0 behavior.
    Point tl(0,0);
    Size  sz((im.RGB.cols/cellx)*cellx, (im.RGB.rows/celly)*celly);
    Rect crop_region(tl,sz);
    const ImRGBZ im_crop = im(crop_region);
    return shared_ptr<const ImRGBZ>(new const ImRGBZ(im_crop));        
  }
  
  shared_ptr<const ImRGBZ> DepthFeatComputer::cropToCells(const ImRGBZ& im)
  {
    // crop image to a multiple of cell size.
    int cellx = getCellSize().width;
    int celly = getCellSize().height; 
    return deformable_depth::cropToCells(im,cellx,celly);
  }
  
  /// SECTION: RGB HOG Feature Implementeation
  void HOGComputer_RGB::compute(const ImRGBZ&im, std::vector< float >& feats) 
  {
    return HOGDescriptor::compute(im.RGB,feats);
  }
  
  int HOGComputer_RGB::cellsPerBlock()
  {
    return 4;
  }
  
  size_t HOGComputer_RGB::getDescriptorSize()
  {
    return HOGDescriptor::getDescriptorSize();
  }

  Size HOGComputer_RGB::getBlockStride() 
  {
    return blockStride;
  }

  Size HOGComputer_RGB::getCellSize() 
  {
    return cellSize;
  }

  int HOGComputer_RGB::getNBins() 
  {
    return nbins;
  }

  Size HOGComputer_RGB::getWinSize() 
  {
    return winSize;
  }
  
  Mat HOGComputer_RGB::show(const string& title, std::vector< double > feat)
  {
    return imagehog(title,*this,feat);
  }
  
  string HOGComputer_RGB::toString() const
  {
    return "RGBHoG";
  }
  
  Size HOGComputer_RGB::getBlockSize()
  {
    return blockSize;
  }

  
  /// SECTION: Depth HOG Feature Implementation
  Size HOGComputer18x4_General::getBlockStride()
  {
    return block_stride;
  }

  Size HOGComputer18x4_General::getCellSize()
  {
    return cell_size;
  }

  size_t HOGComputer18x4_General::getDescriptorSize()
  {
    int cells_x = win_size.width/cell_size.width;
    int cells_y = win_size.height/cell_size.height;
    int blocks_x = cells_x - 1;
    int blocks_y = cells_y - 1;
    return blocks_x*blocks_y*nbins*cellsPerBlock();
  }

  int HOGComputer18x4_General::getNBins()
  {
    return nbins;
  }

  Size HOGComputer18x4_General::getWinSize()
  {
    return win_size;
  }
  
  int HOGComputer18x4_General::cellsPerBlock()
  {
    return 4;
  }
  
  HOGComputer18x4_General::HOGComputer18x4_General
  (im_fun use_fun,Size win_size, Size block_size, Size block_stride, Size cell_size) :
    use_fun(use_fun),
    win_size(win_size), block_size(block_size), 
    block_stride(block_stride), cell_size(cell_size),
    nbins(params::ORI_BINS)
  {
    if(!contrast_sensitive)
      nbins /= 2;
  }

  HOGComputer18x4_General::HOGComputer18x4_General(HOGComputer18x4_General& other) :
    use_fun(other.use_fun),
    win_size(other.win_size),
    block_size(other.block_size), block_stride(other.block_stride),
    cell_size(other.cell_size), nbins(other.nbins)
  {
  }
  
  typedef vector<vector<vector<double> > > HoGCell_Storage;
  
  HoGCell_Storage compute_hog_cells(const Mat&im,int cells_x,int cells_y,Size cell_size,int nbins)
  {
    /// STEP 1:  get gradient magnitudes and orientations
    Mat dx; filter2D(im,dx,CV_32F,params::dxFilter);
    Mat dy; filter2D(im,dy,CV_32F,params::dyFilter);
    assert(dx.type() == DataType<float>::type);
    assert(dy.type() == DataType<float>::type);    
    
    /// STEP 2: Bin into cells
    HoGCell_Storage // cell_y, cell_x, bin
      cells(cells_y,
	   vector<vector<double> >(cells_x,
			  vector<double>(nbins,0)));
    for(int yIter = 0; yIter < im.rows; yIter++)
      for(int xIter = 0; xIter < im.cols; xIter++)
      {
	// spatial bin
	int cell_x = xIter/cell_size.width;
	int cell_y = yIter/cell_size.height;
	
	// gradient value
	double dxx = dx.at<float>(yIter,xIter);
	double dyy = dy.at<float>(yIter,xIter);
	if(!goodNumber(dxx) || !goodNumber(dyy))
	  continue;
	
	// orientation
	int bin = binOf(dxx,dyy);
	while(bin >= nbins)
	  bin -= nbins;
	
	// magnitude
	double mag = std::sqrt(dxx*dxx + dyy*dyy);
	// if RGB, we don't normalize as such
	if(im.type() == DataType<float>::type)
	{
	  if(!goodNumber(im.at<float>(yIter,xIter)))
	    continue;
	  mag /= clamp<float>(
	    params::MIN_Z(),
	    im.at<float>(yIter,xIter), // dx/dy per pixel changes with depth.
	    params::MAX_Z());
	  mag = std::sqrt(mag); // try to add invariance to steps? improves 29% to 32%
	  if(!goodNumber(mag))
	    continue;
	}
	
	// update
	assert(goodNumber(mag));
	assert(0 <= cell_y);
	assert(cell_y < cells.size());
	assert(0 <= cell_x);
	assert(cell_x < cells[cell_y].size());
	assert(0 <= bin);
	assert(bin    < cells[cell_y][cell_x].size());
	cells[cell_y][cell_x][bin] += mag;
      }    
      
    return cells;
  }
  
  void HOGComputer18x4_General::compute(const ImRGBZ& im, std::vector< float >& feats)
  {
    // step 1: extract features
    int cells_x = win_size.width/cell_size.width;
    int cells_y = win_size.height/cell_size.height;
    int blocks_x = cells_x - 1;
    int blocks_y = cells_y - 1;    
    
    // step 2: Spatially bin into cells
    HoGCell_Storage cells = compute_hog_cells(use_fun(im),cells_x,cells_y,cell_size,nbins);
	
    // step 3: normalize and clip/truncate
    feats = vector<float>(getDescriptorSize(),0); 
    for(int block_rIter = 0; block_rIter < blocks_y; block_rIter++)
      for(int block_cIter = 0; block_cIter < blocks_x; block_cIter++)
      {
	// four cells per block.
	const double eps  = 0.0001; // avoid div by zero;
	double N = std::sqrt(
	  dot_self(cells[block_rIter][block_cIter]) + 
	  dot_self(cells[block_rIter+1][block_cIter]) + 
	  dot_self(cells[block_rIter][block_cIter+1])+
	  dot_self(cells[block_rIter+1][block_cIter+1]));
	N += eps;
	
	for(int cellIter = 0; cellIter < cellsPerBlock(); cellIter++)
	{
	  // 0: x + 0; y + 0
	  vector<double>&cell = cells[block_rIter][block_cIter];
	  // 1: x + 0; y + 1
	  if(cellIter == 1)
	    cell = cells[block_rIter+1][block_cIter];
	  // 2: x + 1; y + 0
	  else if(cellIter == 2)
	    cell = cells[block_rIter][block_cIter+1];
	  // 3: x + 1; y + 1
	  else if(cellIter == 3)
	    cell = cells[block_rIter+1][block_cIter+1];
	  for(int binIter = 0; binIter < getNBins(); binIter++)
	  {
	    int index = getIndex(block_cIter,block_rIter,cellIter,binIter);
	    assert(index < feats.size() && index >= 0);
	    float&featVal = feats[index];
	    featVal = cell[binIter]/N; // noramlize
	    featVal = std::min<double>(featVal,.2); // clip/truncate
	    assert(goodNumber(featVal));
	  }
	}
      }
    
    // try to show the computed image...
//     #pragma omp critical
//     {
//       imagehog("HOG Features",*this,vec_f2d(feats)); cvWaitKey(0);
//     }
  }  
  
  vector< FeatVis > HOGComputer18x4_General::show_planes(std::vector< double > feat)
  {
    FeatVis vis = picture_HOG_pn(*this,feat,PictureHOGOptions(contrast_sensitive,true));
    return vector<FeatVis>{vis};
  }
  
  Mat HOGComputer18x4_General::show(const string& title, std::vector< double > feat)
  {
    return imagehog(title,*this,feat,PictureHOGOptions(contrast_sensitive,true));
  }
  
  string HOGComputer18x4_General::toString() const
  {
    return "DepthHOG";
  }
  
  Size HOGComputer18x4_General::getBlockSize()
  {
    return block_size;
  }
    
  ///
  ///
  /// SECTION: Hist of Normals Computer Imlementation
  ///
  ///
  int HistOfNormals::cellsPerBlock()
  {
    return 1;
  }

  static float focus_angular_range(float angle)
  {
    float POW = 1; //1/40.0f;
    float pi = params::PI;
    float sign = (angle - pi/2 > 0)?+1:-1;
    return pi/2+.5*pi/::pow(.5*pi,POW)*sign*::pow(::abs(angle-pi/2),POW);
  }
  
  void HistOfNormals::compute(const ImRGBZ& im, std::vector< float >& feats)
  {
    // start by taking the first derivative
    Mat DX; filter2D(im.Z,DX,-1,params::dxFilter01);
    Mat DY; filter2D(im.Z,DY,-1,params::dyFilter01);
    
    // init the cell hists to be empty
    vector<Mat> cell_hists;
    for(int bin = 0; bin < theta_bins*phi_bins; bin++)
      cell_hists.push_back(Mat(blocks_y(),blocks_x(),DataType<float>::type,Scalar::all(0)));
    
    // for bin iterpolation
    double sigma = 1;
    Mat G_theta = getGaussianKernel(phi_bins,sigma,CV_32F);
    Mat G_phi   = getGaussianKernel(theta_bins,sigma,CV_32F);
    
    // 
    for(int yIter = 0; yIter < im.Z.rows; yIter++)
      for(int xIter = 0; xIter < im.Z.cols; xIter++)
      {
	// 
	
	// compute the bins
	float dx = DX.at<float>(yIter,xIter);
	float dy = DY.at<float>(yIter,xIter);
	float phi = ::atan(dy/dx);
	float theta = ::atan(::sqrt(::pow(dx,2)+::pow(dy,2)));
	phi = focus_angular_range(phi);
	theta = focus_angular_range(theta);
	// atan range 0 to PI
	int hard_bin_phi = clamp<int>(0,phi_bins*phi/params::PI,phi_bins-1);
	int hard_bin_theta = clamp<int>(0,theta_bins*theta/params::PI,theta_bins-1);
	// update the histogram
	int cellX = xIter/cell_size.width;
	int cellY = yIter/cell_size.height;
	
	// update directly
	int bin = hard_bin_phi*theta_bins + hard_bin_theta;
	cell_hists[bin].at<float>(cellY,cellX) += 1.0f/cell_size.area();
	
	// use interpolation
	//int phi_lo = ::floor(phi_bins/params::PI*phi-.5);
	//int phi_hi = ::ceil(phi_bins/params::PI*phi+.5);
	//int theta_lo = ::floor(theta_bins/params::PI*theta-.5);
	//int theta_hi = ::ceil(theta_bins/params::PI*theta+.5);
	// lows are possibly invalid.
      }
      
    // write out the feature
    feats = vector<float>(getDescriptorSize(),0); 
    for(int blockY = 0; blockY < blocks_y(); blockY++)
      for(int blockX = 0; blockX < blocks_x(); blockX++)
	for(int bin = 0; bin < getNBins(); bin++)
	{
	  float&in  = cell_hists[bin].at<float>(blockY,blockX);
	  float&out = feats[getIndex(blockX,blockY,0,bin)];
	  //cout << "writing " << in << endl;
	  out = in;  
	}
  }

  Size HistOfNormals::getBlockSize()
  {
    return cell_size;
  }
  
  Size HistOfNormals::getBlockStride()
  {
    return cell_size;
  }

  Size HistOfNormals::getCellSize()
  {
    return cell_size;
  }

  size_t HistOfNormals::getDescriptorSize()
  {
    return blocks_x()*blocks_y()*getNBins();
  }
  
  int HistOfNormals::getNBins()
  {
    return theta_bins*phi_bins;
  }

  Size HistOfNormals::getWinSize()
  {
    return win_size;
  }

  HistOfNormals::HistOfNormals(
    Size win_size, 
    Size block_size, 
    Size block_stride, 
    Size cell_size) :
    win_size(win_size), cell_size(cell_size)
  {
  }

  Mat HistOfNormals::show(const string& title, std::vector< double > feat)
  {
    vector<double> hog_feat_theta(blocks_x()*blocks_y()*params::ORI_BINS/2,0);
    vector<double> hog_feat_phi(blocks_x()*blocks_y()*params::ORI_BINS/2,0);
    for(int yIter = 0; yIter < blocks_y(); yIter++)
      for(int xIter = 0; xIter < blocks_x(); xIter++)
	for(int theta_bin = 0; theta_bin < theta_bins; theta_bin++)
	  for(int phi_bin = 0; phi_bin < phi_bins; phi_bin++)
	  {
	    // project the normal into XY
	    float weight = feat[getIndex(xIter,yIter,0,phi_bin*theta_bins+theta_bin)];
	    
	    //compute contrast senstiive HOG bin...
	    int hog_bin_theta = theta_bin/(float)theta_bins*(float)params::ORI_BINS/2;
	    hog_feat_theta[xIter*blocks_y()*params::ORI_BINS/2 +
		     yIter*params::ORI_BINS/2 + 
		     hog_bin_theta] += weight;
	    int hog_bin_phi = phi_bin/(float)phi_bins*(float)params::ORI_BINS/2;
	    hog_feat_phi[xIter*blocks_y()*params::ORI_BINS/2 +
		     yIter*params::ORI_BINS/2 + 
		     hog_bin_phi] += weight;
	  }
    
    Mat dpyTheta = imageeq("",picture_HOG(*this,hog_feat_theta),false,false);
    Mat dpyPhi   = imageeq("",picture_HOG(*this,hog_feat_phi),false,false);
    Mat im = vertCat(dpyTheta,dpyPhi);
    image_safe(title,im);
    return im;
  }

  string HistOfNormals::toString() const
  {
    return "HistOfNormals";
  }
  
  ///
  /// SECTION: Area Feature Computer Implementation
  ///
  
  Size HOGComputer_Area::getBlockStride()
  {
    assert(block_stride == cell_size);
    return block_stride;
  }

  Size HOGComputer_Area::getCellSize()
  {
    return cell_size;
  }

  size_t HOGComputer_Area::getDescriptorSize()
  {
    return nbins*blocks_x()*blocks_y();
  }

  int HOGComputer_Area::getNBins()
  {
    return nbins;
  }

  Size HOGComputer_Area::getWinSize()
  {
    return win_size;
  }

  HOGComputer_Area::HOGComputer_Area(
    Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    win_size(win_size), 
    block_size(block_size), block_stride(block_stride),
    cell_size(cell_size), nbins(BIN_COUNT)
  {
  }

  HOGComputer_Area::HOGComputer_Area(
    HOGComputer_Area&other) :
    win_size(other.win_size),
    block_size(other.block_size), block_stride(other.block_stride),
    cell_size(other.cell_size), nbins(other.nbins)
  {
    //printf("HOGComputer_Area::HOGComputer_Area winSize = (%d, %d)\n",
      //win_size.height,win_size.width);
  }
  
  Mat HOGComputer_Area::show(const string& title, std::vector< double > feat)
  {
    auto upVGA  = [](Mat im)
    {
      if(im.size().area() > 640*480)
	return im;
      else return imVGA(im);      
    };
    
    vector<double> tpos, tneg;
    split_feat_pos_neg(feat,tpos,tneg);
    
    Mat vis = tileCat(vector<Mat>{
      upVGA(show_sample(title,tpos)),
      upVGA(show_max(title,tpos)),
      upVGA(show_sample(title,tneg)),
      upVGA(show_max(title,tneg))});
    return vis;
  }
  
  vector< FeatVis > HOGComputer_Area::show_planes(std::vector< double > feat)
  {
    FeatVis plane("HOGComputer_Area");
    plane.setPos(show("",feat));
    plane.setNeg(Mat(1,1,DataType<Vec3b>::type));
    assert(plane.getNeg().type() == DataType<Vec3b>::type);
    assert(plane.getPos().type() == DataType<Vec3b>::type);
    vector<FeatVis> result; 
    result.push_back(plane);
    return result;
  }  
  
  Mat HOGComputer_Area::show_sample(const string& title, std::vector< double > feat)
  {
    int SAMPLE_SIZE = 8;
    Mat weight(blocks_y()*SAMPLE_SIZE,blocks_x()*SAMPLE_SIZE,
	       DataType<float>::type,Scalar::all(0));
    Mat dists(blocks_y()*SAMPLE_SIZE,blocks_x()*SAMPLE_SIZE,
		DataType<Vec3b>::type,Scalar::all(0));
    Vec3b col_near(0,255,0); // green
    Vec3b col_far(255,0,0);  // blue
    Vec3b col_invalid(0,0,255); // red
    
    // first figure out the colors which represent distance...
    // and intenstieis which represent weight in template
    // per block in the feature
    for(int block_y = 0; block_y < blocks_y(); block_y++)
      for(int block_x = 0; block_x < blocks_x(); block_x++)
      {
	// build a histogram over the distances.
	vector<double> dist_hist;
	double sum = 0;
	for(int bin_iter = 0; bin_iter < nbins; bin_iter++)
	{
	  // template weight to prob.
	  double eps = numeric_limits<float>::min();
	  double f = feat[getIndex(block_x,block_y,0,bin_iter)];
	  //double p = (f<=0)?eps:-std::exp(-f)+1;
	  double p = (f<=0)?eps:f;
	  sum += p;
	  dist_hist.push_back(p);
	}
	// normalize
	for(int bin_iter = 0; bin_iter < nbins; bin_iter++)
	  dist_hist[bin_iter] /= sum;
	
	// per pixel in the visualization
	for(int yIter = 0; yIter < SAMPLE_SIZE; yIter++)
	  for(int xIter = 0; xIter < SAMPLE_SIZE; xIter++)
	  {
	    // sample bin and select color and conf
	    int dist_bin = rnd_multinom(dist_hist);
	    assert(0 <= dist_bin && dist_bin < nbins);
	    double conf = feat[getIndex(block_x,block_y,0,dist_bin)];
	    double w_far = ((double)dist_bin)/(double)(nbins+1);
	    Vec3b col = (1-w_far)*col_near + w_far*col_far;
	    
	    // write into the vis matrices
	    weight.at<float>(block_y*SAMPLE_SIZE+yIter,block_x*SAMPLE_SIZE+xIter) = conf;
	    dists.at<Vec3b> (block_y*SAMPLE_SIZE+yIter,block_x*SAMPLE_SIZE+xIter) = col;
	  }
      }
    
    // weight the dists by the intensitites
    Mat intensities = imageeq("",weight,false);
    for(int yIter = 0; yIter < dists.rows; yIter++)
      for(int xIter = 0; xIter < dists.cols; xIter++)
      {
	for(int cIter = 0; cIter < 3; cIter++)
	{
	  dists.at<Vec3b>(yIter,xIter)[cIter] = 
	    std::min<uchar>(
	      intensities.at<uchar>(yIter,xIter),
	      dists.at<Vec3b>(yIter,xIter)[cIter]
	    );
	}
      }
      
    image_safe(title,dists);
    return dists;
  }

  
  Mat deformable_depth::HOGComputer_Area::show_max(
    const string& title, std::vector< double > feats)
  {
    Mat vis(blocks_y(),blocks_x(),DataType<float>::type,Scalar::all(0));
    
    // loop over the values in the feature.
    for(int block_y = 0; block_y < blocks_y(); block_y++)
      for(int block_x = 0; block_x < blocks_x(); block_x++)
      {
	// find the best bin and value for  this block...
	int max_bin = 0;
	float max_val = -inf;
	for(int binIter = 0; binIter < nbins; binIter++)
	{
	  double&feat = feats[getIndex(block_x,block_y,0,binIter)];
	  if(feat > max_val)
	  {
	    max_val = feat;
	    max_bin = binIter;
	  }
	}
	
	printf("max_bin = %d w/ val = %f\n",max_bin,max_val);
	vis.at<float>(block_y,block_x) = max_bin*max_bin;
      }

    float sf = std::sqrt(320*240/vis.size().area());
    //imagesc("debug HoA",vis,false,true);
    resize(vis,vis,Size(),sf,sf,params::DEPTH_INTER_STRATEGY);
    return imageeq(title.c_str(),vis);
  }
    
  double HOGComputer_Area::DistInvarDepth(
    double depth, const Camera&camera, double&max_area_out)
  {
    // hyperparamters
    //const double side_scale = 1.96;
    //const double area_scale = side_scale*side_scale;
    static constexpr double area_scale = 1; //1.44*2.90;
    //static constexpr double bin_scale = 1.0f/2.0f;    
    static constexpr double bin_scale = 1.0f; 
    
    // find the max area
    static constexpr double max_area = std::pow(area_scale*MAX_AREA,bin_scale);
    //float max_area = area_scale*im.camera.pixAreaAtDepth(MAX_DEPTH); // bug
    max_area_out = max_area;
    
    // compute for this pixel
    float area  = area_scale*camera.pixAreaAtDepth(depth);
    area = std::pow(area,bin_scale);

    // this makes sense if the data is from the 
    // same camera as the MAX_DEPTH hyper-parameter.
    // If we don't require this, disable it but things
    // might not work ideally. 
    if(area > 1.5*max_area)
    {
      //imageeq("Problem with ZMap",im.Z); waitKey_safe(0);
      //printf("depth = %f\n",depth);
      log_once("MAX_DEPTH exceeded in HOGComputer_Area::compute",
		printfpp("area = %f > max_area = %f; depth = %f\n",area,max_area,depth));
      //assert(area <= 1.5*max_area);
    }    
    
    return clamp<double>(0,area,max_area_out);
  }
  
  Mat HOGComputer_Area::DistInvarientDepths(const ImRGBZ&im,double&max_area)
  {    
    Mat areas(im.rows(),im.cols(),DataType<float>::type);
    for(int yIter = 0; yIter < im.Z.rows; yIter++)
      for(int xIter = 0; xIter < im.Z.cols; xIter++)
      {
	// find the area using the camera geometry
	double depth = im.Z.at<float>(yIter,xIter);
	double area = DistInvarDepth(depth,im.camera,max_area);
	areas.at<float>(yIter,xIter) = area;
      }    
    
    return areas;
  }
  
  Mat DistInvarientDepth(const ImRGBZ& im)
  {
    double ignore;
    return HOGComputer_Area::DistInvarientDepths(im,ignore);
  }
  
  void HOGComputer_Area::compute(const ImRGBZ& im, std::vector< float >& feats)
  {    
    double max_area; 
    Mat areas = HOGComputer_Area::DistInvarientDepths(im,max_area);
    
    // compute the cell matrix
    vector<vector<vector<double> > > cells(blocks_y(),
      vector<vector<double> >(blocks_x(),
	vector<double>(nbins,0)));
    for(int yIter = 0; yIter < im.Z.rows; yIter++)
      for(int xIter = 0; xIter < im.Z.cols; xIter++)
      {	
	// get the area
	float area = areas.at<float>(yIter,xIter);
	
	// bin the area
	// nbins*area/max_area \in [0,nbins]
	int bin = clamp<int>(0,nbins*area/max_area,nbins-1);
	//printf("area = %f\n",area);
	//printf("bin = %d\n",bin);
	
	// update the cell histogram
	cells[yIter/cell_size.height][xIter/cell_size.width][bin] += 1.f/(double)cell_size.area();
      }
      
    // copy the cell matrix into the feature array
    feats = vector<float>(getDescriptorSize(),0);
    for(int block_y = 0; block_y < blocks_y(); block_y++)
      for(int block_x = 0; block_x < blocks_x(); block_x++)
	for(int binIter = 0; binIter < nbins; binIter++)
	{
	  int index = getIndex(block_x,block_y,0,binIter);
	  assert(index < feats.size());
	  double nf = 1.f; // nf = 1 for regular normalization.
	  float&feat = feats[index];
	  feat = std::pow(cells[block_y][block_x][binIter],nf);
	  assert(goodNumber(feat));
	}
	
    // DEBUG: Show the extracted feature
    //show("DEBUG: Area Feature",vec_f2d(feats));
    //cvWaitKey(0);
    //printf("-HOGComputer_Area::compute\n");
    //cout << areas << endl;
    assert(feats.size() == getDescriptorSize());
  }
  
  int HOGComputer_Area::cellsPerBlock()
  {
    return 1;
  }
  
  string HOGComputer_Area::toString() const
  {
    return "HistOfArea";
  }
  
  Size HOGComputer_Area::getBlockSize()
  {
    return getCellSize();
  }
  
  /// SECTION: DistInvarientDepth
  // this is a very simple feature which merely transforms an image
  // into distance invarient depths
  vector< FeatVis > DistInvarDepth::show_planes(vector< double > feat)
  {
    FeatVis plane("DistInvarDepth");
    plane.setPos(show("",feat));
    plane.setNeg(Mat(1,1,DataType<Vec3b>::type));
    assert(plane.getNeg().type() == DataType<Vec3b>::type);
    assert(plane.getPos().type() == DataType<Vec3b>::type);
    vector<FeatVis> result; 
    result.push_back(plane);
    return result;
  }
  
  int DistInvarDepth::cellsPerBlock()
  {
    return 1;
  }

  void DistInvarDepth::compute(const ImRGBZ& im, vector< float >& feats)
  {
    Mat did = DistInvarientDepth(im);
    feats.resize(did.size().area());
    int iter = 0;
    for(int rIter = 0; rIter < did.rows; rIter++)
      for(int cIter = 0; cIter < did.cols; cIter++)
	feats[iter++] = did.at<float>(rIter,cIter);
  }
  
  DistInvarDepth::DistInvarDepth
    (Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    win_size(win_size)
  {
    assert(cell_size.area() == 1 && block_size.area() == 1);
  }

  Size DistInvarDepth::getBlockSize()
  {
    return Size(1,1);
  }
  
  Size DistInvarDepth::getBlockStride()
  {
    return Size(1,1);
  }

  Size DistInvarDepth::getCellSize()
  {
    return Size(1,1);
  }
  
  size_t DistInvarDepth::getDescriptorSize()
  {
    return win_size.area();
  }

  int DistInvarDepth::getNBins()
  {
    return 1;
  }
  
  Size DistInvarDepth::getWinSize()
  {
    return win_size;
  }

  Mat DistInvarDepth::show(const string& title, vector< double > feat)
  {
    Mat vis(win_size.height,win_size.width,DataType<float>::type);
    int iter = 0;
    for(int yIter = 0; yIter < vis.rows; yIter++)
      for(int xIter = 0; xIter < vis.cols; xIter++)
	vis.at<float>(yIter,xIter) = feat[iter++];
    return imageeq(title.c_str(),vis,false,false);
  }
  
  ///
  /// SECTION: NullFeature_Computer
  ///
  
  NullFeatureComputer::NullFeatureComputer(
    Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    win_size(win_size), block_size(block_size), 
    block_stride(block_stride), cell_size(cell_size)
  {
  }

  vector< FeatVis > NullFeatureComputer::show_planes(vector< double > feat)
  {
    FeatVis plane("NullFeatureComputer");
    plane.setPos(Mat(4*blocks_y(),4*blocks_x(),DataType<Vec3b>::type,Scalar::all(0)));
    plane.setNeg(Mat(4*blocks_y(),4*blocks_x(),DataType<Vec3b>::type,Scalar::all(0)));
    assert(plane.getNeg().type() == DataType<Vec3b>::type);
    assert(plane.getPos().type() == DataType<Vec3b>::type);
    vector<FeatVis> result; 
    result.push_back(plane);
    return result;  
  }
  
  int NullFeatureComputer::cellsPerBlock()
  {
    return 1;
  }

  void NullFeatureComputer::compute(const ImRGBZ& im, vector< float >& feats)
  {
    // Rightly Null :-)
    feats.clear();
  }
  
  Size NullFeatureComputer::getBlockSize()
  {
    return cell_size;
  }
  
  Size NullFeatureComputer::getBlockStride()
  {
    return block_stride;
  }

  Size NullFeatureComputer::getCellSize()
  {
    return cell_size;
  }
  size_t NullFeatureComputer::getDescriptorSize()
  {
    return 0;
  }

  int NullFeatureComputer::getNBins()
  {
    return 0;
  }

  Size NullFeatureComputer::getWinSize()
  {
    return win_size;
  }

  Mat NullFeatureComputer::show(const string& title, vector< double > feat)
  {
    return Mat(4*blocks_y(),4*blocks_x(),DataType<Vec3b>::type,Scalar::all(0));
  }
  
  string NullFeatureComputer::toString() const
  {
      return "NullFeatComp";
  }
}
