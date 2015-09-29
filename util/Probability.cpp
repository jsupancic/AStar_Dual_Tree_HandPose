/**
 * Copyright 2013: James Steven Supancic III
 **/

#include <random>

#include "Probability.hpp"
#include <boost/math/special_functions/round.hpp>
#include "util_real.hpp"
#include "params.hpp"
#include "util.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  int rnd_multinom(vector< double >& thetas)
  {
    // setup the RNG
    static std::mt19937 sample_seq;
    std::uniform_real_distribution<double> unif_01(0,1);
    double r = unif_01(sample_seq); // get the random sample
    
    // accumulate to prepare to sample
    vector<double> cum_dist = {0};
    double cum = 0;
    for(int iter = 0; iter < thetas.size(); iter++)
    {
      auto theta = thetas.at(iter);
      assert(0 <= theta and theta <= 1);
      cum += theta;
      cum_dist.push_back(cum);
    }
    assert(cum <= 2);
    // avoid numeric issues
    cum_dist.front() = -numeric_limits<double>::infinity(); 
    cum_dist.back() = numeric_limits<double>::infinity(); 
    
    // choose the sample
    for(int sample_val = 0; sample_val < thetas.size(); sample_val++)
    {
      double lower = cum_dist[sample_val];
      double upper = cum_dist[sample_val+1];
      if(lower <= r && r <= upper)
	return sample_val;
    }
    
    log_file << "warning: rnd_multinom error: " << toString(thetas) << endl;
    return rand()%thetas.size();
  }

  Point rnd_multinom(const Mat&prob_image)
  {
    assert(prob_image.type() == DataType<double>::type);
    vector<Point> indices;
    vector<double> probs;
    for(int yIter = 0; yIter < prob_image.rows; yIter++)
      for(int xIter = 0; xIter < prob_image.cols; xIter++)
      {
	double p = prob_image.at<double>(yIter,xIter);
	if(not goodNumber(p))
	  p = 0;
	probs.push_back(p);
	indices.push_back(Point(xIter,yIter));
      }
    double s = sum(probs);
    if(s > 0)
    {
      probs = probs / s;
      return indices.at(rnd_multinom(probs));
    }
    else
      return indices.at(thread_rand()%indices.size());
  }
  
  Gaussian::Gaussian(double mu, double sigma) : mu(mu), sigma(sigma)
  {
  }

  Gaussian::Gaussian(const Gaussian& g) : mu(g.mu), sigma(g.sigma)
  {
  }

  double Gaussian::pdf(double x) const
  {
    return 1.0/(sigma*std::sqrt(2*params::PI)) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
  }
  
  ostream& operator<<(ostream& os, const Gaussian& g)
  {
    os << "Gaussian mu = " << g.mu << " sigma = " << g.sigma;
    return os;
  }
  
  double mv_gaussian_pdf(const Mat&x,const cv::Mat&mu, const Mat&sigma, const cv::Mat&sigmaInv)
  {
    assert(mu.type() == DataType<double>::type);
    assert(sigmaInv.type() == DataType<double>::type);
    
    // compute the kernel
    Mat diff = x - mu;
    Mat mat_p = -.5 * diff * sigmaInv * diff.t();
    assert(mat_p.size() == Size(1,1));
    double p = std::exp(mat_p.at<double>(0,0));
    return p;
    
    // compute the normalization constnat
    //double k = mu.cols;
    //double nf = std::pow((2*params::PI),-k/2) * std::pow(cv::determinant(sigma),-.5);
    //p = nf * p;
    
    //cout << "p = " << p << endl;
    //return clamp<double>(1e-10, p,1-1e10);
  }
  
  double mv_gaussian_support(double sigma)
  {
    // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 .
    // (sigma - .8)/0.3 = ((ksize-1)*0.5 - 1) 
    // 2*(sigma - .8)/0.3 + 2 = (ksize-1)
    // 2*(sigma - .8)/0.3 + 3 = ksize
    return 2*(std::abs(sigma) - .8)/0.3 + 3 ;
  }
  
  Mat mv_gaussian_kernel(const Mat&sigma, const Mat&sigmaInv)
  {
    // compute the Parzen?Rosenblatt window
    int xsize = boost::math::round(mv_gaussian_support(std::sqrt(sigma.at<double>(0,0))));
    int ysize = boost::math::round(mv_gaussian_support(std::sqrt(sigma.at<double>(1,1))));
    xsize = std::max(xsize,5);
    ysize = std::max(ysize,5);
    
    // pre-compute the PDF kernel
    cv::Mat center(1,2,DataType<double>::type,Scalar::all(0)); 
    center.at<double>(0) = xsize / 2;
    center.at<double>(1) = ysize / 2;
    Mat kernel(ysize, xsize,DataType<double>::type,Scalar::all(0));
    for(int yIter = 0; yIter < ysize; yIter++)
      for(int xIter = 0; xIter < xsize; xIter++)
      {
	cv::Mat x(1,2,DataType<double>::type,Scalar::all(0));
	x.at<double>(0) = xIter;
	x.at<double>(1) = yIter;	
	kernel.at<double>(yIter,xIter) = (mv_gaussian_pdf(x,center, sigma,sigmaInv));
      } 
      
    return kernel;
  }
  
  double EmpiricalDistribution::cdf(double x) const
  {
    auto iter = std::upper_bound(data.begin(), data.end(), x);
    if(iter == data.end())
      return 1;
    
    auto start = data.begin();
    double idx = &*iter - &*start;
    double q = idx/data.size();
    assert(0 <= q && q <= 1);
    return q;
  }
  
  double EmpiricalDistribution::quant(double p)
  {
    int index = clamp<int>(0,data.size()*p,data.size()-1);
    return data[index];
  }

  ///
  /// SECTION: Entropy Functions
  ///
  double shannon_entropy(double p)
  {
    return shannon_entropy(vector<double>{p,1-p});
  }
  
  double shannon_entropy(vector<double> ps)
  {
    double h = 0;
    
    for(double p : ps)
      if(p > 0)
	h -= p*std::log2(p);
    
    return h;
  }

  double entropy_gaussian(ExampleFn exampleFn,int nExamples,
			  Mat&mean,Mat&cov,int feat_len)
  {
    if(nExamples == 0)
      return 0;
    
    Mat samples(nExamples,feat_len,DataType<double>::type,Scalar::all(0));
    atomic<long> count(0);
    //TaskBlock entropy_gaussian("entropy_gaussian");
    for(int iter = 0; iter < nExamples; ++iter)
    {
      //entropy_gaussian.add_callee([&,iter]()
      {
	Mat ex_feat = exampleFn(iter);
	require_equal<int>(ex_feat.rows,1);
	require_equal<int>(ex_feat.cols,(long)feat_len);
	assert(ex_feat.type() == DataType<double>::type);
	// copy the feature into the matrix
	for(int featIter = 0; featIter < feat_len; ++featIter)
	  samples.at<double>(iter,featIter) = ex_feat.at<double>(0,featIter);
	
	//samples.at<double>(iter,2) = off_z;
	long cur_count = count++;
	if(cur_count % std::max<int>(100,(nExamples/1000)) == 0)
	  log_file << printfpp("entropy_gaussian %d of %d",
			       (int)cur_count,(int)nExamples) << endl;
      }//);
    }
    //entropy_gaussian.execute();
    
    cv::calcCovarMatrix(samples,cov,mean,CV_COVAR_SCALE|CV_COVAR_ROWS|CV_COVAR_NORMAL,CV_64F);
    cov += cv::Mat::eye(cov.rows,cov.cols,cov.type());
    double det = cv::determinant(cov);
    return std::log(det);
  }
}
