/**
 * Copyright 2013: James Steven Supancic III
 * 
 * Implementation of algorithm by Hsuan-Tien Lin, Chih-Jen Lin and Ruby C. Weng (2007)
 **/

#include "Platt.hpp"
#include <cmath>
#include <algorithm>
#include <memory>
#include "Log.hpp"
#include <assert.h>
#include "MetaData.hpp"
#include "Detector.hpp"

namespace deformable_depth
{
  using namespace std;
  
  LogisticPlatt::LogisticPlatt(std::vector< double >& deci, std::vector< bool >& label)
  {
    init(deci,label);
    opt(deci,label);
    printf("A = %f B = %f\n",(float)A,(float)B);
    log_file << "A = " << A << "B = " << B << endl; log_file.flush();
    delete[] t;
  }

  void LogisticPlatt::init(std::vector< double >& deci, std::vector< bool >& label)
  {
    prior1 = accumulate(label.begin(),label.end(),0);
    prior0 = label.size() - prior1;
    
    // parameter settings
    maxiter = 100; // max number of iterations
    minstep = 1e-10; // minium step size taken in line search
    sigma = 1e-12; // set to any value > 0
    // construct initial values: 
    // (1) target support in array t
    // (2) initial function value in fval
    hiTarget = (prior1+1.0)/(prior1+2.0);
    loTarget = 1/(prior0+2.0);
    len = deci.size();
    t = new double[len];
    for(int iter = 0; iter < len; iter++) 
      if(label[iter] > 0)
	t[iter] = hiTarget;
      else
	t[iter] = loTarget;
    // compute initial objective value
    A = 0.0;
    B = std::log((prior0+1.0)/(prior1+1.0));
    fval0 = fval = fObj(deci,A,B);
  }
  
  double LogisticPlatt::fObj(std::vector< double >& deci, double A, double B) const
  {
    printf("LogisticPlatt::fObj len = %d\n",(int)len);
    double fobj = 0.0;
    for(int iter = 0, last_iter = -1; iter < len; last_iter = iter, iter++)
    {
      assert(iter > last_iter);
      double fApB = deci[iter]*A+B;
      if(fApB >= 0)
	fobj +=  t[iter]*fApB+std::log(1+exp(-fApB));
      else
	fobj += (t[iter]-1)*fApB+std::log(1+exp(fApB));
    }
    return fobj;
  }

  void LogisticPlatt::opt(std::vector< double >& deci, std::vector< bool >& label)
  {
    int it = 0;
    for(it = 0; it < maxiter; it++)
    {
      printf("platt opt iter %d\n",it);
      // Update Gradient and Hessian (using H' = H + sigma I)
      double h11 = sigma, h22 = sigma;
      double h21 = 0, g1 = 0, g2 = 0;
      for(int iter = 0; iter < len; iter++)
      {
	double fApB = deci[iter]*A+B;
	double p,q;
	if(fApB >= 0)
	{
	  p = exp(-fApB)/(1+exp(-fApB));
	  q = 1.0/(1.0+exp(-fApB));
	}
	else
	{
	  p = 1.0/(1.0+exp(fApB));
	  q = exp(fApB)/(1.0+exp(fApB));
	}
	double d2 = p*q;
	h11 += deci[iter]*deci[iter]*d2;
	h22 += d2;
	h21 += deci[iter]*d2;
	double d1 = t[iter]-p;
	g1 += deci[iter]*d1;
	g2 += d1;
      }
      
      // check stopping critieria
      if(abs(g1) < 1e-5 && abs(g2) < 1e-5)
	break;
      
      // compute modified newton directions
      double det = h11*h22 - h21*h21;
      double dA = -(h22*g1 - h21*g2)/det;
      double dB = -(-h21*g1 + h11*g2)/det;
      double gd = g1*dA + g2*dB;
      double stepsize = 1;
      // line search
      while(stepsize >= minstep) 
      {
	double newA = A + stepsize*dA;
	double newB = B + stepsize*dB;
	double newF = fObj(deci,newA,newB);
	if(newF < fval + .0001*stepsize*gd)
	{
	  A = newA;
	  B = newB;
	  fval = newF;
	  break; // sufficent decrease satisfied
	}
	else
	  stepsize /= 2.0;
      }
      if(stepsize < minstep)
      {
	printf("Line Search Failed!\n");
	break;
      }
    }
    printf("LogisticPlatt: covergence took %d iterations\n",it);
    printf("fvall %f to %f\n",(float)fval0,(float)fval);
  }
  
  double LogisticPlatt::prob(double deci)
  {
    return 1/(1+std::exp(A*deci+B));
  }
  
  void platt_test()
  {
    vector<double> deci  = {-1, 0, 1, -.5};
    vector<bool>   label = {0,  0, 1, 0};
    LogisticPlatt platt(deci,label);
    for(float val = -1; val <= 1; val += .5)
      cout << "platt(" << val << ") = " << platt.prob(val) << endl;
  }
  
  static string cache_filename(
    string prefix,
    std::vector< shared_ptr<MetaData> > pos_set, 
    std::vector< shared_ptr<MetaData> > neg_set,
    Model::TrainParams params)
  {
    vector<string> pfilenames = filenames(pos_set);
    std::sort(pfilenames.begin(),pfilenames.end());
    vector<string> nfilenames = filenames(neg_set);
    std::sort(nfilenames.begin(),nfilenames.end());
    // first, try to load the platt results from the cache_file
    string cache_file = printfpp("cache/%s=%sNeg=%sKey=%s.yml",
				 prefix.c_str(),
				 hash(pfilenames).c_str(),
				 hash(nfilenames).c_str(),
				 params.subset_cache_key.c_str());    
    return cache_file;
  }  
  
  tuple<vector<double>,vector<bool> >
    collect_detections(
			  ExposedTrainingSet& model, 
	                  std::vector< shared_ptr<MetaData> > pos_set, 
			  std::vector< shared_ptr<MetaData> > neg_set,
			  Model::TrainParams params)
  {
    // try to get from cache
    string cache_file = cache_filename("collected_dets",pos_set,neg_set,params);
    FileStorage cache(cache_file,FileStorage::READ);
    if(cache.isOpened())
    {
      vector<double> deci;
      vector<bool> label;
      cache["deci"] >> deci;
      vector<uchar> uchar_labels;
      cache["label"] >> uchar_labels;
      label = vector<bool>(uchar_labels.begin(),uchar_labels.end());
      return std::make_tuple(deci,label);
    }
      
    // collection criteria
    float inf = numeric_limits<float>::infinity();
    DetectionFilter filter(-inf,numeric_limits<int>::max());
    filter.supress_feature = true;
    filter.sort = false;
    
    // collect the responces
    vector<double> deci;
    vector<bool> label;
    // collect positives
    static mutex m;
    TaskBlock platt_collect("platt_collect");
    for(shared_ptr<MetaData> &metadata : pos_set)
    {
      params.positive_iterations = 1;
      params.negative_iterations = 0;
      platt_collect.add_callee([params,metadata,&model,&deci,&label,&filter]()
      {
	model.collect_training_examples(
	  *metadata, params, 
	  [&deci,&label](DetectorResult pos){
	    lock_guard<mutex> l(m);
	    deci.push_back(pos->resp); label.push_back(true);},
	  [&deci,&label](DetectorResult neg){assert(false);}
	  ,filter
	);	
      });
    }
    // collect negatives
    for(shared_ptr<MetaData> &metadata : neg_set)
    {
      params.positive_iterations = 0;
      params.negative_iterations = 1;            
      platt_collect.add_callee([params,metadata,&model,&deci,&label,&filter]()
      {      
	vector<double> frame_deci;
	vector<bool> frame_label;
	model.collect_training_examples(
	  *metadata, params, 
	  [&frame_deci,&frame_label](DetectorResult pos){assert(false);},
	  [&frame_deci,&frame_label](DetectorResult neg)
	    {
	      frame_deci.push_back(neg->resp); frame_label.push_back(false);}
	  ,filter);
	
	lock_guard<mutex> l(m);
	deci.insert(deci.end(),frame_deci.begin(),frame_deci.end());
	label.insert(label.end(),frame_label.begin(),frame_label.end());
      });
    }
    platt_collect.execute();    
    
    cache.open(cache_file,FileStorage::WRITE);
    cache << "deci" << deci;
    vector<double> uchar_labels(label.begin(),label.end());
    cache << "label" << uchar_labels;
    return std::make_tuple(deci,label);
  }
    
  shared_ptr<LogisticPlatt> train_platt(ExposedTrainingSet& model, 
					std::vector< shared_ptr<MetaData> > pos_set, 
					std::vector< shared_ptr<MetaData> > neg_set,
					Model::TrainParams params)
  {
    shared_ptr<LogisticPlatt> regresser;    
    string cache_file = cache_filename("platPos",pos_set,neg_set,params);
    FileStorage cache;
    cache.open(cache_file,FileStorage::READ);
    if(cache.isOpened())
    {
      regresser.reset(new LogisticPlatt());
      cache["platt"] >> (LogisticPlatt&)*regresser;
      cache.release();
      log_file << "Platt from cache" << endl;
      return regresser;
    }
    log_file << "Platt not from cache" << endl;
    
    // collect examples
    vector<double> deci;
    vector<bool> label;
    std::tie(deci,label) = collect_detections(model, pos_set, neg_set,params);
      
    // compute the optimal regression parameters
    regresser.reset(new LogisticPlatt(deci,label));
    
    // store to cache
    cache.open(cache_file,FileStorage::WRITE);
    cache << "platt" << *regresser;
    vector<double> pos_resps, neg_resps;
    for(int iter = 0; iter < label.size(); ++iter)
      if(label[iter])
	pos_resps.push_back(deci[iter]);
      else
	neg_resps.push_back(deci[iter]);
    cache << "pos_resps" << pos_resps;
    cache << "neg_resps" << neg_resps;
    cache.release();
    
    return regresser;
  }
  
  /// SECTION: Serialization
  
  void read(const FileNode& node, LogisticPlatt& platt, 
			      const LogisticPlatt& defaultValue)
  {
    platt.A = node["A"];
    platt.B = node["B"];
  }

  void read(FileNode node, shared_ptr< LogisticPlatt >& platt, shared_ptr< LogisticPlatt > )
  {
    platt.reset(new LogisticPlatt());
    read(node,*platt);
  }
  
  void write(FileStorage& fs, const string& , const LogisticPlatt& platt)
  {
    fs << "{";
    fs << "A" << platt.A;
    fs << "B" << platt.B;
    fs << "}";
  }
  
  void write(FileStorage& fs, string& str, const shared_ptr< LogisticPlatt >& platt)
  {
    write(fs,str,*platt);
  }
}
