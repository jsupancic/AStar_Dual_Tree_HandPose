/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "QP.hpp"
#include "util.hpp"
#include "params.hpp"
#include "vec.hpp"
#include "Log.hpp"
#include <boost/graph/graph_concepts.hpp>

namespace deformable_depth
{
  using namespace std;
    
  constexpr float BETA = 1;
  
  /// SECTION: LDA General
  size_t LDA::getFeatureLength() const
  {
    return getW().size();
  }
  
  /// SECTION: QP
  
  void QP::prime(int wLength)
  {
    w = vector<double>(wLength+1,0);
  }
  
  double clamp(double min, double value, double max)
  {
    if(value > max)
      return max;
    if(value < min)
      return min;
    return value;
  }  
    
  void QP::opt_one(Example&example,double&loss)
  {
    SparseVector&x = example.getX();
    
    // (a)
    double G  = x*w + (-example.getB());
    
    // (b)
    double PG = G;
    //if(alpha[idx] <= 0)
    //  PG = std::min<double>(G,0);
    //if(alpha[idx] >= C)
      // PG = std::max<double>(G,0);
    if((example.alpha == 0 && G >= 0) || (example.alpha == C && G <= 0))
      PG = 0;
    
    // hacky way to reduce the active step and boost performance in the average case
    if(example.alpha == 0 && G > 0 && !example.pos)
      example.active = false;
    
    if(G < 0)
      loss = loss - G;
    
    // (c)
    if(std::abs(PG) > 1e-12)
    {
      double a_bar = example.alpha;
      assert(example.D != 0);
      example.alpha = clamp<double>(0,example.alpha - G/example.D,C);
      bool wx_size_match = w.size() == x.size();
      if(!wx_size_match) 
      {
	cout << "w.size = " << w.size() << endl;
	cout << "x.size = " << x.size() << endl;
      }
      assert(wx_size_match);
      w = x*(example.alpha - a_bar) + w;
      l += example.getB()*example.alpha;
    }
  }
  
  void QP::activate_cache()
  {
    for(auto&example : cache)
      example.second.active = true;
  }
  
  void QP::trainSVM_fast(int ITER, double TOL)
  {
    ub = numeric_limits<double>::infinity();
    lb = -ub;
    
    // on the first call we need to init w to zeros
    if(cache.size() == 0)
      return;
    if(w.size() == 0)
      w = vector<double>(cache.begin()->second.getX().size(),0);
    
    // reactive all.
    activate_cache();
    
    // while alpha is not optimal...
    for(int opt_iter = 0; opt_iter < ITER; opt_iter++)
    {
      // randomly permute the indexes
      vector<Example*> permutation;
      for(auto iter = cache.begin(); iter != cache.end(); iter++)
	if(iter->second.active)
	  permutation.push_back(&iter->second);
      std::random_shuffle(permutation.begin(),permutation.end());
      double loss = 0;
      nsv = 0;
      l = 0;
      for(Example*ex : permutation)
      {
	// coordinate ascent!
	opt_one(*ex,loss);
	
	if(ex->alpha > 0)
	  nsv++;
      }
      
      // update the fast bounds
      lb = l - .5 * dot(w,w);
      assert(goodNumber(lb));
      ub = .5 * dot(w,w) + C*loss;
      assert(goodNumber(ub));      
	
      // sanity checks, (SLOW)
      // check that we are improving the objective each iteration?
      //printf("=====================================\n");
//       double newUB = obj_ub();
//       double newLB = obj_lb();
//       if(newLB < LB)
// 	printf("qp_opt: warning LB down from %f to %f\n",LB,newLB);
//       UB = newUB;
//       LB = newLB;
      string message = printfpp("qp_opt: iter = %d\t(lb,ub) = (%f,%f)\tnsv=%d",
			      opt_iter,lb,ub,nsv);
      cout << message << endl;
      log_file << message << endl;
      // lb/ub apply only to the active set
      double convrgance_ratio = 1-lb/ub;
      if(lb > 0 && convrgance_ratio < TOL)
	if(1 - lb/obj_ub() < TOL)
	  break;
	else
	  activate_cache();
    }
    printf("qp_opt: DONE\t(lb,ub) = (%f,%f)\tnsv=%d \n",lb,ub,nsv);
    
    assert(w.size() == cache.begin()->second.getX().size());
    trained = true;
  }  
  
  bool QP::isTrained()
  {
    return trained;
  }
  
  void QP::opt(int iter, double tol)
  {
    recompute_weights();
    trainSVM_fast(iter,tol);
    recompute_weights();
    update_capacity();
    prune();
    trainSVM_fast(iter,tol);
    recompute_weights();
  }
  
  void QP::update_capacity()
  {
    size_t sv_size = 0;
    for(auto&& example : cache)
      if(example.second.alpha > 0)
	sv_size += example.second.getX().footprint_bytes();
      
    max_size = std::max(MIN_SIZE,4*sv_size);
    log_file << 
      printfpp("qp::update_capacity: cache limited to %fMB",max_size/1e6) << 
      endl;
  }
  
  bool QP::is_overcapacity()
  {
    size_t size = footprint();
    return size > max_size; // 8 GB Max
  }
  
  void QP::recompute_weights()
  {
    int w_length = cache.begin()->second.getX().size();
    log_file << printfpp("cache.begin()->second.getX().size() = %d",w_length) << endl;
    vector<double>w_new(w_length,0);
    
    for(auto iter = cache.begin(); iter != cache.end(); iter++)
      if(iter->second.active)
	w_new = iter->second.getX() * iter->second.alpha + w_new;
    
    w = w_new;
  }
  
  void QP::prune()
  {
    if(cache.size() == 0 || !is_overcapacity())
      return;
    
    // allocate new storage  
    double l_new = 0;
    
    map<string/*feat*/,Example> XTrain_new;
    m_footprint = 0;
    
    // copy support vectors into new storage
    for(auto iter = cache.begin(); iter != cache.end(); iter++)
    {
      // copy the correct values into the new storage
      float resp = iter->second.getX() * w;
      // Support vector
      if(resp <= 1 || iter->second.pos || iter->second.alpha > 0)
      {
	XTrain_new.insert(pair<string,Example>(iter->first,iter->second));
	m_footprint += iter->second.getX().footprint_bytes();
	l_new += iter->second.getB()*iter->second.alpha;
      }
    }
    
    printf("===================================================================\n");
    printf("============================= WARNING =============================\n");
    printf("Pruned %d of %d examples\n",(int)(cache.size()-XTrain_new.size()),(int)cache.size());
    printf("===================================================================\n");
    printf("===================================================================\n");
    
    // overwrite old storage with new
    l = l_new;
    cache = XTrain_new;
    recompute_weights();
  }

  size_t QP::num_SVs() const
  {
    size_t sv_count = 0;
    
    for(auto iter = cache.begin(); iter != cache.end(); iter++)
      if(iter->second.alpha > 0)
	sv_count++;
    
    return sv_count;
  }
  
  size_t QP::cache_size() const
  {
    return cache.size();
  }
  
  size_t QP::footprint() const
  {
    return m_footprint;
  }
  
  // from the primal
  // (Primal) min_{w,e}  .5*||w||^2 + C*sum_i e_i
  //              s.t.   w*x_i >= b_i - e_i 
  //			e_i >= 0
  double QP::obj_ub()
  {
    double cost = 0.5 * dot(w,w);
    for(auto iter = cache.begin(); iter != cache.end(); iter++)
    {
      double resp = iter->second.getX()*w;
      double hinge = std::max<double>(0,1-resp);
      cost += C*hinge;
    }
    return cost;
  }
  
  // (Dual) max_{a} -.5*sum_ij a_i(x_ix_j)a_j + sum_i b_i*a_i s.t.  0 <=
  //           a_i <= C
  
  double QP::apx_lb()
  {
    return lb;
  }

  double QP::apx_ub()
  {
    return ub;
  }
  
  float QP::predict(std::vector< float > x) const
  {    
    x.push_back(BETA);
    
    if(x.size() != w.size())
    {
      printf("x.size() = %d\n",(int)x.size());
      printf("w.size() = %d\n",(int)w.size());
      assert(x.size() == w.size());
    }
    
    return dot(x,w);
  }
  
  float QP::predict(Mat_< float > x) const
  {
    //printf("predict x.cols = %d\n",x.cols);
    if(w.size() != x.cols + 1)
    {
      printf("QP:predict |w| = %d |x| = %d\n",(int)w.size(),(int)x.cols);
      assert(w.size() == x.cols + 1);
    }
    // convert Mat to vector and add constant feature
    vector<double> xx;
    for(int iter =0; iter < x.cols; iter++)
      xx.push_back(x.at<float>(0,iter));
    xx.push_back(BETA);
    return dot(xx,w);
  }

  vector<double> QP::getW() const
  {
    if(w.size() < 2)
      cout << "w.size < 2; cache.size = " << cache.size() << endl;
    assert(w.size() >= 2);
    return vector<double>(w.begin(),w.end()-1);
  }
  
  double QP::getB() const
  {
    return w[w.size()-1];
  }
  
  vector<float> QP::getWf() const
  {
    vector<float> wf(w.size(),0);
    for(int iter = 0; iter < w.size(); iter++)
      wf[iter] = w[iter];
    return vector<float>(wf.begin(),wf.end()-1);
  }
  
  bool QP::write(SparseVector& xx, float y, string id)
  {
    // special debug assert for part models
    //assert(xx.sparsity() > 0.5);
    
    // add constant features
    xx.push_back((double)BETA);
    
    // we multiply negatives by -1 to simplify implmenetation
    bool is_pos = y >= 0;
    if(!is_pos && is_overcapacity())
      return false;
    if(!is_pos)
      xx = xx * (double)-1;
    
    double resp = trained?xx*w:0;
    // only add margin violations
    if(!trained || is_pos || resp < 1)
    {
      Example newExample(xx,id,0,xx.dot_self(),1,is_pos);
      if(cache.find(newExample.key()) == cache.end())
      {
	cache.insert(pair<string,Example>(newExample.key(),newExample));
	m_footprint += xx.footprint_bytes();
	return true;
      }
      else
      {
	// we already have the key in the cache
	return cache.at(newExample.key()).update(id,newExample,w);
      }
    }    
    else
      return false;
  }
  
  bool QP::write(std::vector< float > xx, float y, string id) 
  {
    return write(vec_f2d(xx),y,id);
  }
  
  bool QP::write(std::vector< double > xx, float y, string id)
  {
    SparseVector x(xx);
    return write(x,y,id);
  }

  QP::QP(float C_) : 
    trained(false), C(C_), lb(0), ub(0), l(0), 
    m_footprint(0), max_size(MIN_SIZE)
  {
    log_file << "QP Allocated C = " << C << endl;
  }	
  
  void QP::prime(std::vector< double > w, double b)
  {
    this->w = w;
    this->w.push_back(b);
  }
  
  /// SECTION: QPExample
  SVMExample::SVMExample(SparseVector&x_, string id, double alpha_, double D_, double B_, bool pos)
   : x(x_), alpha(alpha_), D(D_), B(B_), pos(pos), active(true)
  {
  }
  
  double SVMExample::y() const
  {
    if(pos)
      return +1.0;
    else
      return -1.0;
  }
  
  /// SECTION: MILSVMExample, old style
  MILSVMExample::MILSVMExample(
    SparseVector& x_, string id, double alpha_, double D_, double B_, bool pos) : 
    SVMExample(x_,id,alpha_,D_,B_,pos), weight(1) 
  {
    ids.insert(id);    
  }
  
  bool MILSVMExample::update(string id)
  {
    if(!WEIGHT_BY_COUNT)
      return false; 
    
    if(ids.find(id) == ids.end())
    {
      ids.insert(id);
      weight = ids.size();
      return true;
    }
    else 
      return false;
  }
  
  double MILSVMExample::getB()
  {
    return weight*B;
  }

  SparseVector&MILSVMExample::getX()
  {
    return x;
  }
  
  string MILSVMExample::key() const
  {
    return x.hash();
  }
  
  // SECTION: LatentSVMExample, wherein we implement the latent update stage
  double LatentSVMExample::getB()
  {
    return B;
  }
  
  SparseVector& LatentSVMExample::getX()
  {
    return x;
  }
  
  string LatentSVMExample::key() const
  {
    return id;
  }
  
  LatentSVMExample::LatentSVMExample
    (SparseVector& x_, string id, double alpha_, double D_, double B_, bool pos): 
    SVMExample(x_, id, alpha_, D_, B_, pos), id(id)
  {
  }
  
  bool LatentSVMExample::update(string id,LatentSVMExample&newExample,const vector<double>&w)
  {
    double this_resp = y()*(x*w);
    double new_resp = newExample.y()*(newExample.x*w);
    
    // always take the highest scoring BB
    bool accept_pos = (pos  && this_resp < new_resp);
    bool accept_neg = (!pos && this_resp < new_resp);
    bool accept = accept_pos || accept_neg;
    
    if(accept)
    {
      *this = newExample;
      if(pos)
	log_file << "Accepted Latent Positive Update for " << id << endl;
      return true; // accept update
    }
    else
    {
      return false; // reject update
    }
  }
}
