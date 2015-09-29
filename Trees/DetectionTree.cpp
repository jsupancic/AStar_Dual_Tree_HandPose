/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "DetectionTree.hpp"
#include "util_mat.hpp"
#include "util_real.hpp"
#include "omp.h"
#include "vec.hpp"
#include "hashMat.hpp"

namespace deformable_depth
{
  /** SECTION
   * Serialization Support
  **/ 
  void write(FileStorage& fs, const string& , const DetectionTree& tree)
  {
    fs << "{"
    << "n_pos" << tree.n_pos
    << "n_neg" << tree.n_neg
    << "n_agg" << tree.n_agg
    << "node_type" << tree.node_type;
    
    if(tree.node_type == DetectionTree::INTERNAL)
    {
      fs << "decision" << tree.decision;
      fs << "branch_true" << *(tree.branch_true);
      fs << "branch_false" << *(tree.branch_false);
    }
    
    fs << "}";
  }

  template<typename Store>
  void read(const Store& node, DetectionTree& tree, const DetectionTree& default_value)
  {
    tree.n_pos = node["n_pos"];
    tree.n_neg = node["n_neg"];
    tree.n_agg = node["n_agg"];
    int raw_type = node["node_type"];
    tree.node_type = DetectionTree::NodeType(raw_type);
    
    if(tree.node_type == DetectionTree::INTERNAL)
    {
      // load the subtrees after allocating space
      tree.branch_true.reset(new DetectionTree());
      tree.branch_false.reset(new DetectionTree());
      node["branch_true"] >> *tree.branch_true;
      node["branch_false"] >> *tree.branch_false;
      
      // now, get the damn decision boundary...
      node["decision"] >> tree.decision;
    }
  }
  template void read(const FileStorage& node, DetectionTree& tree, const DetectionTree& default_value);
  template void read(const FileNode& node, DetectionTree& tree, const DetectionTree& default_value);  
  
  void write(FileStorage& fs, const string& , const shared_ptr< Decision >& writeMe)
  {
    fs << "{";
    writeMe->write(fs);
    fs << "}";
  }
  
  template<typename Store>
  void read(const Store& node, shared_ptr< Decision >& decision, 
	    const shared_ptr< Decision >& default_value = shared_ptr<Decision>())
  { 
    string decision_type; node["decision_type"] >> decision_type;
    double mu, sigma;
    node["mu"] >> mu;
    node["sigma"] >> sigma;
    
    if(decision_type == "Range")
    {
      int var;
      float low, high;
      node["var"] >> var;
      node["low"] >> low;
      node["high"] >> high;
      decision.reset(new DecisionRange(mu,sigma,var,low,high));
    }
    else if(decision_type == "Threshold")
    {
      int best_var;
      float best_value;
      node["best_var"] >> best_var;
      node["best_value"] >> best_value;
      decision.reset(new DecisionThreshold(mu,sigma,best_var,best_value));
    }
    else 
      throw std::exception();  
  }
  
  /**
   * Section: Information Theory
   **/
  
  double entropyH(double count1, double count2)
  {
    if(count1 == 0 || count2 == 0)
      return 0.0;
    
    double sum = (count1 + count2);
    double p1 = count1/sum;
    double p2 = count2/sum;
    
    return -p1*log2(p1) - p2*log2(p2);
  }
  
  double info_gain(double neg_below, double pos_below, double neg_above, double pos_above)
  {
    double total_below = neg_below+pos_below;
    double total_above = neg_above+pos_above;
    double total = total_above+total_below;
    
    return 
      -total_below/total*entropyH(neg_below,pos_below) + 
      -total_above/total*entropyH(pos_above,neg_above);
  }
  
  struct VarInfo
  {
    vector<int>&var_uses;
    shared_ptr<vector<float> >&sorted;
    vector<int>&idxs;
    vector<double>&cum_pos;
    vector<double>&cum_neg;
  };
  
  void DetectionTree::eval_thresholds(
    int var_idx,
    dptr&best_decsion,
    double&best_info_gain,
    VarInfo&info) const
  {
    // find the best index into the sorted Ys
    //#pragma omp parallel for if(info.idxs.size() > 100000)
    for(int iter = 0; iter < info.idxs.size(); iter++)
    {
      //printf("(Var,Value) = (%d,%f)\n",var_idx,(float)sorted[iter]);
      double igain = info_gain(info.cum_neg[iter],info.cum_pos[iter],
			       n_neg-info.cum_neg[iter],n_pos-info.cum_pos[iter]);
      #pragma omp critical
      if(igain > best_info_gain)
      {
	best_info_gain = igain;
	best_decsion = dptr(new DecisionThreshold(info.sorted,var_idx,(*info.sorted)[iter]));
      }
    }    
  }
  
  void DetectionTree::eval_ranges(
    int var_idx, 
    dptr& best_decsion, 
    double& best_info_gain, 
    VarInfo& info) const
  {
    if(info.var_uses[var_idx] < 2)
      return;
    int step = std::sqrt(info.idxs.size());
    
    //#pragma omp parallel for if(info.idxs.size() > 100000)
    for(int iter = 0; iter < info.idxs.size(); iter+=step)
      for(int jter = iter + 1; jter < info.idxs.size(); jter+=step)
      {
	double pos_in  = info.cum_pos[jter] - info.cum_pos[iter];
	double pos_out = n_pos - pos_in;
	double neg_in  = info.cum_neg[jter] - info.cum_neg[iter];
	double neg_out = n_neg - neg_in;
	double igain = info_gain(neg_in,pos_in,neg_out,pos_out);
	#pragma omp critical
	if(igain > best_info_gain)
	{
	  best_info_gain = igain;
	  double low = (*info.sorted)[iter], high = (*info.sorted)[jter];
	  best_decsion = dptr(new DecisionRange(info.sorted,var_idx,low,high));
	}
      }
  }
  
  dptr DetectionTree::choose_split(cv::Mat& X, cv::Mat& Y, std::vector< int >&var_uses) const
  {
    double best_info_gain = -numeric_limits<double>::infinity();
    dptr best_decsion;
   
    #pragma omp parallel for if(X.rows > 100000)
    for(int var_idx = 0; var_idx < X.cols; var_idx++)
    {
      if(var_uses[var_idx] <= 0)
	continue;
      
      printf("var_idx = %d\n",var_idx);
      // sort the training data according to the given variable.
      shared_ptr<vector<float> > sorted(new vector<float>);
      vector<int> idxs;
      sort(X.col(var_idx),*sorted,idxs);
      
      // find the best value to split at...
      // compute cumulative sums over the sorted Y
      vector<double> cum_pos(Y.rows,0), cum_neg(Y.rows,0);
      //#pragma omp parallel for if(X.rows > 100000)
      for(int iter = 0; iter < idxs.size(); iter++)
      {
	// select into positive or negative
	float label = Y.at<float>(idxs[iter]);
	if(label > 0)
	  cum_pos[iter]++;
	else
	  cum_neg[iter]++;
	
	// accumulate
	if(iter > 0)
	{
	  cum_pos[iter] += cum_pos[iter-1];
	  cum_neg[iter] += cum_neg[iter-1];
	}
      }
      double n_pos = cum_pos.back(), n_neg = cum_neg.back();
      
      VarInfo varInfo {var_uses,sorted,idxs,cum_pos,cum_neg};
      eval_thresholds(var_idx,best_decsion,best_info_gain,varInfo);
      eval_ranges(var_idx,best_decsion,best_info_gain,varInfo);
    }
    
    best_decsion->computeMoments();
    
    #pragma omp critical
    {cout << "best decision: "; best_decsion->println(); cout << endl;}
    best_decsion->update_uses(var_uses);
    return best_decsion;
  }
  
  void DetectionTree::split_data(Mat&XPrime, Mat&YPrime, Mat X, Mat Y, int side) const
  {
    // select rows for this branch
    for(int rowIter = 0; rowIter < Y.rows; rowIter++)
    {
      if(decision->decide(X.row(rowIter)) && side == 0)
      {
	// take
	XPrime.push_back<float>(X.row(rowIter));
	YPrime.push_back<float>(Y.row(rowIter));
	assert(X.rows >= 1);
      }
      if(!decision->decide(X.row(rowIter)) && side == 1)
      {
	// take
	XPrime.push_back<float>(X.row(rowIter));
	YPrime.push_back<float>(Y.row(rowIter));	  
	assert(X.rows >= 1);
      }
    }    
  }
  
  void DetectionTree::train(Mat& X, Mat& Y, std::vector< int > var_uses, int depth)
  {
    // default value for var_uses
    if(var_uses.size() == 0)
    {
      var_uses = {2,1,2,1,1,1,1,1};
    }
    if(var_uses.size() != X.cols)
    {
      cout << "error where" << X.rows << ", " << X.cols << endl;
      assert(false);
    }
    
    // count the poss and negs
    n_pos = 0; n_neg = 0; n_agg = X.rows;
    for(int idx = 0; idx < X.rows; idx++)
      if(Y.at<float>(idx) < 0)
	n_neg++;
      else
	n_pos++;
    #pragma omp critical
    cout << "DetectionTree::DetectionTree (pos,neg) = (" << n_pos << ", " << n_neg << ")" << endl;
	
    auto make_leaf = [this]() 
    {
      node_type = LEAF; 
    };
    if(X.rows < MIN_SPLIT || n_neg == 0 || n_pos == 0 || depth + 1 >= MAX_DEPTH)
    {
      // we don't need to split.
      make_leaf();      
      return;
    }
    
    // otherwise, choose the optimal split point.
    decision = choose_split(X,Y,var_uses);
    node_type = INTERNAL;
    
    // in parallel, compute the subtrees
    // #pragma omp parallel for
    for(int side = 0; side < 2; side++)
    {
      Mat XPrime, YPrime;
      split_data(XPrime, YPrime, X, Y, side);
     
      // special case,... we couldn't gain information with a split?
      if(XPrime.rows == n_agg || XPrime.rows == 0)
      {
	make_leaf();
	printf("warning: Split failed\n");
	break;
      }  	     
     
      // recursively create two new trees.
      ((side==0)?branch_true:branch_false) = shared_ptr<DetectionTree>(new DetectionTree(XPrime,YPrime,var_uses,depth+1));
    }
  }
  
  DetectionTree::DetectionTree(cv::Mat& X, cv::Mat& Y, std::vector< int > var_uses, int depth) 
  { 
    if(depth == 0)
    {
      ostringstream oss;
      oss << "cache/detection_tree_" << hashMat(X) << ".yml";
      FileStorage cache_file(oss.str(),FileStorage::READ);
      
      if(cache_file.isOpened())
      {
	cache_file["DetectionTree"] >> *this;
	cache_file.release();
	return;
      }
    }
    train(X, Y, var_uses, depth);
    
    // try to save
    if(depth == 0)
    {
      ostringstream oss;
      oss << "cache/detection_tree_" << hashMat(X) << ".yml";
      FileStorage cache_file(oss.str(),FileStorage::WRITE);
      cache_file << "DetectionTree" << *this;
      cache_file.release();
    }
  }
  
  DetectionTree::DetectionTree()
  {
  }
  
  double DetectionTree::predict_here()
  {
    if(POS_WEIGHT*n_pos > n_neg)
      return +inf;
    else
      return -inf;
  }

  double DetectionTree::predict(Mat& x)
  {
    auto resp = [this,&x](shared_ptr<DetectionTree> branch,shared_ptr<DetectionTree> alternate)
    {
      double cur_resp = branch->predict(x);
      double cur_sign = sign(cur_resp);
      
      bool same_sign = cur_resp * alternate->predict_here() > 0;
      if(same_sign)
	return cur_resp;
      
      cur_resp = cur_sign * cur_resp; // work with positve
      double new_dist = decision->boundary_dist(x);
      return cur_sign*std::min(cur_resp,new_dist);
    };
    
    if(node_type == LEAF)
      return predict_here();
    else
      if(decision->decide(x))
      {
	return resp(branch_true,branch_false);
      }
      else
      {
	return resp(branch_false,branch_true);
      }
  }
  
  static void tabs(int count)
  {
    while(count > 0)
    {
      cout << "\t";
      count--;
    }
  }
  
  void DetectionTree::show(int depth)
  {
    if(node_type == LEAF)
    {
      tabs(depth);
      cout  << "LEAF pos% = " << 100*n_pos/n_agg 
	    << " pos = " << n_pos 
	    << " neg = " << n_neg << endl;
    }
    else
    {
      tabs(depth);
      decision->println();
      branch_false->show(depth+1);
      branch_true->show(depth+1);
    }
  }
  
  /// SECTION: Decision Implementation
  DecisionThreshold::DecisionThreshold(shared_ptr<vector<float >>  values, int best_var, float best_value) :
    best_var(best_var),
    best_value(best_value),
    Decision(values)
  {
  }
  
  bool DecisionThreshold::decide(const Mat& x)
  {
    return x.at<float>(0,best_var) < best_value;
  }

  void DecisionThreshold::println()
  {
     cout << "INTERN: split var = " << best_var << " thresh = " <<  best_value << endl;
  }
  
  void DecisionThreshold::update_uses(std::vector< int >& var_uses)
  {
    var_uses[best_var]--;
  }
  
  double DecisionThreshold::boundary_dist(const Mat& x)
  {
    float xv = x.at<float>(0,best_var);
    return std::abs(xv-best_value)/varience();
  }
  
  void DecisionThreshold::write(FileStorage& fs)
  {
    fs << "decision_type" << "Threshold";
    deformable_depth::Decision::write(fs);
    fs << "best_var" << best_var;
    fs << "best_value" << best_value;
  }
  
  DecisionThreshold::DecisionThreshold(double mu, double sigma, int best_var, float best_value): 
    Decision(mu, sigma), best_var(best_var), best_value(best_value)
  {

  }
  
  /// SECTION: Range decision
  bool DecisionRange::decide(const Mat& x)
  {
    float xv = x.at<float>(0,var);
    return xv >= low && xv <= high;
  }

  DecisionRange::DecisionRange(std::shared_ptr< std::vector< float > > values, int var, float low, float high) : 
    var(var), low(low), high(high), Decision(values)
  {
  }

  void DecisionRange::println()
  {
    cout  << "INTERN: Split Var = " << var 
	  << " Range = [" << low << ", " << high << "]" << endl;
  }

  void DecisionRange::update_uses(std::vector< int >& var_uses)
  {
    var_uses[var] -= 2;
    assert(var_uses[var] >= 0);
  }
  
  double DecisionRange::boundary_dist(const Mat& x)
  {
    float xv = x.at<float>(0,var);
    return std::min(std::abs(xv-low)/varience(),std::abs(xv-high)/varience());
  }
  
  void DecisionRange::write(FileStorage& fs)
  {
    fs << "decision_type" << "Range";
    deformable_depth::Decision::write(fs);
    fs << "var" << var;
    fs << "low" << low;
    fs << "high" << high;
  }
  
  DecisionRange::DecisionRange(double mu, double sigma, int var, float low, float high): 
    Decision(mu, sigma),
    var(var), low(low), high(high)
  {

  }
  
  /// SECTION: Common Decision Parameters
  Decision::Decision(shared_ptr<vector<float >> values) :
    values(values)
  {
  }
  
  void Decision::computeMoments()
  {
    mu = deformable_depth::sum(*values)/(float)values->size();
    vector<float> v2 = *values;
    for(float&v : v2)
      v *= v;
    double x2 = deformable_depth::sum(v2)/(double)values->size();
    sigma = x2 - mu*mu;
  }
  
  double Decision::mean()
  {
    return mu;
  }

  double Decision::varience()
  {
    return sigma;
  }

  Decision::~Decision()
  {
  }
  
  void Decision::write(FileStorage& fs)
  {
    fs << "mu" << mu;
    fs << "sigma" << sigma;
  }
  
  Decision::Decision(double mu, double sigma) :
    mu(mu), sigma(sigma)
  {

  }
}

