/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "FauxLearner.hpp"
#include "util.hpp"
#include "vec.hpp"

namespace deformable_depth
{  
  /// FAUX Learner implementation section
  FauxLearner::FauxLearner(std::vector< double > w, double beta) : 
    w(w), beta(beta)
  {
  }
  
  double FauxLearner::getB() const
  {
    return beta;
  }

  std::vector< double > FauxLearner::getW() const
  {
    return w;
  }

  void FauxLearner::opt(int iter, double tol)
  {
    // nop
  }

  std::vector< float > FauxLearner::getWf() const
  {
    return vec_d2f(w);
  }

  bool FauxLearner::write(std::vector< float > xx, float y,string id)
  {
    // nop
    return false;
  }

  bool FauxLearner::write(std::vector< double > xx, float y,string id)
  {
    // nop
    return false;
  }

  bool FauxLearner::write(SparseVector& x, float y, string id)
  {
    // nop
    return false;
  }  
  
  void FauxLearner::setB(double beta)
  {
    this->beta = beta;
  }

  void FauxLearner::setW(std::vector< double > w)
  {
    this->w = w;
  }
  
  void FauxLearner::setWB(std::vector< double > wb)
  {
    beta = wb[wb.size()-1];
    wb.pop_back();
    w = wb;
  }
  
  float FauxLearner::predict(std::vector< float > x) const
  {
    vector<float> wfb = getWf();
    wfb.push_back(getB());
    x.push_back(1);
    return dot(x,wfb);
  }
  
  void FauxLearner::prime(int wLength)
  {
    if(w.size() == 0)
    {
      beta = 0;
      w = vector<double>(wLength,0);
    }
  }
  
  /// serialization for the multi feature model
  void write(FileStorage& fs, const string& , const FauxLearner& write)
  {
    fs << "{" << "w" << write.getW() << "b" << write.getB() << "}";
  }

  void read(const FileNode& node, FauxLearner& read, const FauxLearner& default_value)
  {
    vector<double> w; 
    double B;
    node["w"] >> w;
    B = node["B"];
    read = FauxLearner(w,B);
  }
  
  void read(const FileNode& node, shared_ptr< FauxLearner >& learner, shared_ptr< FauxLearner > )
  {
    learner.reset(new FauxLearner());
    read(node,*learner);
  }  
}
