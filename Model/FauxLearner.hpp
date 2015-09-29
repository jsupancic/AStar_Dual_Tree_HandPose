/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_FAUX_LEARNER
#define DD_FAUX_LEARNER

#include "Detector.hpp"
#include "OneFeatureModel.hpp"

namespace deformable_depth
{
  class FauxLearner : public LDA
  {  
  private:
    vector<double> w;
    double beta;
  public:
    FauxLearner() {};
    FauxLearner(vector<double> w, double beta);
    virtual void opt(int iter = 1000, double tol = 0.02);
    virtual bool write(std::vector< float > xx, float y,string id);
    virtual bool write(SparseVector&x, float y, string id);
    virtual bool write(std::vector< double > xx, float y,string id);
    virtual vector<double> getW() const;
    virtual vector<float> getWf() const;
    virtual double getB() const;
    void setW(vector<double> w);
    void setB(double beta);
    void setWB(vector<double> wb);
    float predict(vector<float> x) const;
    virtual void prime(int wLength);
  public:
    friend void write(FileStorage& fs, const string& , const FauxLearner& write);
    friend void read(const FileNode& node, FauxLearner& read,const FauxLearner& default_value);
    friend void read(const FileNode&, shared_ptr<FauxLearner>&, shared_ptr<FauxLearner>);
  };
  void write(FileStorage& fs, const string& , const FauxLearner& write);
  void read(const FileNode& node, FauxLearner& read,
	    const FauxLearner& default_value = FauxLearner());
  void read(const FileNode&, shared_ptr<FauxLearner>&, shared_ptr<FauxLearner>);
}

#endif


