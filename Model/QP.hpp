/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_QP
#define DD_QP

#define use_speed_ 0
#include <cv.h>

#include <boost/functional/hash.hpp>
#include <unordered_set>
#include "vec.hpp"
#include "params.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
    
  static constexpr double DEFAULT_TOLERANCE = 0.20;
  
  struct SVMExample
  {
  protected:
    double B; // margin
    SparseVector x;
  public:
    double alpha;
    double D;
    bool pos;
    bool active;
    
  public:
    SVMExample(SparseVector&x_, string id, 
	       double alpha_, double D_, double B_, bool pos);
    virtual SparseVector&getX() = 0;
    virtual double getB() = 0;
    virtual string key() const = 0;
    double y() const;
  };
  
  class MILSVMExample : public SVMExample
  {
  public:
    MILSVMExample(SparseVector&x_, string id, double alpha_, double D_, double B_, bool pos);
    virtual SparseVector&getX();
    virtual double getB();
    virtual bool update(string id);
    virtual string key() const;
  protected:
    set<string> ids;
    double weight;
    static constexpr bool WEIGHT_BY_COUNT = false;
  };
  
  class LatentSVMExample : public SVMExample
  {
  public:
    LatentSVMExample(SparseVector&x_, string id, double alpha_, double D_, double B_, bool pos);
    virtual SparseVector&getX();
    virtual double getB();
    virtual bool update(string id,LatentSVMExample&newExample,const vector<double>&w);
    virtual string key() const;
  protected:
    string id;
  };
  
  typedef LatentSVMExample Example;
  
  // linear disciminative analysis
  class LDA
  {
  public:
    virtual void opt(int iter = 1000, double tol = DEFAULT_TOLERANCE) = 0;
    virtual bool write(SparseVector&x, float y, string id) = 0;
    virtual bool write(std::vector< float > xx, float y, string id)  = 0;
    virtual bool write(std::vector< double > xx, float y, string id) = 0;
    virtual vector<double> getW() const = 0;
    virtual vector<float> getWf() const = 0;
    virtual double getB() const = 0;
    virtual float predict(vector<float> x) const = 0;
    // init the template to a specified value
    virtual void prime(int wLength) = 0;
  public:
    size_t getFeatureLength() const;
  };
  
  constexpr static size_t MIN_SIZE = params::GB4/2; // 2GB
  
  class QP : public LDA
  {
  public:
    QP(float C = 1);
    bool write(SparseVector&x, float y, string id);
    bool write(std::vector< float > xx, float y, string id);
    bool write(std::vector< double > xx, float y, string id);
    void opt(int iter = 1000, double tol = DEFAULT_TOLERANCE);
    float predict(Mat_<float> x) const; 
    float predict(vector<float> x) const;
    double obj_ub();
    double apx_lb();
    double apx_ub();
    bool isTrained();
    // access the trained discriminant
    vector<double> getW() const;
    vector<float> getWf() const;
    double getB() const;
    void prime(int wLength);
    void prime(vector<double> w, double b);
    size_t cache_size() const;
    size_t footprint() const;
    size_t num_SVs() const;
    bool is_overcapacity();
  private:
    vector<double> w;
    // vector of training examples
    map<string/*feat hash*/,Example> cache;
    bool trained; 
    double C;
    double lb, ub;
    double l;
    int nsv;
    size_t m_footprint;
    size_t max_size;
  private:
  protected:
    void recompute_weights();
    void update_capacity();
    void activate_cache();
    void prune();
    void trainSVM_fast(int ITER, double TOL);
    void opt_one(Example&example,double&loss);
  };
}

#endif
