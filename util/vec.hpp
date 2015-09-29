/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_VEC
#define DD_VEC

#include <vector>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string>
#ifdef DD_MULTPREC
#include <gmpxx.h>
#endif
#include <opencv2/opencv.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <limits>

namespace deformable_depth
{
  using namespace std;
  using cv::Vec3b;
  
  class SparseVector;
  class PreciseVector;
#ifdef DD_MULTPREC
  // used for w vector in QP
  class PreciseVector : public vector<mpf_class>
  {
  protected:
    mp_bitcnt_t precision;
  public:
    PreciseVector(mp_bitcnt_t precision, size_t length);
    PreciseVector(mp_bitcnt_t precision, const vector<double>&init);
    void MAC(SparseVector&v,mpf_class coeff);
    mpf_class dot(const PreciseVector&other) const;
    mpf_class dot(const vector<float>&other) const;
    void set_precision(mp_bitcnt_t new_prec);
    PreciseVector operator-(const PreciseVector&other);
    operator vector<double>() const;
  };
#endif
  // used for data terms
  class SparseVector
  {
  public:
    SparseVector(const vector<double>&copyMe);
    SparseVector(const vector<float>&copyMe);
    SparseVector(vector<float>&&copyMe);
    SparseVector(int length);
    SparseVector(SparseVector&&tmp);
    SparseVector(const SparseVector&other);
    SparseVector();
    SparseVector& operator=(const SparseVector&other);
    // this is inefficent O(log[n]) but provided for reference
    double operator[](size_t idx) const;
    operator vector<double>() const;
    operator vector<float>() const;
    void set(size_t at,const SparseVector&values);
    vector<float> get(size_t at) const;
    size_t strip_count();
    size_t strip_start(size_t strip_id);
    void push_back(SparseVector&appendMe);
    void push_back(double value);
    size_t size() const;
    string hash() const;
    size_t footprint_bytes() const;
    // how sparse is this sparse vector
    // 1 indicates it's essentially a vector<float>
    // 0 indicates it's all empty
    double sparsity() const;
    // arithmetic operators
  public:
    // dot product
    double operator*(const vector<double>&other) const;
    double dot_self() const;
    // scalar product
    SparseVector operator*(double scalar) const;
    vector<double> operator+(vector<double>&other) const;
    SparseVector operator+(double scalar) const;
  protected:
    void set_strip(size_t at,const vector<double>&values);
    int strip_at_idx(size_t idx) const;
  protected:
    string hash_value;
    size_t length;
    vector<size_t> starts;
    vector<vector<double> > strips;
  };  
  
  vector<double> mult(double s, vector<double> vec);
  vector<double> add(double s, vector<double> vec);
  vector<double> add(vector<double> v1, vector<double> v2);
  vector<double> vec_f2d(const vector<float>&in);
  vector<float> vec_d2f(const vector<double>&in);
  double norm(const vector<double>&in);
  vector<double> normalize(const vector<double>&in);
  template<typename R, typename S>
  vector<R>&operator+=(vector<R>&me, vector<S>&add)
  {
    assert(me.size() == add.size());
    for(int iter = 0; iter < me.size(); iter++)
      me[iter] += add[iter];
  }
  template<typename R>
  vector<R> operator+ (const vector<R>&a, const vector<R>&b)
  {
    assert(a.size() == b.size());
    vector<R> r(a.size());
    
    for(size_t iter = 0; iter < a.size(); ++iter)
      r[iter] = a[iter] + b[iter];
    
    return r;
  }
  template<typename R>
  vector<R> operator/(const vector<R>&a, const vector<R>&b)
  {
    assert(a.size() == b.size());
    vector<R> r(a.size());
    
    for(size_t iter = 0; iter < a.size(); ++iter)
      r[iter] = a[iter] / b[iter];
    
    return r;
  }
  template<typename R>
  vector<R> operator/(const vector<R>&a, const R b)
  {
    vector<R> r(a.size());
    
    for(size_t iter = 0; iter < a.size(); ++iter)
      r[iter] = a[iter] / b;
    
    return r;
  }
  template<typename R>
  vector<R> operator*(const vector<R>&a, const R b)
  {
    vector<R> r(a.size());
    
    for(size_t iter = 0; iter < a.size(); ++iter)
      r[iter] = a[iter] * b;
    
    return r;
  }
  template<typename R,typename S>
  vector<R>&operator/=(vector<R>&me, S&div)
  {
    for(int iter = 0; iter < me.size(); iter++)
      me[iter] /= div;
    return me;
  }
  template<typename R>
  vector<R> operator-(const vector<R>&v1,const vector<R>&v2)
  {
    vector<R> d(v1.size());
    
    assert(v1.size() == v2.size());
    for(int iter = 0; iter < v1.size(); iter++)
      d[iter] = v1[iter] - v2[iter];
    
    return d;
  }
  template<typename R>
  vector<R> operator-(const vector<R>&v1, R scalar)
  {
    vector<R> d(v1.size());
    
    for(int iter = 0; iter < v1.size(); ++iter)
      d[iter] = v1[iter] - scalar;
    
    return d;
  }
  template<typename R>
  vector<R> operator-(const vector<R>& in)
  {
    vector<R> out = in;
    for(auto && value : out)
      value = -value;
    return out;
  }
  template<typename R>
  std::ostream& write_vec(std::ostream&out, const std::vector<R>& vec)
  {
    for(int iter = 0; iter < vec.size(); ++iter)
      if(iter == vec.size() - 1)
	out << vec[iter];
      else
	out << vec[iter] << ", ";
    return out;
  }
  template<typename R>
  std::ostream& operator<< (std::ostream&out, const std::vector<R>& vec)
  {
    out << "[";
    
    write_vec(out,vec);
    
    out << "]";
    
    return out;
  }
  
  template<typename T>
  T sum(const std::vector< T >& in)
  {
    T s = 0;
    for(int iter = 0; iter < in.size(); iter++)
      if(!std::isinf(in[iter]) && !std::isnan(in[iter]))
	s += in[iter];
    return s;
  }
  
  template<typename T>
  T mean(const std::vector<T> & in)
  {
    double s = sum(in);
    return s/static_cast<double>(in.size());
  }
  
  template<typename A, typename B>
  static double dot(const vector<A>&a,const vector<B>&b)
  {
    if(a.size() != b.size())
    {
      printf("a.size() = %d and b.size() = %d\n",(int)a.size(),(int)b.size());
      assert(a.size() == b.size());
    }
    double dot = 0;
    for(int iter = 0; iter < a.size(); iter++)
      dot += a[iter]*b[iter];
    return dot;
  }
  
  template<typename A>
  static double norm_l2(const vector<A>&a)
  {
    return std::sqrt(dot(a,a));
  }
  
  template<typename A>
  static double dot_self(const vector<A>&a)
  {
    return dot(a,a);
  }
  
  // make vector have zero mean and unit standard deviation
  template<typename T>
  void standardize(vector<T>&v)
  {
    // compute thew mean
    double mean = sum(v)/v.size();    
    
    // compute the second moment
    vector<double> v_sq;
    for(T value : v)
      v_sq.push_back(value*value);
    double sd = std::sqrt(sum(v_sq)/v_sq.size()) - mean*mean;
    
    // subtract mean and divide by sd
    v = v - mean;
    v /= sd;
  }
  
  float order(const vector<float>&vec,float order);
  
  cv::Vec3b max(cv::Vec3b a,cv::Vec3b b);
  
  void print(vector<double> & x);
  double weighted_geometric_mean(vector<double> values, vector<double> weights);
  vector<double> convert(const cv::Vec3d&);
  
  template<typename T>
  vector<size_t> sorted_indexes(const vector<T>&v)
  {
    vector<size_t> indexes(v.size());
    std::iota(indexes.begin(),indexes.end(),0);
    
    std::sort(indexes.begin(),indexes.end(),
	      [&](size_t idx1, size_t idx2){return v[idx1] < v[idx2];});
    
    return indexes;
  }
  
  template<typename T>
  double pearson(const vector<T>&v1, const vector<T>&v2)
  {
    double mu1 = mean(v1);
    double mu2 = mean(v2);
    
    double t = 0, l = 0, r = 0;
    for(size_t iter = 0; iter < v1.size(); iter++)
    {
      t += (v1[iter] - mu1)*(v2[iter] - mu2);
      l += (v1[iter] - mu1)*(v1[iter] - mu1);
      r += (v2[iter] - mu2)*(v2[iter] - mu2);
    }
    
    return t / (std::sqrt(l)*std::sqrt(r));
  }
  
  template<typename T>
  double spearman(const vector<T>&v1, const vector<T>&v2)
  {
    vector<size_t> idxs1 = sorted_indexes(v1);
    vector<size_t> idxs2 = sorted_indexes(v2);
    
    return pearson(idxs1,idxs2);
  }

  template<typename T>
  T max(const vector<T>&vec)
  {
    T m = -std::numeric_limits<T>::infinity();
    for(auto && v : vec)
      if(v > m)
	m = v;
    return m;
  }
  
  void test_sparse();
}

#endif
