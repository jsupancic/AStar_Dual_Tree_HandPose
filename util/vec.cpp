/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "vec.hpp"
#include <assert.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "hashMat.hpp"
#include "util_real.hpp"

namespace deformable_depth
{
  using std::cout;
  using std::endl;
  
  vector<double> mult(double s, vector<double> vec)
  {
    for(int iter = 0; iter < vec.size(); iter++)
      vec[iter] *= s;
    return vec;
  }
      
  std::vector< double > add(double s, std::vector< double > vec)
  {
    for(int iter = 0; iter < vec.size(); iter++)
      vec[iter]+=s;
    return vec;
  }

  double norm(const vector< double >& in)
  {
    double norm = 0;
    
    for(const double&x : in)
      norm += x*x;
    
    return std::sqrt(norm);
  }
  
  vector< double > normalize(const vector< double >& in)
  {
    return mult(1/norm(in),in);
  }
  
  std::vector< double > add(std::vector< double > v1, std::vector< double > v2)
  {
    vector<double> result = v1;
    assert(v1.size() == v2.size());
    
    for(int iter = 0; iter < v1.size(); iter++)
      result[iter] += v2[iter];
      
    return result;
  }
      
  vector<double> vec_f2d(const vector<float>&in)
  {
    vector<double> r;
    
    for(const float&f : in)
      r.push_back(f);
      
    return r;
  }
  
  std::vector< float > vec_d2f(const std::vector< double >& in)
  {
    vector<float> r;
    for(const double&f : in)
      r.push_back(f);
    return r;
  }
  
  void print(std::vector< double >& x)
  {
    cout << "vector = [" ;
    for(int iter = 0; iter < x.size(); iter++)
    {
      cout << x[iter];
      if(iter + 1 < x.size())
	cout << ", ";
    }
    cout << "]" << endl;
  }
  
  double weighted_geometric_mean(vector<double> values, vector<double> weights)
  {
    assert(values.size() == weights.size());
    double num = 0;
    double den = 0;
    for(int iter = 0; iter < values.size(); iter++)
    {
      num += weights[iter]*std::log(values[iter]);
      den += weights[iter];
    }
    return std::exp(num/den);
  }
  
  /// Implemnting the sparse vector
  
  SparseVector::operator vector<float>() const
  {
    vector<float> result(length);
    
    for(int strip = 0; strip < strips.size(); strip++)
    {
      std::copy(strips[strip].begin(),strips[strip].end(),result.begin()+starts[strip]);
    }
    
    return result;
  }

  SparseVector::operator vector<double>() const
  {
    vector<double> result(length);
    
    for(int strip = 0; strip < strips.size(); strip++)
    {
      std::copy(strips[strip].begin(),strips[strip].end(),result.begin()+starts[strip]);
    }
    
    return result;
  }
  
  void SparseVector::set_strip(std::size_t at, const std::vector< double >& values)
  {
    // make sure it fits
    bool good_position = at >= 0 && at + values.size() - 1 < length;
    if(!good_position)
    {
      cout << "at = " << at << endl;
      cout << "values.size() = " << values.size() << endl;
      cout << "length = " << length << endl;
    }
    assert(good_position);
    
    // make sure it doesn't overlap another strip
    int prev_strip = strip_at_idx(at);
    assert(prev_strip  == strip_at_idx(at + values.size() - 1));
    assert(prev_strip == -1 || starts[prev_strip ] + strips[prev_strip ].size() - 1 < at);
    
    // insert
    int insert_at = prev_strip + 1;
    starts.insert(starts.begin()+insert_at,at);
    strips.insert(strips.begin()+insert_at,values);
  }
  
  void SparseVector::set(std::size_t at, const SparseVector&values)
  {
    for(int iter = 0; iter < values.strips.size(); iter++)
    {
      const vector<double>&strip = values.strips[iter];
      size_t strip_start_in_values = values.starts[iter];
      size_t strip_start_in_this = at + strip_start_in_values;
      set_strip(strip_start_in_this,strip);
    }
  }

  void SparseVector::push_back(SparseVector& appendMe)
  {
    for(int other_strip_iter = 0; other_strip_iter < appendMe.strips.size(); other_strip_iter++)
    {
      strips.push_back(appendMe.strips[other_strip_iter]);
      starts.push_back(length);
      length += appendMe.strips[other_strip_iter].size();
    }
  }
  
  void SparseVector::push_back(double value)
  {
    starts.push_back(length);
    length += 1;
    vector<double> nv = {value};
    strips.push_back(nv);
  }
  
  int SparseVector::strip_at_idx(size_t idx) const
  {
    // find the candidate strip with binary search
    int strip_id = std::upper_bound(starts.begin(),starts.end(),idx) - 1 - starts.begin();
    if(strip_id < -1)
    {
      cout << "starts.size = " << starts.size() << endl;
      cout << "strip_id: " << strip_id << endl;
      assert(false);
    }
    return strip_id;
  }
  
  double SparseVector::operator[](size_t idx) const
  {
    int strip_id = strip_at_idx(idx);
    if(strip_id == -1)
      return 0.0;
    
    int index_into_strip = idx - starts[strip_id];
    const vector<double>& strip = strips[strip_id];
    if(index_into_strip >= 0 && index_into_strip < strip.size())
      return strip[index_into_strip];
    else
      return 0.0;
  }

  SparseVector::SparseVector(const std::vector< double >& copyMe)
  {
    length = copyMe.size();
    starts.push_back(0);
    strips.push_back(copyMe);
  }
  
  SparseVector::SparseVector(const std::vector< float >& copyMe)
  {
    length = copyMe.size();
    starts.push_back(0);
    strips.push_back(vec_f2d(copyMe));
  }
  
  SparseVector::SparseVector(std::vector< float >&& copyMe)
  {
    length = copyMe.size();
    starts.push_back(0);
    vector<double> dCopyMe = vec_f2d(copyMe);
    strips.push_back(dCopyMe);
  }

  SparseVector::SparseVector(SparseVector&& tmp)
  {
    length = std::move(tmp.length);
    starts = std::move(tmp.starts);
    strips = std::move(tmp.strips);
  }

  SparseVector::SparseVector(const SparseVector& other)
  {
    length = other.length;
    starts = other.starts;
    strips = other.strips;
  }
  
  SparseVector::SparseVector(int length) : 
    length(length)
  {
  }
  
  size_t SparseVector::size() const
  {
    return length;
  }

  vector<double> SparseVector::operator+(std::vector< double >& other) const
  {
    vector<double> result = other;
    
    for(int strip_id = 0; strip_id < strips.size(); strip_id++)
    {
      const vector<double>&strip = strips[strip_id];
      int index_into_result = starts[strip_id];
      for(int index_into_strip = 0; index_into_strip < strip.size(); index_into_strip++, index_into_result++)
	result[index_into_result] += strip[index_into_strip];
    }
    
    return result;
  }
  
  vector< float > SparseVector::get(std::size_t at) const
  {
    assert(at < strips.size());
    return vec_d2f(strips[at]);
  }
  
  std::size_t SparseVector::strip_count()
  {
    return strips.size();
  }
  
  std::size_t SparseVector::strip_start(std::size_t strip_id)
  {
    assert(strip_id < starts.size());
    return starts[strip_id];
  }
  
  // dot product
  double SparseVector::operator*(const vector<double>& other) const
  {
    double dot = 0;
    
    for(int strip_id = 0; strip_id < strips.size(); strip_id++)
    {
      const vector<double>& strip = strips[strip_id];
      assert(strip_id < starts.size());
      int global_index = starts[strip_id];
      assert(global_index + strip.size() - 1 < other.size());
      for(int local_index = 0; local_index < strip.size(); local_index++, global_index++)
	dot += other[global_index]*strip[local_index];
    }
    
    return dot;
  }
 
  double SparseVector::dot_self() const
  {
    double dot = 0;
    
    for(const vector<double>&strip: strips)
      for(const double&value : strip)
	dot += value*value;
    
    return dot;
  }

  SparseVector SparseVector::operator+(double scalar) const
  {
    SparseVector result(*this);
    
    for(vector<double> & strip : result.strips)
      for(int iter = 0; iter < strip.size(); iter++)
	strip[iter] += scalar;
    
    return result;
  }
  
  SparseVector SparseVector::operator*(double scalar) const
  {
    SparseVector result(*this);
    
    for(vector<double> & strip : result.strips)
      for(int iter = 0; iter < strip.size(); iter++)
	strip[iter] *= scalar;
    
    return result;    
  }
  
  string SparseVector::hash() const
  {
    if(hash_value != string())
      return hash_value;
    
    vector<double> d_starts;
    for(size_t start : starts)
      d_starts.push_back(start);
      
    MD5_HASH hash_value = deformable_depth::hash(strips)^deformable_depth::hash(d_starts);
    
    return hash2string(hash_value);
  }

  SparseVector& SparseVector::operator=(const SparseVector& other)
  {
    length = other.length;
    strips = other.strips;
    starts = other.starts;
    return *this;
  }
  
  size_t SparseVector::footprint_bytes() const
  {
    size_t footprint = 0;
    for(const vector<double>&strip : strips)
      footprint += sizeof(double)*strip.size();
    return footprint;
  }
  
  double SparseVector::sparsity() const
  {
    double fp_bytes = footprint_bytes();
    double mx_bytes = size()*sizeof(double);
    return 1 - fp_bytes/mx_bytes;
  }
  
  void test_sparse()
  {
    // setup
    SparseVector test(10);
    vector<double> vec = {2,3};
    SparseVector sv(vec);
    test.set(5,sv);
    test.set(1,sv);
    
    // test via printing
    for(int iter = 0; iter < 10; iter++)
      cout << test[iter] << endl;
  }
  
  /// SECTION: PreciseVector implementation
#ifdef DD_MULTPREC
  mpf_class PreciseVector::dot(const PreciseVector& other) const
  {
    mpf_class pdot(0,std::max((*this)[0].get_prec(),other[0].get_prec()));

    assert(size() == other.size());
    for(int pos = 0; pos < other.size(); pos++)
      pdot += (*this)[pos]*other[pos];
      
    return pdot;
  }

  void PreciseVector::MAC(SparseVector& v, mpf_class coeff)
  {
    vector<float> vv = v;
    
    for(int pos = 0; pos < size(); pos++)
      (*this)[pos] += coeff * vv[pos];
  }
  
  mpf_class PreciseVector::dot(const std::vector< float >& other) const
  {
    mpf_class dv(0,precision);
    
    assert(size() == other.size());
    for(int pos = 0; pos < size(); pos++)
      dv += (*this)[pos] * other[pos];
    
    return dv;
  }

  PreciseVector PreciseVector::operator-(const PreciseVector& other)
  {
    PreciseVector result(precision,size());
    
    for(int pos = 0; pos < size(); pos++)
      result[pos] = (*this)[pos] - other[pos];
    
    return result;
  }
  
  PreciseVector::PreciseVector(mp_bitcnt_t precision, size_t length) :
    precision(precision), vector(length,mpf_class(0,precision))
  {
    for(int pos = 0; pos < size(); pos++)
      assert((*this)[pos].get_d() == 0);
  }
  
  PreciseVector::PreciseVector(mp_bitcnt_t precision, const std::vector< double >& init) :
    precision(precision), vector(init.size(),mpf_class(0,precision))
  {
    for(int iter = 0; iter < init.size(); iter ++)
      (*this)[iter] = mpf_class(init[iter],precision);
  }  
  
  void PreciseVector::set_precision(mp_bitcnt_t new_prec)
  {
    precision = new_prec;
    for(int pos = 0; pos < size(); pos++)
      (*this)[pos].set_prec(new_prec);
  }

  PreciseVector::operator vector<double>()const
  {
    vector<double> result(size());
    
    for(int pos = 0; pos < size(); pos++)
      result[pos] = (*this)[pos].get_d();
    
    return result;
  }
#endif

  cv::Vec3b max(cv::Vec3b a,cv::Vec3b b)
  { 
    double v0 = (double)a[0] + (double)b[0];
    double v1 = (double)a[1] + (double)b[1];
    double v2 = (double)a[2] + (double)b[2];
    
    return Vec3b(v0/2,v1/2,v2/2);
  }  
  
  float order(const vector< float >& vec, float order)
  {
    int idx = clamp<int>(0,order*(vec.size()-1),vec.size());
    return vec[idx];
  }
  
  vector< double > convert(const Vec3d& v3d)
  {
    return vector<double>{v3d[0],v3d[1],v3d[2]};
  }
}
  