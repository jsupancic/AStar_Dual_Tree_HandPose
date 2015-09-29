/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Kahan.hpp"
#include <iostream>
#include <math.h>
#include "util_real.hpp"
#include "util_mat.hpp"
#include "boost/multi_array.hpp"

namespace deformable_depth
{
  using namespace std;
  
  /// SECTION: Implements the Kahan Summation algorithm
  
  KahanSummation::KahanSummation() : 
    summation(0),
    correction(0)
  {
  }

  KahanSummation& KahanSummation::operator=(double reset_to)
  {
    summation = reset_to;
    correction = 0;
    return *this;
  }
  
  KahanSummation& KahanSummation::operator=(const KahanSummation& reset_to)
  {
    summation = reset_to.summation;
    correction = reset_to.correction;
    return *this;
  }
  
  double KahanSummation::current_total() const
  {
    return summation;
  }
  
  KahanSummation& KahanSummation::operator+=(double amount)
  {
    double y = amount - correction;
    double t = summation + y;
    correction = (t - summation) - y;
    summation = t;
    return *this;
  }
  
  void kahan_test()
  {
    float f_sum = 0;
    KahanSummation k_sum;
    float inc = 1e-6;
    for(long iter = 0; iter < 1000000; iter++)
    {
      f_sum += inc;
      k_sum += inc;
      if(iter % 10000)
      {
	cout << "===============================" << endl;
	cout << "k_sum: " << k_sum.current_total() << endl;
	cout << "f_sum: " << f_sum << endl;
	cout << "times: " << inc*iter << endl;
      }
    }
  }
  
  /// SECTION: Implements the KahanVector of Summations
  KahanVector::KahanVector(size_t length) : 
    values(length), dot_self_cache(qnan)
  {
  }
  
  KahanVector& KahanVector::operator=(const KahanVector& other)
  {
    values = other.values;
    dot_self_cache = other.dot_self_cache;
    return *this;
  }
  
  KahanVector::KahanVector() : dot_self_cache(qnan)
  {}
  
  KahanVector::KahanVector(const std::vector< double >& prime) :
    values(prime.size()), dot_self_cache(qnan)
  {
    for(int pos = 0; pos < prime.size(); pos++)
      values[pos] = prime[pos];
  }

  void KahanVector::onModified()
  {
    dot_self_cache = qnan;
  }
  
  KahanVector& KahanVector::operator+=(const vector<double>& other)
  {
    onModified();
    
    assert(values.size() == other.size());
    for(int pos = 0; pos < values.size(); pos++)
      values[pos] += other[pos];
    return *this;
  }

  KahanVector KahanVector::operator-(const KahanVector& other) const
  {
    KahanVector result(values.size());
    
    for(int pos = 0; pos < values.size(); pos++)
    {
      result.values[pos] = values[pos];
      result.values[pos] += -other.values[pos].current_total();
    }
    
    return result;
  }
  
  KahanVector::operator vector<double>() const
  {
    vector<double> result(values.size());
    
    for(int pos = 0; pos < values.size(); pos++)
      result[pos] = values[pos].current_total();
    
    return result;
  }
  
  double KahanVector::operator*(const std::vector< float > other) const
  {
    double ddot = 0;
    
    assert(other.size() == values.size());
    for(int pos = 0; pos < values.size(); pos++)
      ddot += values[pos].current_total()*other[pos];
    
    return ddot;
  }
  
  double KahanVector::dot_self() const
  {
    if(goodNumber(dot_self_cache))
      return dot_self_cache;
    
    KahanSummation d;
    
    for(const KahanSummation&value : values)
    {
      double current = value.current_total();
      assert(goodNumber(current));
      d += current*current;
    }
    
    dot_self_cache = d.current_total();
    return d.current_total();
  }

  size_t KahanVector::size() const
  {
    return values.size();
  }
  

  const KahanSummation&KahanVector::operator[](size_t idx) const
  {
    return values[idx];
  }
}
