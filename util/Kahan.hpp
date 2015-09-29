/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_KAHAN
#define DD_KAHAN

#include <vector>
#include <stdlib.h>
#include <assert.h>
#include <opencv2/opencv.hpp>

namespace deformable_depth
{
  using std::vector;
  
  class KahanSummation
  {
  protected:
    double summation;
    double correction;
  public:
    KahanSummation();
    KahanSummation& operator+=(double amount);
    KahanSummation& operator=(double reset_to);
    KahanSummation& operator=(const KahanSummation&reset_to);
    double current_total() const;
  };
  
  class KahanVector
  {
  protected:
    vector<KahanSummation> values;
    mutable double dot_self_cache;
    void onModified();
  public:
    KahanVector(size_t length);
    KahanVector();
    KahanVector(const vector<double>&prime);
    KahanVector& operator=(const KahanVector&other);
    KahanVector& operator+=(const vector<double>&other);
    KahanVector operator -(const KahanVector&other) const;
    double operator*(const vector<float> other) const;
    operator vector<double>() const;
    const KahanSummation&operator[](size_t idx) const;
    double dot_self() const;
    size_t size() const;
  };
  
  void kahan_test();
}

#endif
