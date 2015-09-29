/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_GENERAL_HOUGH_TRANSFORM
#define DD_GENERAL_HOUGH_TRANSFORM

#include <functional>
#include <boost/multi_array.hpp>
#include "Detection.hpp"
#include <memory>
#include "PCA_Pose.hpp"
#include "Probability.hpp"
#include <sstream>
#include "Detection.hpp"

namespace deformable_depth
{
  using std::function;
  
  // represents a vote into a Hough space
  struct HoughVote
  {
    Mat 
      mu_true, mu_false, cov_true, cov_false, sigmaInv_true, sigmaInv_false;
    Mat voting_kernel_true, voting_kernel_false;    
    
    void update_votes(double cov_add_eye_weight = 0);
    Mat log_kernels() const;
    const Mat&cov(bool prediction) const;
    const Mat&mu(bool prediction) const;
    const Mat&voting_kernel(bool prediction) const;
    Gaussian marginalize(bool prediction, int dim) const;
    
    string message;
  };
  void read(const cv::FileNode&, deformable_depth::HoughVote&, deformable_depth::HoughVote);
  void write(cv::FileStorage&, std::string&, const deformable_depth::HoughVote&);  
    
  class HoughLikelihoods;
  
  struct HoughOutputSpace 
  {
  public:
    static constexpr int LATENT_SPACE_DIM = 8;
    
    class Matrix3D : public vector<cv::Mat> 
    {
    public:
      Matrix3D();
      Matrix3D(const vector<cv::Mat>&copy);
      Matrix3D(size_t N, cv::Mat init);
      double&at(int x, int y, int z);
      double at(int x, int y, int z) const;
      int xSize() const;
      int ySize() const; 
      int zSize() const;
    };
    
    HoughOutputSpace();
    virtual Matrix3D likelihood_ratio(const Matrix3D&active_positive, const Matrix3D&active_negative) const;
    virtual shared_ptr<HoughLikelihoods> likelihood_ratio(const ImRGBZ& im) const;
    virtual double vote(
      bool prediction,
      double correct_ratio,
      Point3d center_pos, 
      Point3d center_neg,
      Mat&voting_kernel,
      const PCAPose&pose,
      const HoughVote&vote_pose);
    
    // the observed space
    Matrix3D positive;
    Matrix3D negative;
    
    int xSize() const;
    int ySize() const;
    
    double max_confidence() const;
    static double NCC2(const Matrix3D&m1,const Matrix3D&m2);
    static double spearman_binary_delta
      (const Matrix3D&ground_truth,const Matrix3D&delta_pos, const Matrix3D&delta_neg);
  };
  // functions for dense responce maps
  Mat max_z(HoughOutputSpace::Matrix3D&space,Mat&indexes,Mat&lconf);
    
  struct LatentHoughOutputSpace : public HoughOutputSpace
  {
  public:
    // the independent observed X latent spaces
    vector<Matrix3D> lht_positives;
    vector<Matrix3D> lht_negatives;
    PCAPose pcaPose;
    
    LatentHoughOutputSpace(PCAPose pcaPose);
    virtual shared_ptr<HoughLikelihoods> likelihood_ratio(const ImRGBZ& im) const;
    virtual double vote(
      bool prediction,
      double correct_ratio,
      Point3d center_pos, 
      Point3d center_neg,
      Mat&voting_kernel,
      const PCAPose&pose,
      const HoughVote&vote_pose);  
  };
  
  class HoughLikelihoods
  {
  protected:
    HoughOutputSpace::Matrix3D observed_likelhoods;
    Mat flat_resps;
    
    Point3d center(DetectorResult& win, shared_ptr<const ImRGBZ>& im) const;
    
  public:
    virtual void read_detection(DetectorResult& win, shared_ptr<const ImRGBZ>& im);
    
    friend HoughOutputSpace;
    friend LatentHoughOutputSpace;
  };    
  
  class LatentHoughLikelihoods : public HoughLikelihoods
  {
  protected:
    vector<Mat> latent_estimates;
    PCAPose pcaPose;
    
  public:
    virtual void read_detection(DetectorResult& win, shared_ptr<const ImRGBZ>& im);
    
    friend LatentHoughOutputSpace;
  };      
}

#include "RandomHoughFeature.hpp"

#endif
