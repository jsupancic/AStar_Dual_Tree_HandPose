/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#ifndef DD_PCA_POSE
#define DD_PCA_POSE

#include <MetaData.hpp>
#include "Probability.hpp"

namespace deformable_depth
{
  class PCAPose
  {
  public:
    static constexpr int N_DIM = 2;
    static constexpr double PROJ_Q_MAX = 50.0;
    
  protected:
    cv::PCA pca;
    Mat mu, sigma, sigmaInv;
    vector<double> latent_mins;
    vector<double> latent_maxes;
    vector<EmpiricalDistribution> ecdfs;
    
    double latentMin(int dimension) const;
    double latentMax(int dimension) const;    
    Mat vectorize_metadata(const shared_ptr<MetaData>&datum) const;
    set<string>& compute_part_names(const shared_ptr<MetaData>&datum) const;
    Mat project(const shared_ptr<MetaData>&datum) const;
    map<string/*part name*/,Detection/*parts*/ > unproject(
      const Mat&pca_feat,DetectorResult root_det,const ImRGBZ&im);
    
    
  public:
    PCAPose();
    void train(vector<shared_ptr<MetaData> > training_set);
    Mat project_q(const shared_ptr<MetaData>&datum) const;
    map<string/*part name*/,Detection/*parts*/ > unproject_q(
      const Mat&pca_feat,DetectorResult root_det,const ImRGBZ&im);
    void vis_dim(vector<shared_ptr<MetaData> >&data,int dimension);
  };
  
  void pca_pose_train();
}

#endif
