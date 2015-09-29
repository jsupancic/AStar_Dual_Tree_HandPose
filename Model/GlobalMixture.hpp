/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_GLOBAL_MIXTURE
#define DD_GLOBAL_MIXTURE

#define use_speed_ 0
#include <cv.h>

#include "Detector.hpp"
#include "FauxLearner.hpp"
#include "Platt.hpp"

#include "vec.hpp"
#include "FeatInterp.hpp"

namespace deformable_depth
{  
  /**
   * This class implements a supervised mixture (local or global)
   * 	which can be used within a (joint) Tree or Star part based model. It also
   * 	has legacy support for independent training.
   **/
  class SupervisedMixtureModel : public SettableLDAModel
  {
  public:
    typedef OneFeatureModel ModelType;    
    typedef vector<shared_ptr<MetaData> > TrainingSet;
  public:
    SupervisedMixtureModel(string part_name = "HandBB");
    DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    DetectionSet detect_mixture(const ImRGBZ&im,DetectionFilter filter, int mixture) const;
    void train(vector<shared_ptr<MetaData> >&train_files,TrainParams train_params);
    void prime(vector<shared_ptr<MetaData> >&train_files,TrainParams train_params);
    Mat show(const string&title);   
    map<string,shared_ptr<ModelType> >&get_models();
    map<string,TrainingSet>& get_training_sets();
    string ith_pose(int i) const;
    int    mixture_id(string mixture_name) const;
    virtual SparseVector extractPos(MetaData&metadata,AnnotationBoundingBox bb) const;
    virtual LDA&getLearner();
    virtual void setLearner(LDA*lda);
    virtual void update_model();
    virtual Mat vis_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;
  protected:
    void train_collect_sets(std::vector< shared_ptr<MetaData> >& train_files);
    void train_templ_sizes();
    void train_commit_sets(std::vector< shared_ptr<MetaData> >& train_files,TrainParams&train_params);
    void train_create_subordinates(std::vector< shared_ptr<MetaData> >& train_files,TrainParams&train_params);
    void train_logist_platt(Model::TrainParams train_params);
    void train_joint_interp();
  private:
    vector<shared_ptr<MetaData > > all_train_files;
    map<string, shared_ptr<LogisticPlatt> > regressers;
    map<string, vector<shared_ptr<MetaData> > > training_sets;
    map<string,shared_ptr<ModelType> > models;
    map<string,Size> TSizes;
    string part_name;
    // used with the mixture is part of a joint model
    shared_ptr<LDA> learner;
    shared_ptr<FeatureInterpretation> joint_interp;
    // used for training template sizes
    map<string, vector<float> > widths;
    map<string, vector<float> > heights;
    map<string, vector<float> > aspects;
    
    friend void read(const FileNode&, shared_ptr<SupervisedMixtureModel>&, 
	    shared_ptr<SupervisedMixtureModel>);
    friend void write(cv::FileStorage&, std::string&, const SupervisedMixtureModel&);
  };
  
  class SupervisedMixtureModel_Builder : public Model_Builder
  {
  private:
  public:
    virtual Model* build(Size gtSize,Size imSize) const;
    virtual string name() const;    
  };
  
  void read(const FileNode&, shared_ptr<SupervisedMixtureModel>&, 
	    shared_ptr<SupervisedMixtureModel>);
  void write(cv::FileStorage&, std::string&, const SupervisedMixtureModel&);
  void write(cv::FileStorage&, std::string&, const shared_ptr<SupervisedMixtureModel>&);
}

#endif

