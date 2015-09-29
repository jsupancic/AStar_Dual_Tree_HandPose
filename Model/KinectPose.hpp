/**
 * Copyright 2014: James Steven Supancic III
 **/ 
#ifndef DD_KINECT_POSE
#define DD_KINECT_POSE

#include "Model.hpp"
#include "HoughForest.hpp"
#include "boost/multi_array.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: Specific forest based systems
  ///
  class KinectPoseModel : public Model
  {
  private:   
  protected:
    bool is_left; // trained on left hands...
    HoughForest detection_forest;    

    struct Prediction
    {
      Mat map;
      DetectorResult det;
    };

    // methods
    virtual Prediction predict(const Mat&seg,const Mat&Z,const CustomCamera&camera) const = 0;
    Prediction predict_un_in_plane_rot(
      const Mat&seg,const Mat&Z,const CustomCamera&camera,double theta) const;

  public:
    // pure virtual methods
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;   
    virtual Mat show(const string&title = "Model");
    virtual ~KinectPoseModel();
  };  

  typedef boost::multi_array<shared_ptr<atomic<long> >, 3> AtomicArray3D; // x , y , clasif

  class KeskinsModel : public KinectPoseModel
  {
  protected:
    // data
    vector<StochasticExtremelyRandomTree> forest;
    int next_color = 0;
    unordered_map<Vec3b,int> discrete_colors;
    unordered_map<int,Vec3b> colors_discrete;

    // methods
    virtual Prediction predict(const Mat&seg,const Mat&Z,const CustomCamera&camera) const override;
    Mat vis_map(const Mat&seg,const Mat&Z,AtomicArray3D&posteriors) const;

  public:
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
  };

  class XusModel : public KinectPoseModel
  {
  protected:
    map<string,HoughForest> hough_forests;

    // methods
    virtual Prediction predict(const Mat&seg,const Mat&Z,const CustomCamera&camera) const override;    

  public:
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
  };

  DetectorResult predict_joints_mean_shift(const Mat&Z,const Mat&probImage,const CustomCamera&camera,
    double metric_side_length_cm = 5);
}

#endif

