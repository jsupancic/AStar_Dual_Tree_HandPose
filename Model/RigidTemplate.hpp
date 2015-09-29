/**
 * Copyright 2013: James Steven Supancic III
 **/

#if !defined(DD_RIGID_TEMPLATE) && !defined(WIN32)
#define DD_RIGID_TEMPLATE

#include "Training.hpp"
#include "MetaData.hpp"
#include "FeatPyr.hpp"
#include "AreaModel.hpp"

namespace deformable_depth
{
  class Model_RigidTemplate : public SettableLDAModel
  {
  public:
    constexpr static double DEFAULT_C = 1;
    constexpr static int s_cell = 2;
    constexpr static double default_minArea = 25*25; // finger 8x8 ;// hand 25*25
    constexpr static double DEFAULT_AREA = 5*5; // 4 for finger, 16 for hand
  private: // hyper-parameters
    // RGB 15*15 > 10*10 > 6*6
    // RGB: .01 > .1
    // RGB: .01 > .001
    double minArea;
    double C;        
  protected:
    float area;
    shared_ptr<LDA> learner;
    int nx, ny;
    Size ISize, TSize;
    AreaModel areaModel;
  public:
    Model_RigidTemplate(Size gtSize,Size ISize,
			double C = DEFAULT_C, 
			double area = DEFAULT_AREA,
			shared_ptr<LDA> learner = shared_ptr<LDA>());
    virtual ~Model_RigidTemplate();
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    void prime(vector<shared_ptr<MetaData> >&train_files,TrainParams train_params);
    virtual Mat show(const string&title) = 0;
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    LDA&getLearner();
    void setLearner(LDA*lda);
    virtual void update_model();
    int getNX() const;
    int getNY() const;
  public:
    virtual Mat resps(const ImRGBZ&im) const = 0;
    void setMinArea(double newMinArea);
    virtual double min_area() const;
  private:
    map<FeatPyr_Key,DetectionSet> detect_compute(
      const ImRGBZ& im, DetectionFilter filter) const;
    DetectionSet detect_collect(
      const ImRGBZ& im, DetectionFilter filter, map<FeatPyr_Key,DetectionSet>&results_by_scale) const;
    void detect_log(const ImRGBZ& im, DetectionFilter filter, DetectionSet&results) const;  
    // gather the areas... useful for isometric detectors mainly.
    void train_areas(vector<shared_ptr<MetaData> > &training_set,TrainParams train_params);
    void log_resp_image(const ImRGBZ&im,DetectionFilter filter,DetectionSet&) const;
  protected:
    Model_RigidTemplate() {};
    virtual DetectionSet detect_at_scale(
      const ImRGBZ&im, DetectionFilter filter,FeatPyr_Key scale) const = 0;
    friend void read(FileNode, Model_RigidTemplate&);
    friend void write(cv::FileStorage&, std::string&, const Model_RigidTemplate&);
    void unPyrBB(Rect_<double>&BB, const FeatPyr_Key&key) const;
  };
  void read(FileNode, Model_RigidTemplate&);
  void write(cv::FileStorage&, std::string&, const Model_RigidTemplate&);  
  
  // useful functions for non-LDA image pyramid methods
  double getScaleId(double area,double scale_base, double tarea);
  vector<double> getScaleFactors(double minSize, double maxSize, double tarea, double base = 1.25);  
}

#endif


