/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_AREA_MODEL
#define DD_AREA_MODEL

#include "Model.hpp"

namespace deformable_depth
{
  class AreaModel : public Model
  {
  protected:
    // areas of bbs, used in isometric detection or subclasses
    vector<float> bb_areas; 
    double mean_bb_area;
    
  public:
    AreaModel();
    ~AreaModel();
    void validRange(double world_area_variance,bool testing_mode,float&min_area,float&max_area) const;
    double meanArea() const;
    const vector<float>&getBBAreas() const;
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    virtual Mat show(const string&title = "Model");
    
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::AreaModel&);
    friend void read(const cv::FileNode&, deformable_depth::AreaModel&, deformable_depth::AreaModel);
  };
}

#endif

