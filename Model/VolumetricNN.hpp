/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Model.hpp"
#include "HeuristicTemplates.hpp"
#include "SphericalVolumes.hpp"

namespace deformable_depth
{
  class TemplateUniScalar
  {
  public:
    shared_ptr<Mat> T;
    shared_ptr<MetaData> datum;
    float theta; // radians
    float depth;

    TemplateUniScalar(shared_ptr<Mat> T, shared_ptr<MetaData> datum, float theta);
  };
 
  struct VolTrainingExample  
  {
    shared_ptr<Mat> t;
    float z_min;
    double area;
    double entropy;

    bool operator< (const VolTrainingExample&o) const
    {
      return entropy < o.entropy;
      //return area < o.area;
    }
  };

  struct FilterResult
  {
    shared_ptr<TemplateUniScalar> filter;    
    Extrema extrema;
    double depth; // detection depth
    double template_depth;
  };

  // represents results from all filters.
  struct FilterResults
  {
    multimap<double,FilterResult > filters_by_resps;
    Mat top_resps;
    Mat top_evidence;
  };
  
  class VolumetricNNModel : public Model
  {
  protected:
    // data
    multimap<double, shared_ptr<TemplateUniScalar> > templates_by_depth;    
    mutex monitor;

    // methods
    FilterResults filterImages(
      DetectionFilter&filter,
      const ImRGBZ&im, SphericalOccupancyMap&SOM,Visualization&vis,
      multimap<double,shared_ptr<TemplateUniScalar> >&templates_by_depth) const;
    void do_training_write(VolTrainingExample&t,const shared_ptr<MetaData>&datum);
    VolTrainingExample do_training_extract(const shared_ptr<MetaData>&datum) const;
    void do_training(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params);
    void load_templates();
    
  public:
    // ctor
    VolumetricNNModel();
    // model virtuals
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual void train_on_test(vector<shared_ptr<MetaData>>&training_set,
			       TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
  };
}

