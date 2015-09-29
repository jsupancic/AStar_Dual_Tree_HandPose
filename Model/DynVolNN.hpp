/**
 * Copyright 2014: James Steven Supancic III
 **/
#if !defined(DD_DYN_VOL_NN) && !defined(WIN32)
#define DD_DYN_VOL_NN

#include "Model.hpp"
#include "HeuristicTemplates.hpp"
#include "SphericalVolumes.hpp"

namespace deformable_depth
{
  struct DynVolTempl
  {
    Mat t;
    float z_min;
    float z_max;
  };
  
  class DynVolNN : public Model
  {
  protected:
    vector<DynVolTempl> templates;
    float min_z, max_z;
    mutable mutex monitor;
    mutable map<int,long> performance_times;
    mutable map<int,long> performance_counts;

    void log_times() const;
    
  public:
    // ctor
    DynVolNN();
    virtual ~DynVolNN();
    // model virtuals
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const override;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) override;
    virtual void train_on_test(vector<shared_ptr<MetaData>>&training_set,
			       TrainParams train_params = TrainParams()) override;
    virtual Mat show(const string&title = "Model") override;
  };  
}

#endif
