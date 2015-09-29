/**
 * Copyright 2013: James Steven Supancic III
 **/

#if !defined(DD_MODEL) && !defined(WIN32)
#define DD_MODEL

#define use_speed_ 0
#include <cv.h>
#include <boost/concept_check.hpp>

#include <functional>
#include <Detection.hpp>

#include "QP.hpp"
#include "DepthFeatures.hpp"
#include "MetaData.hpp"
#include "util.hpp"
#include "FeatPyr.hpp"
#include "Visualization.hpp"

namespace deformable_depth
{
  class Model_Builder;
  class MetaData;
    
  class Model
  {
  public:
    struct TrainParams
    {
      int positive_iterations;
      int negative_iterations;
      // additional files from which to extract only negatives (used in GMM training)
      vector<shared_ptr<MetaData>> negatives_only;
      TrainParams(int p, int n) : positive_iterations(p), negative_iterations(n){};
      TrainParams() : positive_iterations(1), negative_iterations(1){};
      TrainingBBSelectFn part_subset;
      string subset_cache_key; // used to identify the cache location
      // used to configure mixtures in a mixture model
      shared_ptr<Model_Builder> subordinate_builder; 
    };    
  public:
    // pure virtual methods
    virtual DetectionSet detect(const ImRGBZ&im,DetectionFilter filter) const = 0;
    virtual void train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams()) = 0;
    virtual Mat show(const string&title = "Model") = 0;

    // has default implementations
    virtual void train_on_test(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params = TrainParams());
    virtual Mat vis_result(const ImRGBZ&,Mat&background,DetectionSet&dets) const;
    virtual Visualization visualize_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const;
    virtual ~Model(){};
    virtual double min_area() const {return 1;};
    virtual bool is_part_model() const {return false;};
    virtual bool write(FileStorage&fs) {return false;};
  };  
}

#include "Training.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
  // forward declare
  class MetaData;
  string printfpp(const char*format,...);
  
  struct BadBoundingBoxException : std::exception
  {
  };
            
  class Model_Builder
  {
  public:
    virtual Model* build(Size gtSize,Size imSize) const = 0;
    virtual string name() const = 0;
    virtual ~Model_Builder() {};
  };

  template<typename T>
  class TrivialModelBuilder : public Model_Builder
  {
  public:
    virtual Model* build(Size gtSize,Size imSize) const
    {
      return new T;
    }
    virtual string name() const
    { return "Trivial";};
    virtual ~TrivialModelBuilder() {};
  };
}

#endif
