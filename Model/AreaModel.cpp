/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "AreaModel.hpp"
#include "params.hpp"
#include "OcclusionReasoning.hpp"

namespace deformable_depth
{
  AreaModel::AreaModel()
  {
  }
  
  AreaModel::~AreaModel()
  {
  }
  
  void AreaModel::train(vector< shared_ptr< MetaData > >& training_set, TrainParams train_params)
  {
    TaskBlock train_each_area("Model_RigidTemplate::train_areas");
    for(int iter = 0; iter < training_set.size(); iter++)
    {
      train_each_area.add_callee([&,iter]()
      {
	if(!training_set[iter]->use_positives())
	  return;
	  
	map<string,AnnotationBoundingBox> posBBs = training_set[iter]->get_positives();
	posBBs = train_params.part_subset?train_params.part_subset(posBBs):posBBs;
	if(posBBs.empty())
	  return;
	assert(train_params.part_subset);
	shared_ptr<const ImRGBZ> im = training_set[iter]->load_im();
	for(pair<string,AnnotationBoundingBox> posEx : posBBs)
	{
	  if(posEx.second.visible < 1)
	    continue;
	  
	  // get the depths
	  //float z = medianApx(im->Z,posEx.second,.05);
	  //vector<float> zs{medianApx(im->Z,posEx.second,.25)}; 
	  vector<float> zs = manifoldFn_prng(*im,posEx.second);
	  vector<float> world_areas(zs.size());
	  std::transform(zs.begin(),zs.end(),world_areas.begin(),[&](float z)
	  {
	    return im->camera.worldAreaForImageArea(z,posEx.second);
	  });
	  
	  {
	    static mutex m; unique_lock<mutex> l(m);
	    bb_areas.insert(bb_areas.end(),world_areas.begin(),world_areas.end());
	    //log_file << "world_area: " << world_area << endl;
	    //log_file << "bb_area: " << posEx.second.area() << endl;
	    //log_file << "z: " << z << endl;
	  }
	}	
      });
    }
    train_each_area.execute();
    
    // compute the average and print the list
    mean_bb_area = 0;
    //log_file << "bb_areas: ";
    for(float bb_area : bb_areas)
    {
      //log_file << bb_area << " ";
      mean_bb_area += bb_area;
    }
    //log_file << endl;
    mean_bb_area /= bb_areas.size();
    
    sort(bb_areas.begin(),bb_areas.end());
    string areas_string = toString(bb_areas);
    log_once(printfpp("areas: %s",areas_string.c_str()));
    log_file << "Model_RigidTemplate::train_areas complete" << endl;
  }
  
  DetectionSet AreaModel::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    return DetectionSet{};
  }

  const vector< float >& AreaModel::getBBAreas() const
  {
    return bb_areas;
  }

  double AreaModel::meanArea() const
  {
    return mean_bb_area;
  }

  void read(const FileNode&fn, AreaModel&model, AreaModel)
  {
    fn["bb_areas"] >> model.bb_areas;
    fn["mean_bb_area"] >> model.mean_bb_area;
  }

  Mat AreaModel::show(const string& title)
  {
    return Mat();
  }

  void write(FileStorage&fs, string&, const AreaModel&model)
  {
    fs << "{";
    fs << "bb_areas" << model.bb_areas;
    fs << "mean_bb_area" << model.mean_bb_area;
    fs << "}";
  }
  
  void AreaModel::validRange(double world_area_variance,bool testing_mode,
			       float& min_world_area, float& max_world_area) const
  {
    // make it sparse by skiping things which are clearly out of scale
    const vector<float>&bb_areas = getBBAreas();
    min_world_area = bb_areas[clamp<int>(0,.33*bb_areas.size(),bb_areas.size()-1)];
    max_world_area = bb_areas[clamp<int>(0,.66*bb_areas.size(),bb_areas.size()-1)];
    // add a little wiggle room to prevent overfitting.
    // A' = s*A = s*l^2 
    // we are scaling side lengths by sqrt(s)
    // 2/3 to 3/2 works great on CreativeCam data but fails on 
    //     Xtion data. 
    // nope, 1/2 may be needed for some examples (Xtion data?)
    // Tuning notes AFTER bugfixes (2013.9.29)
    // 1.05 is to low
    // 1.2 seems to work, but hurts finger stuff
    //log_once(printfpp("world_area_variance = %f",world_area_variance));
    float acting_world_variance = params::world_area_variance;
    if(world_area_variance != params::world_area_variance)
      log_once("warning: world_area_variance != params::world_area_variance");
    if(testing_mode)
    {
      log_once("entered testing mode");
      min_world_area *= 1.0/acting_world_variance;//4.0/5.0;
      max_world_area *= acting_world_variance/1.0;//1.25;
    }
    else
    {
      log_once("entered training mode");
      min_world_area *= 1.0/acting_world_variance;//4.0/5.0;
      max_world_area *= acting_world_variance/1.0;//1.25; 
    }
    log_once(printfpp("min_world_area = %f",min_world_area));
    log_once(printfpp("max_world_area = %f",max_world_area));
  }
}
