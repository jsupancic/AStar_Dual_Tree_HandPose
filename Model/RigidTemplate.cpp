/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "RigidTemplate.hpp"
#include "util.hpp"
#include "Log.hpp"
#include "FauxLearner.hpp"
#include "GlobalFeat.hpp"
#include "GlobalFeat.hpp"
#include "OcclusionReasoning.hpp"

namespace deformable_depth 
{
  static int n_comp(double area, double width, double height)
  {
    return clamp<int>(1,::sqrt(area*width/height)+.5,500);
  }
  
 /// SECTION: Model_RigidTemplate
  Model_RigidTemplate::Model_RigidTemplate(Size gtSize, Size ISize,
					   double C, double area,
					   shared_ptr<LDA> learner) :
    area(area),
    C(C),
    nx((int)n_comp(area,gtSize.width,gtSize.height)),
    ny((int)n_comp(area,gtSize.height,gtSize.width)),
    ISize(ISize),
    TSize(s_cell*nx,s_cell*ny),
    learner(learner)
  {
    printf("nx = %d ny = %d\n",nx,ny);
    
    setMinArea(default_minArea);
    if(!this->learner)
      this->learner = shared_ptr<LDA>(new QP(this->C));
  }
  
  double Model_RigidTemplate::min_area() const
  {
      return minArea;
  }
  
  int Model_RigidTemplate::getNX() const
  {	
    return nx;
  }

  int Model_RigidTemplate::getNY() const
  {
    return ny;
  }
  
  void Model_RigidTemplate::setMinArea(double newMinArea)
  {
   double cannonical_area = nx*ny*s_cell*s_cell; 
    
    minArea = std::max(cannonical_area/4,newMinArea);
  }
  
  Model_RigidTemplate::~Model_RigidTemplate()
  {
  }
      
  // log_b(a)
  static double log_ab(double b, double a)
  {
    return ::log(a)/::log(b);
  }

  // choose a sf to match the template area to $area
  double getScaleId(double area,double scale_base, double tarea) 
  {
    //printf("area = %f, scale_base = %f\n",area,scale_base);
    double sf = ::sqrt(tarea/area);
    //printf("getScaleId: sf = %f\n",(float)sf);
    double index = log_ab(scale_base,sf);
    
    // log the scales we use
    // bad: area: 64.000000 appox: 1.000000 sf: 1.000000 index: 64.000000 tarea: 0.000000
    // area: 64.000000 sf_appox: 1.000000 sf: 1.000000 index: 0.000000 tarea: 64.000000
    //log_once(printfpp("getScaleId: area: %f sf_appox: %f sf: %f index: %f tarea: %f nx = %f ny = %f s_cell = %f",
      //area, ::pow(scale_base,index), sf, (double)index, tarea,(double) nx, (double)ny, (double)s_cell));
    
    return index;
  }
  
  vector<double> getScaleFactors(
    double minSize, double maxSize, double tarea, double scale_base) 
  {
    log_once(printfpp("Model_RigidTemplate::min_area = %f",(double)minSize));
    // scales given by scale_base^iter
    // e.g. 1.2.^[-2 -1 0 1 2 3 4 5 6]
    // gives   0.6944    0.8333    1.0000    1.2000    1.4400    
    // 1.7280    2.0736    2.4883    2.9860
    //double scale_base = 1.25;
    //double scale_base = 1.1;
    int start = getScaleId(maxSize,scale_base,tarea);
    int end   = getScaleId(minSize,scale_base,tarea);
    //int start = getScaleId(areaModel.meanArea()*1.5,scale_base);
    //int end   = getScaleId(areaModel.meanArea()/1.5,scale_base);
    
    //printf("Detecting minScale = %f maxScale = %f\n",
	//   ::pow(scale_base,start),::pow(scale_base,end));    
    
    vector<double> sfs;
    for(int scaleId = start; scaleId <= end; scaleId++)
    {
      double sf = ::pow(scale_base,scaleId);
      sfs.push_back(sf);
    }
    
    log_once(printfpp("Model_RigidTemplate::getScales Using %d scales",(int)sfs.size()));
    return sfs;
  }
    
  // generate a proper resp image
  void Model_RigidTemplate::log_resp_image(
    const ImRGBZ& im, 
    DetectionFilter filter,
    DetectionSet&dets) const
  {
    Mat vis(im.rows(),im.cols(),DataType<float>::type,Scalar::all(-inf));
    
    for(DetectorResult&det : dets)
    {
      Point2d loc = rectCenter(det->BB);
      if(loc.x <0 || loc.y < 0 || loc.x >= im.cols() || loc.y >= im.rows())
	continue;
      
      float&cur_value = vis.at<float>(loc.y,loc.x);
      cur_value = std::max<float>(cur_value,det->resp);
    }
    
    log_im("resp_image",horizCat(im.RGB,imageeq("",vis,false,false)));
  }
    
  map<FeatPyr_Key,DetectionSet> Model_RigidTemplate::detect_compute(
    const ImRGBZ& im, DetectionFilter filter) const
  {
    log_once(printfpp("min_area = %f",minArea));
    double tarea = nx*ny*s_cell*s_cell;
    vector<double> scale_factors = 
      im.camera.is_orhographic()?
	getScaleFactors(.80*areaModel.getBBAreas().front(),1.20*areaModel.getBBAreas().back(),tarea,1.1):
	getScaleFactors(minArea,im.RGB.size().area(),tarea);
    // compute the detections in parallel over scales
    map<FeatPyr_Key,DetectionSet> results_by_scale;
    bool overlap_windows = (nx*ny <= 4);
    bool computed = false;
    log_once(printfpp("overlap_windows = %d",(int)overlap_windows));
    for(int tx = 0; tx <= (overlap_windows?s_cell:0); tx += s_cell)
      for(int ty = 0; ty <= (overlap_windows?s_cell:0); ty += s_cell)
      {
	for(int sfIter = 0; sfIter < scale_factors.size(); sfIter++)
	{   
	  //printf("\rdetecting sf = %f",sf); fflush(stdout);   
	  double sf = scale_factors[sfIter]; assert(sf < 4);
	  FeatPyr_Key fpKey(sf,tx,ty);
	  if(filter.feat_pyr == nullptr)
	    filter.feat_pyr = shared_ptr<FeatPyr>(new FeatPyr(im));
	  assert(filter.feat_pyr); // make sure pyr is valid.
	  if(filter.feat_pyr->addIm(im,fpKey))
	  {
	    DetectionSet results_for_scale = detect_at_scale(
		filter.feat_pyr->getIm(fpKey),
		filter,fpKey);
	    for(int iter = 0; iter < results_for_scale.size(); ++iter)
	      assert(results_for_scale[iter] != nullptr);
	    // critical section
	    static mutex m; unique_lock<mutex> l(m);
	    results_by_scale[fpKey] = results_for_scale;
	    log_once(printfpp("%d dets from scale %f",(int)results_for_scale.size(),sf));
	  }
	}
	
	// debug assertion
	if(!overlap_windows)
	  assert(computed == false);
	computed = true;	
      }
	
    return results_by_scale;
  }
  
  void Model_RigidTemplate::unPyrBB(Rect_< double >& BB, const FeatPyr_Key& key) const
  {
      // update translations
      BB.x += key.tx;
      BB.y += key.ty;	
      
      // update using scale
      BB.x /= key.scale;
      BB.y /= key.scale;
      BB.width /= key.scale;
      BB.height /= key.scale;
  }
  
  DetectionSet Model_RigidTemplate::detect_collect(
    const ImRGBZ& im, DetectionFilter filter, map<FeatPyr_Key,DetectionSet>&results_by_scale) const
  {
    // loop over scales  to collect the results in serial
    DetectionSet results;
    for(auto sub_dets : results_by_scale)
    {
      //printf("\rcollecting sf = %f",sf); fflush(stdout);      
      double sf = sub_dets.first.scale;
      DetectionSet results_for_scale = results_by_scale[sub_dets.first];
      
      // correct bounding boxes
      for(int rIter = 0; rIter < results_for_scale.size(); rIter++)
      {
	// get the detection object
	assert(results_for_scale[rIter] != nullptr);	
	Detection&cur_det = *results_for_scale[rIter];
	
	unPyrBB(cur_det.BB,sub_dets.first);
	
	// apply the global features
	if(GlobalFeature::length > 0)
	{
	  vector<double> w = learner->getW();
	  vector<double> glbl_feat = GlobalFeature::calculate(im,cur_det.BB,nx,ny);
	  for(int wIter = w.size() - GlobalFeature::length, gblIter = 0; 
	      wIter < w.size(); ++wIter, ++gblIter)
	    cur_det.resp += w[wIter] * glbl_feat[gblIter];
	  auto root_feature = cur_det.feature;
	  cur_det.feature = 
	    [root_feature,glbl_feat]() -> SparseVector 
	  {
	    SparseVector feat = root_feature();
	    SparseVector sparce_glbl(glbl_feat);
	    feat.push_back(sparce_glbl);
	    return feat;
	  };
	}
      }
      
      results.insert(results.end(),results_for_scale.begin(),results_for_scale.end());
    }
    //printf("\n");    
    
    return results;
  }
  
  void Model_RigidTemplate::detect_log(
    const ImRGBZ& im, DetectionFilter filter, DetectionSet&results) const
  {
    if(filter.verbose_log)
      log_resp_image(im,filter,results);
    
    // sort
    filter.apply(results);
    
    if(filter.verbose_log)
    {
      float world_area = im.camera.worldAreaForImageArea(results[0]->depth,results[0]->BB);
      static mutex m;
      unique_lock<mutex> l(m);
      log_file << printfpp("det depth = %f",results[0]->depth) << endl;
      log_file << printfpp("%s world area: %f",im.filename.c_str(),world_area) << endl;
      log_file << printfpp("camera res: %d, %d fov: %f %f",
			   (int)im.rows(),(int)im.cols(),
			   (double)im.camera.vFov(),(double)im.camera.hFov()) << endl;
    }    
  }
    
  DetectionSet Model_RigidTemplate::detect(const ImRGBZ& im, DetectionFilter filter) const
  {
    if(filter.verbose_log)
    {
      filter.thresh = -inf;
      filter.nmax = numeric_limits<decltype(filter.nmax)>::max();
    }
    
    // compute
    map<FeatPyr_Key,DetectionSet> dets_by_scale = detect_compute(im, filter);
    
    // debug
    DetectionSet dets = detect_collect(im, filter, dets_by_scale);    
    
    // log
    detect_log(im, filter, dets);     
    
    return dets;
  }
  
  void Model_RigidTemplate::setLearner(LDA* lda)
  {
    learner.reset(lda);
  }
    
  LDA& Model_RigidTemplate::getLearner()
  {
    assert(learner);
    return *learner;
  }
    
  void Model_RigidTemplate::train_areas(
    vector< shared_ptr< MetaData > >& training_set,TrainParams train_params)
  {
    areaModel.train(training_set,train_params);
  }
    
  void Model_RigidTemplate::train(
    std::vector< std::shared_ptr< deformable_depth::MetaData > >& training_set, deformable_depth::Model::TrainParams train_params)
  {
    // sort the training files to ensure
    // (1) same cache location
    // (2) same computational results
    std::sort(training_set.begin(),training_set.end(),
	      [](const shared_ptr<MetaData>&v1,const shared_ptr<MetaData>&v2)
	      {return *v1 < *v2;}
	     );    
    
    // try to recover from cache
    auto v_filenames = filenames(training_set);
    sort(v_filenames.begin(),v_filenames.end());
    string cache_file = printfpp("cache/DetectorLDAfiles=%sC=%fAREA=%fSubset=%s.yml",
				 hash(v_filenames).c_str(),C,area,
				 train_params.subset_cache_key.c_str());
    FileStorage cache;
    cache.open(cache_file,FileStorage::READ);
    if(cache.isOpened())
    {
      learner.reset(new FauxLearner());
      cache["learner"] >> (FauxLearner&)*learner;
      cache.release();
      log_file << "Model_RigidTemplate skiped training via cache" << endl;
      return;
    }
    log_file << "RigidTemplateModel cache file not found, re-computing: " << cache_file << endl;
        
    // train the SVM
    training_set = pseudorandom_shuffle(training_set);
    prime(training_set, train_params);
    train_smart(*this,training_set,TrainParams(train_params));
    if(train_params.negatives_only.size() > 0)
    {
      Model::TrainParams neg_params = train_params;
      neg_params.negative_iterations = 1;
      neg_params.positive_iterations = 0;
      train_smart(*this,train_params.negatives_only,neg_params);
    }
    //train_serial(train_files,train_params);
      
    // store to cache
    cache.open(cache_file,FileStorage::WRITE);
    cache << "learner" << FauxLearner(learner->getW(),learner->getB());
    cache.release();
  }
  
  void Model_RigidTemplate::prime(
    vector< shared_ptr< MetaData > >& train_files, Model::TrainParams train_params)
  {
    // for ISO models, determine what scales to search.
    train_areas(train_files,train_params);
  }
    
  void Model_RigidTemplate::update_model()
  {
    learner->opt();
  }  
  
  /// SECTION: Serialization
  void write(FileStorage& fs, string& , const Model_RigidTemplate& model)
  {
    fs << "{";
    fs << "minArea" << model.minArea;
    fs << "area" << model.area;
    fs << "learner" << FauxLearner(model.learner->getW(),model.learner->getB());
    fs << "nx" << model.nx;
    fs << "ny" << model.ny;
    fs << "ISize" << model.ISize;
    fs << "TSize" << model.TSize;
    fs << "areaModel" << model.areaModel;
    fs << "}";
  }
  
  void read(FileNode node, Model_RigidTemplate& model)
  {
    node["minArea"] >> model.minArea;
    node["area"] >> model.area;
    // the learner is bit tricky
    shared_ptr<FauxLearner> lda;
    node["learner"] >> lda;
    model.learner = lda;
    //
    node["nx"] >> model.nx;
    node["ny"] >> model.ny;
    read<int>(node["ISize"],model.ISize);
    read<int>(node["TSize"],model.TSize);
    node["areaModel"] >> model.areaModel;
  }  
}
