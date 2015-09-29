/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "FLANN.hpp"
#include "Detection.hpp"
#include "ScanningWindow.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;

  typedef uint8_t DescriptorType;

  DetectorResult FLANN_Model::detect_one(const ImRGBZ&im,DetectorResult&window,double theta) const
  {
    // extract the query template         
    RotatedRect bb(rectCenter(window->BB),window->BB.size(),rad2deg(theta));
    ImRGBZ im_roi = im.roi_light(bb);
    shared_ptr<NNTemplateType> X = make_shared<NNTemplateType>(
      im_roi,window->depth,nullptr,bb);
    
    // present it to the FLANN database index
    Mat XVis = X->getTIm();
    if(X->getTIm().empty())
      return nullptr;
    Mat XFlat = XVis.reshape(0,1);
    XFlat.convertTo(XFlat,DataType<DescriptorType>::type);
    Mat indices, dists;
    flann_index->knnSearch(XFlat,indices,dists,1);
    assert(indices.type() == DataType<int>::type);
    double cor = at(dists,0,0);
    size_t idx = indices.at<int>(0);
    
    // lookup the template
    string match_key = index2pose.at(idx);
    const NNTemplateType& TT = extracted_templates.allTemplates.at(match_key);
    auto parts = extracted_templates.parts_per_ex.at(match_key);
	
    // look up the exemplar and use it to regress a detection.
    auto det = make_shared<Detection>();	
    det->BB = window->BB;
    det->depth = window->depth;
    det->resp = cor;
    // generate the parts
    set_det_positions(TT, index2pose.at(idx),X->getExtractedFrom(),parts,det,window->depth);
    return det;
  }

  DetectionSet FLANN_Model::detect(const ImRGBZ&im,DetectionFilter filter)  const
  {
    // get the candidate windows
    filter.manifoldFn = manifoldFn_kmax;
    DetectionSet windows = enumerate_windows(im,filter);
    DetectionSet dets;
    if(windows.size() == 0)
    {
      log_file << "warning: zero candidate windows in " << im.filename << endl;
      return dets;
    }
    else
      log_file << "candidate windows in " << im.filename << endl;
    // do the flann subsampling
    double flann_samples = windows.size();
    double speedup = 1;
    if(g_params.has_key("FLANN_SAMPLES"))
    {      
      flann_samples = fromString<double>(g_params.require("FLANN_SAMPLES"));
      speedup = windows.size() / flann_samples;
      log_file << "FLANN_Model::detect sampling " << flann_samples << " from " << windows.size() << endl;
      windows = DetectionSet(random_sample_w_replacement(windows,flann_samples));
    }

    Timer t;
    for(auto && window : windows)
    {
      for(int iter = 0; iter < ROTS_PER_WINDOW; ++iter)
      {
	double theta = interpolate_linear(iter,0,ROTS_PER_WINDOW,0,2*params::PI);
	auto det = detect_one(im,window,theta);
	if(det)
	  dets.push_back(det);
      }
    }
    
    dets = sort(dets);
    double best_resp = qnan;
    if(dets.size() > 0)
      best_resp = dets[0]->resp;

    double adjusted_time = t.toc() * speedup;
    log_once(safe_printf("% took % milliseconds with FLANN resp = %",im.filename,adjusted_time,best_resp));

    dets = sort(dets);
    return dets;
  }

  void FLANN_Model::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params) 
  {    
    // setup the training set
    vector<shared_ptr<MetaData> >  positive_set;
    vector<shared_ptr<MetaData> >  negtive_set;    
    split_pos_neg(training_set,positive_set,negtive_set);
    //this->training_set = training_set;
    int npos = fromString<int>(g_params.require("NPOS"));
    auto active_set = random_sample<shared_ptr<MetaData> >(positive_set,npos);

    //
    extracted_templates = extract_templates(active_set,train_params);
    log_once(safe_printf("FLANN_Model::train got % templates",extracted_templates.allTemplates.size()));

    // generate a matrix...
    for(auto templ : extracted_templates.allTemplates)
    {
      Mat TVis = templ.second.getTIm();
      Mat Tflat = TVis.reshape(0,1);
      Tflat.convertTo(Tflat,DataType<DescriptorType>::type);
      template_database.push_back(Tflat);
      int alloced_id = index++;
      index2pose[alloced_id] = templ.first;
    }
    log_file << "template_database size = " << template_database.size() << endl;

    // construct a KD Tree.
    cv::flann::KDTreeIndexParams kd_index_params;
    
    // construct a LSH table
    int table_number = fromString<int>(g_params.get_value("FLANN_LSH_TABLE_NUMBER","15"));
    int key_size = fromString<int>(g_params.get_value("FLANN_LSH_KEY_SIZE","15"));
    int multi_probe_level = fromString<int>(g_params.get_value("FLANN_LSH_MULTI_PROBE_LEVEL","2"));
    cv::flann::LshIndexParams lsh_index_params(table_number,key_size,multi_probe_level);

    // build the index
    const cv::flann::IndexParams& index_params = lsh_index_params;
    flann_index = make_shared<cv::flann::Index>(template_database,index_params);
    log_file << "flann index trained" << endl;
  }

  Mat FLANN_Model::show(const string&title) 
  {
    return Mat();
  }
}
