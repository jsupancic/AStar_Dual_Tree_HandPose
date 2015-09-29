/**
 * Copyright 2014: James Steven Supancic III
 **/

#include <cstdlib>

#ifdef DD_DEEP_CAFFE
 
#include "Export2Caffe.hpp"
#include "MetaDataKITTI.hpp"
#include "util.hpp"
#include "StackedModel.hpp"
#include "Eval.hpp"
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include "ExternalModel.hpp"
#include "ThreadPoolCXX11.hpp"
#include "caffe/proto/caffe.pb.h"

namespace deformable_depth
{
  void export_one_example(
    leveldb::DB&db,
    const ImRGBZ&im,Rect_<double>&bb,float depth,const string&name, bool label)
  {
    // extract the sub-window
    double sf = 1.2;
    Rect bb_prime = rectResize(bb,sf,sf);
    //if(!rectContains(im.Z,bb_prime))
    //{
      //log_once("Skipping Example: " + name);
    //}
    Size export_resolution(40,40);
    const ImRGBZ t = (im)(bb_prime).resize(export_resolution);
    string label_str = label?"pos":"neg";
    if(rand() % 100 == 0)
      log_im(string("2Caffe") + label_str,t.RGB);
    
    // write to caffe
    std::string value;
    caffe::Datum datum;
    
    datum.set_channels(3); // just RGB for now
    datum.set_height(export_resolution.height);
    datum.set_width(export_resolution.width);
    datum.set_label(label);
    // write the "features"
    string* datum_string = datum.mutable_data();
//     for (int c = 0; c < 3; ++c) 
//     {
//       for (int h = 0; h < t.rows() ; ++h) 
//       {
// 	for (int w = 0; w < t.cols(); ++w) 
// 	{
// 	  datum_string->push_back(
//             static_cast<char>(t.RGB.at<cv::Vec3b>(h, w)[c]));
// 	}
//       }
//     }
    // write the depth
    Mat vis(t.rows(),t.cols(),DataType<Vec3b>::type,Scalar::all(0));
    for(int h = 0; h < t.rows(); ++h)
      for(int w = 0; w < t.cols(); ++w)
      {
	char voxel = clamp<char>(0,(t.Z.at<float>(h,w) - depth)/2,127);
	datum_string->push_back(
	  static_cast<char>(voxel)
	);
	vis.at<Vec3b>(h,w) = Vec3b(2*voxel,2*voxel,2*voxel);
      }
    log_im("exported",vis);

    datum.SerializeToString(&value);
    static mutex m; lock_guard<mutex> l(m);
    db.Put(leveldb::WriteOptions(), std::string(name), value);
  }
  
  void export_one_frame(leveldb::DB&db,MetricVolumeModel&subordinate,MetaData&metadatum)
  {
    cout << "exporting for caffe: " << metadatum.get_filename() << endl;
    
    // load the image datum
    shared_ptr<const ImRGBZ> im = metadatum.load_im();
    map<string,AnnotationBoundingBox> pos_bbs = metadatum.get_positives();
    map<string,AnnotationBoundingBox> train_bbs = params::defaultSelectFn()(pos_bbs);
    
    // export the positives
    if(metadatum.use_positives())
      for(auto pos_bb : train_bbs)
      {
	for(AnnotationBoundingBox& metric_bb : 
	  subordinate.metric_positive(metadatum,*im,pos_bb.second))
	  export_one_example(db,*im,metric_bb,metric_bb.depth,im->filename + pos_bb.first,true);
      }
    
    // export the negatives
    if(metadatum.use_negatives())
    {
      DetectionSet dets_for_frame = subordinate.detect(*im,DetectionFilter(-inf,inf));
      for(auto curDet : dets_for_frame)
	if(curDet->is_black_against(pos_bbs,0.25))
	  export_one_example(db,*im,curDet->BB,curDet->depth,curDet->toString(),false);
    }
  }
  
  void export_2_caffe(
    vector<shared_ptr<MetaData> >  metadata, MetricVolumeModel&subordinate,string db_filename)
  {
    // open a leveldb
    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.error_if_exists = true;
    leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
    assert(status.ok());
    
    // write each metadata to the leveldb
    TaskBlock exportFrames("exportframes");
    int counter = 0;
    for(auto datum : metadata)
      exportFrames.add_callee([&,datum]()
      {
	export_one_frame(*db,subordinate,*datum);
	static mutex m; lock_guard<mutex> l(m);
	log_once(printfpp("exported %d to deep caffe",counter++));
      });
    exportFrames.execute();
    cout << "finished exportation, cleaning up" << endl;
    
    // close the leveldb
    delete db;
  }
  
  void export_2_caffe()
  {
    // EXPORT KITTI
//     vector<shared_ptr<MetaData> > training_set = KITTI_default_train_data();
//     shared_ptr<MetaData> prime_ex;
//     Rect BB;
//     seed_data(training_set,BB,prime_ex);    
//     StackedModel baseline_model(BB.size(),prime_ex->load_im()->RGB.size());
//     train_model_one(baseline_model,training_set);
    // three sets of interest
    //export_2_caffe(KITTI_validation_data(),baseline_model,params::out_dir()+"/validation-leveldb");
    //export_2_caffe(KITTI_default_test_data(),baseline_model,params::out_dir()+"/test-leveldb");
    //export_2_caffe(training_set,baseline_model,params::out_dir()+"/training-leveldb");
    
    ApxNN_Model metric_model;
    export_2_caffe(default_train_data(),metric_model,params::out_dir()+"/training-leveldb");
  }
}
#else
namespace deformable_depth
{
  void export_2_caffe()
  {
    std::abort();
  }
}
#endif
