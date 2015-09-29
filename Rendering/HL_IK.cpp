/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "HL_IK.hpp"
#include "Eval.hpp"
#include "ScanningWindow.hpp"
#include "Cache.hpp"

namespace deformable_depth
{
  static shared_ptr<MetaData> do_closest_exemplar_ik(MetaData&metadata,double&min_dist)
  {
    min_dist = inf;
    auto gt_poss = metadata.get_positives();
    vector<shared_ptr<MetaData> > training_poses = default_train_data();
    shared_ptr<MetaData> closestExemplar;    
    TaskBlock find_closest_exemplar("find_closest_exemplar");
    for(auto && training_example : training_poses)
    {
      if(!training_example->use_positives() || !training_example->loaded())
	continue;
      
      find_closest_exemplar.add_callee(
	[&,training_example]()
	{
	  auto ex_poss = training_example->get_positives();
	  double dist = Poselet::min_dist(gt_poss,ex_poss,false).min_ssd;
	  if(dist < min_dist)
	  {
	    static mutex m; lock_guard<mutex> l(m);
	    if(dist < min_dist)
	    {
	      min_dist = dist;
	      closestExemplar = training_example;
	    }
	  }
	});
    }
    find_closest_exemplar.execute();
    return closestExemplar;
  }

  IKSyntherError do_synther_error(MetaData&metadata)
  {
    IKSyntherError ikerr;

    // find the training example with closest pose to the testing example
    double min_dist;
    shared_ptr<MetaData> oracle_datum = do_closest_exemplar_ik(metadata,min_dist);

    // calc the pose error
    ikerr.pose_error = min_dist;

    // get the bbs
    vector<AnnotationBoundingBox> test_bbs  = metric_positive(metadata);    
    vector<AnnotationBoundingBox> train_bbs = metric_positive(*oracle_datum);
    if(test_bbs.size() > 0 and train_bbs.size() > 0)
    {
      // calc the template error.    
      AnnotationBoundingBox test_bb = test_bbs.front();
      ImRGBZ test_im = (*metadata.load_im())(test_bb);
      VolumetricTemplate test_templ(test_im,test_bb.depth,nullptr,RotatedRect());
      //
      AnnotationBoundingBox train_bb = train_bbs.front();
      ImRGBZ train_im = (*oracle_datum->load_im())(train_bb);
      VolumetricTemplate train_templ(train_im,train_bb.depth,nullptr,RotatedRect());
      //
      ikerr.template_error = 1 - test_templ.cor(train_templ);

      log_im_decay_freq("do_synther_error",[&]()
			{
			  Mat vis_test = imageeq("",test_templ.getTIm(),false,false);
			  Mat vis_train = imageeq("",train_templ.getTIm(),false,false);
			  
			  return horizCat(vis_test,vis_train);
			});
    }
    else
      ikerr.template_error = 1.0;

    return ikerr;
  }
  

  // returns how well 
  IKSyntherError synther_error(MetaData&metadata)
  {
    static Cache<IKSyntherError> error_cache;
    
    return error_cache.get(
      metadata.get_filename(),
      [&](){
	return do_synther_error(metadata);
      });
  }

  shared_ptr<MetaData> closest_exemplar_ik(MetaData&metadata)
  {
    double min_dist;
    return do_closest_exemplar_ik(metadata,min_dist);
  }
}

