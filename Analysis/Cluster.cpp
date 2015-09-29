/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "Cluster.hpp"
#include "MetaData.hpp"
#include "Eval.hpp"
#include "Poselet.hpp"
#include "ThreadPoolCXX11.hpp"
#include "main.hpp"
#include "Log.hpp"
#include "Annotation.hpp"
#include <boost/filesystem.hpp>

namespace deformable_depth
{
  void normalize_keypoints(vector<Point2d>&keypoints)
  {
    // compute the first moment
    Point2d mean(0,0);
    for(Point2d&kp : keypoints)
      mean += kp;
    mean.x /= keypoints.size();
    mean.y /= keypoints.size();
    // subtrack the mean
    for(Point2d&kp : keypoints)
      kp -= mean;
    
    // Comptue the second moment.
    Point2d m2;
    for(Point2d&kp : keypoints)
      m2 += Point2d(kp.x*kp.x,kp.y*kp.y);
    m2.x /= keypoints.size();
    m2.y /= keypoints.size();
    // divide by the stadandard devaition
    double psudo_sd = (std::sqrt(m2.x) + std::sqrt(m2.y))/2.0;
    for(Point2d&kp : keypoints)
    {
      kp.x /= psudo_sd;
      kp.y /= psudo_sd;
    }  
  }
  
  vector<shared_ptr<MetaData> > load_synthetic_set()
  {
    vector<shared_ptr<MetaData> > set  = metadata_build_all(
      params::synthetic_directory(),true,false);
    random_shuffle(set.begin(),set.end());       
    return set;
  }
  
  Mat vis_cluster(
    vector<shared_ptr<MetaData> >&set,string cluster_name)
  {
    vector<Mat> cluster_members;
    TaskBlock vis_cluster("vis_cluster");
    for(int iter = 0; iter < set.size(); ++iter)
      vis_cluster.add_callee([&,iter,cluster_name]()
      {
	if(cluster_name == set[iter]->get_pose_name())
	{
	  Mat vis1 = show_one(*set[iter]);
	  static mutex m; unique_lock<mutex> l(m);
	  cluster_members.push_back(vis1);
	}
      });
    vis_cluster.execute();
    Mat viz = tileCat(cluster_members);
    log_im(printfpp("cluster_%s_",cluster_name.c_str()),viz);    
    return viz;
  }
  
  void cluster()
  {
    // load the examples
    //vector<shared_ptr<MetaData> > set  = metadata_build_all(default_train_dirs(),false);
    vector<shared_ptr<MetaData> > set = load_synthetic_set();
    cout << "cluster: loaded data set" << endl;
    
    // compute a feature matrix
    Mat keypoint_features;
    for(auto&&example : set)
    {
      vector<Point2d> kps = Poselet::keypoints(*example);
      normalize_keypoints(kps);
      Mat X_kps(1,2*kps.size(),DataType<float>::type);
      for(int iter = 0; iter < kps.size(); ++iter)
      {
	X_kps.at<float>(0,2*iter)   = kps[iter].x;
	X_kps.at<float>(0,2*iter+1) = kps[iter].y;
      }
      keypoint_features.push_back<float>(X_kps);
    }
    cout << "cluster: Extracted keypoint features" << endl;
    
    // pass to k-means
    Mat labels;
    cv::TermCriteria term_crit;
    cout << "Invoking OpenCV Kmeans" << endl;
    // 25 examples per class
    // 25 = E/K, K = E/25
    int K = (g_params.has_key("K"))?fromString<int>(g_params.get_value("K")):set.size()/25;
    kmeans(keypoint_features,K,labels,term_crit,K,KMEANS_PP_CENTERS);
    cout << "OpenCV K-Means is done" << endl;
    
    // export the cluster IDs
    for(int iter = 0; iter < set.size(); ++iter)
    {
      // write the cluster
      shared_ptr<MetaData>&metadata = set[iter];
      int k = labels.at<int>(iter);
      shared_ptr<MetaData_Editable> editable = 
	std::dynamic_pointer_cast<MetaData_Editable>(metadata);
      editable->setPose_name(printfpp("synth_cluster_%d",(int)k));
      
      // re-export the metadata
      editable->export_annotations();
    }
    FileStorage clusters(params::synthetic_directory() + "/cluster_assignment.yaml",
			 FileStorage::WRITE);
    clusters << "assignments" << labels;
    clusters.release();
    
    // visualize the results
    for(int k = 0; k < K; ++k)
    {
      string cluster_name = printfpp("synth_cluster_%d",(int)k);
      vis_cluster(set,cluster_name);
    }
  }
  
  void export_cluster()
  {
    string out_dir = g_params.get_value("out_dir");
    string cluster_name = g_params.get_value("cluster_name");
    vector<shared_ptr<MetaData> > set = load_synthetic_set();
    
    // try create the directory
    boost::filesystem::path dir(out_dir);
    boost::filesystem::create_directory(dir);
    
    // actually write the to the new cluster directory
    int ctr = 0;
    for(auto & metadata : set)
      if(metadata->get_pose_name() == cluster_name)
      {
	shared_ptr<MetaData_Editable> editable = 
	  std::dynamic_pointer_cast<MetaData_Editable>(metadata);
	string out_file = printfpp("%s/synth%d.yaml.gz",out_dir.c_str(),ctr++);
	editable->change_filename(out_file);
      }
      
    // create a nifty visulization
    Mat vis = vis_cluster(set,cluster_name);
    log_im(cluster_name,vis);
    imwrite(params::out_dir() + "/cluster.jpg",vis);
      
    cout << "export complete" << endl;
  }
}

