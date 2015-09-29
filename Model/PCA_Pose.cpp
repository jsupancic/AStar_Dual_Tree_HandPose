/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "PCA_Pose.hpp"
#include <string>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "MetaData.hpp"
#include "util.hpp"
#include "Eval.hpp"
#include "Orthography.hpp"
#include "Cache.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  using params::RESP_ORTHO_X_RES;
  using params::RESP_ORTHO_Y_RES;  
  using params::RESP_ORTHO_Z_RES;
  static Size ortho_res(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES);  
  
  // WARNING : Assumes HandBB is the root
  static string root_part = "HandBB";
  
  vector< double > vectorize(const map< string, AnnotationBoundingBox >& pose)
  {
    vector<double> vec((pose.size()-1) * 2);
    int index = 0;
    
    auto hand_center = rectCenter(pose.at(root_part));
    
    for(auto & pair : pose)
    {
      if(pair.first == root_part)
	continue;
      
      Rect_<double> bb = pair.second;
      Point2d center = rectCenter(bb);
      vec[index++] = center.x - hand_center.x;
      vec[index++] = center.y - hand_center.y;
      //vec[index++] = bb.br().x;
      //vec[index++] = bb.br().y;
      assert(index <= vec.size());
    }
    
    return vec;
  }  
  
  map< string , Detection/*parts*/> PCAPose::unproject(
    const Mat& pca_feat,
    DetectorResult root_det,
    const ImRGBZ&im)
  {
    // finish the backprojection
    Mat feat = pca_feat;
    //cout << "backprojecting: " << feat << endl;
    feat = (pca.backProject(feat) * sigma + mu);
    //cout << "backprojected: " << feat << endl;
    
    // unvectorize
    Point2d rootCenter = rectCenter(root_det->BB);
    double root_z = root_det->getDepth(im);
    Rect_<double> root_ortho = map2ortho(im.camera,ortho_res,
		 root_det->BB, root_z);
    Point2d root_ortho_center = rectCenter(root_ortho);
    //cout << "root_ortho: " << root_ortho << endl;
    map< string , Detection/*parts*/> parts;
    int index = 0;
    assert(feat.type() == DataType<double>::type);
    set<string>& part_names = compute_part_names(nullptr);
    for(const string & part_name : part_names)
    {
      Detection part;
      
      // get the offsets in Orthographic coordinates
      double offset_x = feat.at<double>(index++);
      double offset_y = feat.at<double>(index++);
      
      // unvectorize the current part
      Point2d center(offset_x + root_ortho_center.x,
		     offset_y + root_ortho_center.y);
      part.BB = rectFromCenter(center,Size(3.5,3.5)/*FIXME*/);
      part.BB = mapFromOrtho(im.camera,ortho_res,part.BB,root_z);
      
      parts[part_name] = part;
    }
    require_equal<int>(feat.size().area(),index);
    
    return parts;
  }  
  
  set<string>&PCAPose::compute_part_names(const shared_ptr< MetaData >& datum) const
  {
    static Cache<shared_ptr<set<string> >> part_names_cache;
    shared_ptr<set<string> > part_names = part_names_cache.get("",[&]()
    {
      shared_ptr<set<string> > part_names_computing = make_shared<set<string>>( );
      // first time, insert the part names into the directory
      assert(datum != nullptr);
      auto pposs = datum->get_positives();
      for(auto & pos : pposs)
      {
	if(pos.first == root_part)
	  continue;
	
	log_file << "vectorize_part: " << pos.first << endl;
	part_names_computing->insert(pos.first);
      }
      return part_names_computing;
    });    
    return *part_names;
  }
  
  Mat PCAPose::vectorize_metadata(const shared_ptr<MetaData>&datum) const
  {
    shared_ptr<const MetaData> cDatum = std::dynamic_pointer_cast<const MetaData>(datum);
    shared_ptr<const ImRGBZ> im = cDatum->load_im();

    set<string>& part_names = compute_part_names(datum);
    auto pposs = datum->get_positives();
    // requrie that the set of part names does not change.
    for(auto & pos : pposs)
    {
      // skip the root_part
      if(pos.first == root_part)
	continue;
      
      // check that we already have said part
      if(part_names.find(pos.first) == part_names.end())
      {
	cout << "part_names does not contain: " << pos.first << endl;
	assert(false);
      }
    }
    auto poss = map2ortho(im->camera,ortho_res,pposs,*im);
    Mat v_poss(vectorize(poss));
    return v_poss.t();
  }
    
  void PCAPose::vis_dim(
    vector<shared_ptr<MetaData> >&data,int dimension)
  {
    // sort data by PCA dimension
    std::sort(data.begin(),data.end(),[&]
      (const shared_ptr<MetaData>&v1,const shared_ptr<MetaData>&v2)
    {
      Mat proj1 = project(v1);
      Mat proj2 = project(v2);
      assert(proj1.type() == DataType<double>::type);
      double vd1 = proj1.at<double>(dimension);
      double vd2 = proj2.at<double>(dimension);
      cout << "vd1 = " << vd1 << " vd2 = " << vd2 << endl;
      return vd1 < vd2;
    });
    
    vector<Mat> vis_tile;
    for(auto & datum : data)
    {
      shared_ptr<ImRGBZ> im = datum->load_im();
      vis_tile.push_back(im->RGB);
    }
    log_im(printfpp("pca_pose_dim_%d_",dimension),tileCat(vis_tile));
  }
  
  Mat PCAPose::project(const shared_ptr< MetaData >& datum) const
  {
    //cout << "projecting: " << vectorize_metadata(datum) << endl;
    Mat vp = (vectorize_metadata(datum) - mu)*sigmaInv;
    Mat proj = pca.project(vp);
    //cout << "projected to: " << proj << endl;
    return proj;
  }
  
  PCAPose::PCAPose()
  {
  }
  
  void PCAPose::train(vector< shared_ptr< MetaData > > training_set)
  {
    // convert the pose data into a Matrix
    Mat raw_pose_data;
    for(auto && datum : training_set)
    {
      raw_pose_data.push_back<double>(vectorize_metadata(datum));
    }
    require_equal<int>(raw_pose_data.rows,training_set.size());
    
    // choose the centering parameters
    cv::calcCovarMatrix(raw_pose_data,sigma,mu,CV_COVAR_NORMAL|CV_COVAR_ROWS|CV_COVAR_SCALE);
    sigma = Mat::eye(sigma.rows,sigma.cols,DataType<double>::type); // numeric issues and invertability
    //sigma += 1e-8;
    mu = Mat(1,sigma.cols,DataType<double>::type,Scalar::all(0));
    
    // center the data
    sigmaInv = sigma.inv();
    for(int row = 0; row < raw_pose_data.rows; row++)
    {
      raw_pose_data.row(row) -= mu;
      raw_pose_data.row(row) *= sigmaInv;
    }
    cout << "mu: " << mu << endl;
    cout << "sigma: " << sigma << endl;
    cout << "sigmaInv: " << sigmaInv << endl;
    
    // run PCA on it
    pca(raw_pose_data,noArray(),CV_PCA_DATA_AS_ROW,N_DIM);
    cout << "Mean: " << pca.mean << endl;
    cout << "EigenVectors: " << pca.eigenvectors << endl;
    cout << "EigenValues : " << pca.eigenvalues  << endl;
    
    // visualize for debugging
    auto sample20 = random_sample(training_set,50);
    for(int iter = 0; iter < N_DIM; ++iter)
      vis_dim(sample20,iter);
    
    // compute the latent mins and maxes
    for(int iter = 0; iter < N_DIM; ++iter)
    {
      latent_maxes.push_back(0);
      latent_mins.push_back(0);
      ecdfs.push_back(EmpiricalDistribution());
    }
    log_file << "latent0: ";
    for(int iter = 0; iter < training_set.size(); ++iter)
    {
      Mat p = project(training_set[iter]);
      log_file << p.at<double>(0) << " ";
      for(int jter = 0; jter < N_DIM; ++jter)
      {
	double&max = latent_maxes[jter];
	double&min = latent_mins[jter];
	max = std::max<double>(max,p.at<double>(jter));
	min = std::min<double>(min,p.at<double>(jter));
	ecdfs[jter].data.push_back(p.at<double>(jter));
      }
    }
    log_file << endl;
    for(int iter = 0; iter < N_DIM; ++iter)
    {
      log_file << printfpp("latent_min = %f latent_max = %f",
			   latent_mins[iter],latent_maxes[iter]) << endl;
      std::sort(ecdfs[iter].data.begin(),ecdfs[iter].data.end());
    }    
  }
  
  Mat PCAPose::project_q(const shared_ptr< MetaData >& datum) const
  {
    Mat pfeat = project(datum);
    assert(pfeat.type() == DataType<double>::type);
    for(int iter = 0; iter < N_DIM; ++iter)
      pfeat.at<double>(iter) = PROJ_Q_MAX*ecdfs[iter].cdf(pfeat.at<double>(iter));
    
    return pfeat;
  }
  
  map<string/*part name*/,Detection/*parts*/ > PCAPose::unproject_q(
    const Mat& pca_feat,DetectorResult rootBB,const ImRGBZ&im)
  {
    //cout << "raw unproject_q input: " << pca_feat << endl;
    Mat unnormalized_feat = pca_feat;
    assert(unnormalized_feat.type() == DataType<double>::type);
    for(int iter = 0; iter < N_DIM; ++iter)
      unnormalized_feat.at<double>(iter) = 
	ecdfs[iter].quant((1.0/PROJ_Q_MAX)*unnormalized_feat.at<double>(iter));
    
    return unproject(unnormalized_feat,rootBB,im);
  }
  
  double PCAPose::latentMax(int dimension) const
  {
    return latent_maxes[dimension];
  }

  double PCAPose::latentMin(int dimension) const
  {
    return latent_mins[dimension];
  }
  
  void pca_pose_train()
  {
    // get a collection of poses
    vector<shared_ptr<MetaData> > training_set = default_train_data();
    cout << printfpp("Loaded %d training points",(int)training_set.size()) << endl;
    
    PCAPose pca_pose;
    pca_pose.train(training_set);
    
    // test the forward - backward projection code
    for(auto & datum : training_set)
    {
      cout << "Datum = " << datum->get_filename() << endl;
      auto poss = datum->get_positives();
      Rect_<double> handBB = poss.at(root_part);
      cout << "root_bb: " << handBB << endl;
      DetectorResult det(new Detection());
      det->BB = handBB;
      shared_ptr<ImRGBZ> im = datum->load_im();
      
      Mat proj = pca_pose.project_q(datum);
      map<string/*part name*/,Detection/*parts*/ > unproj = 
	pca_pose.unproject_q(proj,det,*im);
      for(auto&part : unproj)
      {
	Rect_<double> unproj_part = part.second.BB;
	Rect_<double> orig_part = poss[part.first];
	cout << part.first << 
	  " orig_part: " << rectCenter(orig_part) << 
	  " unproj_part: " << rectCenter(unproj_part) << endl;
      }
    }
  }
}
