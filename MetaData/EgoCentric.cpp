/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "EgoCentric.hpp"
#include "Video.hpp"
#include "RegEx.hpp"
#include "IKSynther.hpp"

namespace deformable_depth
{
  static Mat drawRegions(const Mat&greg_sem)
  {
    assert(greg_sem.type() == DataType<Vec3b>::type);
    Mat seg(greg_sem.rows,greg_sem.cols,DataType<Vec3b>::type,Scalar::all(0));
    
    for(int yIter = 0; yIter < seg.rows; yIter++)
      for(int xIter = 0; xIter < seg.cols; xIter++)
      {
	int BASE = 8;
	int index = greg_sem.at<Vec3b>(yIter,xIter)[0];
	int raw_index = index;
	int b0 = index % BASE;
	index /= BASE;
	int b1 = index % BASE;
	index /= BASE;
	int b2 = index % BASE;
	index /= BASE;	
	require(index == 0,safe_printf("% %",index,raw_index));
	Vec3b color(b0 * 43, b1 * 43, b2 * 43);
	
	seg.at<Vec3b>(yIter,xIter) = color;
      }

    return seg;
  }

  vector<shared_ptr<MetaData> > EgoCentric_Poser_training_set()
  {
#ifdef DD_ENABLE_HAND_SYNTH
    string direc = ("/home/grogez/Egocentric_Synth_Poser/");
    vector<string> gt_files = find_files(direc, boost::regex(".*\\.txt"));
    log_file << "found " << gt_files.size() << " ground truths" << endl;

    vector<shared_ptr<MetaData> > result;
    for(string gt_file : gt_files)
    {
      log_file << "loading gt = " << gt_file << endl;
      int number = fromString<int>(regex_match(gt_file, boost::regex("\\d+")).front());
      string gt_path = safe_printf("%/gen%.txt",direc,number);
      string rgb_path = safe_printf("%/gen%_rgb.png",direc,number);
      string depth_path = safe_printf("%/gen%_Z.png",direc,number);
      string labels_path = safe_printf("%/gen%_labels.png",direc,number);

      // load the data
      IK_Grasp_Pose joint_positions = greg_load_grasp_pose_one(direc,gt_file,false);
      Mat RGB = imread(rgb_path);
      Mat Z = imread(depth_path,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
      Z.convertTo(Z,DataType<float>::type);
      Z /= 10; // mm to cm
      Mat labels = imread(labels_path,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
      labels.convertTo(labels,DataType<float>::type);
      labels = imageeq("",labels,false,false);
      
      // construct an image for the datum
      string metadata_filename = "EgoCentricPoser" + uuid();
      CustomCamera pxc_camera(params::H_RGB_FOV,params::V_RGB_FOV, qnan,qnan);
      shared_ptr<ImRGBZ> im(new ImRGBZ(RGB,Z,metadata_filename + "image",pxc_camera));
      // alloc the datum
      shared_ptr<Metadata_Simple> metadata = make_shared<Metadata_Simple>(metadata_filename+".yml",true,true,false);
      metadata->setIm(*im);
      // write the keypoints, these are all right hands
      setAnnotations(metadata.get(),joint_positions);
      metadata->set_is_left_hand(false);
      // 
      Mat seg = drawRegions(labels);
      metadata->setSegmentation(seg);
      //metadata->setSegmentation(labels);

      // draw the contructed to verify correct implementation.
      log_im_decay_freq("egoposer_loaded",[&]()
			{
			  return horizCat(image_datum(*metadata,false),labels);
			});
      result.push_back(metadata); // :-)
    }
    
    return result;
#else
    return vector<shared_ptr<MetaData> >{};
#endif
  }
}
