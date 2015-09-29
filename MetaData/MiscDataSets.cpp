/**
 * Copyright 2014: James Steven Supancic III
 **/ 

#include "MiscDataSets.hpp"
#include "PXCSupport.hpp"
#include "Aggregate.hpp"

namespace deformable_depth
{
  shared_ptr<MetaDataAggregate> form_Metadatum(Mat RGB, Mat D, Mat UV,string im_name,string md_name)
  {
    if(RGB.empty())
    {
      RGB = imageeq("",D,false,false);
      for(int rIter = 0; rIter < RGB.rows; rIter++)
	for(int cIter = 0; cIter < RGB.cols; cIter++)
	  RGB.at<Vec3b>(rIter,cIter)[2] = 255;
    }
    log_file << "D.szie = " << printfpp("%d %d",(int)D.rows,(int)D.cols) << endl;
    resize(RGB,RGB,D.size());

    // create the raw image
    CustomCamera pxc_camera(params::H_RGB_FOV,params::V_RGB_FOV, qnan,qnan);
    shared_ptr<ImRGBZ> im_raw(new ImRGBZ(RGB,D,im_name,pxc_camera));
    Mat regRGB;
    PXCRegistration regInfo;
    if(UV.empty())
    {
      regRGB = RGB.clone();
    }
    else
    {
      regInfo = registerRGB2Depth_adv(RGB,D,UV);   
      regRGB = regInfo.registration;
    }
    shared_ptr<ImRGBZ> im(new ImRGBZ(regRGB,D,im_name,pxc_camera));
    if(UV.empty())
      im->valid_region = Rect(regInfo.tl_valid,regInfo.br_valid);

    // create the metadata object    
    shared_ptr<Metadata_Simple> metadata_left(new Metadata_Simple(md_name + "_left",true,true));
    shared_ptr<Metadata_Simple> metadata_right(new Metadata_Simple(md_name + "_right" ,true,true));
    metadata_left->setRawIm(*im_raw); metadata_right->setRawIm(*im_raw);
    metadata_left->setIm(*im); metadata_right->setIm(*im);

    map<string,shared_ptr<MetaData_YML_Backed> > comps
    {{"right_hand",metadata_left},{"left_hand",metadata_right}};
    shared_ptr<MetaDataAggregate> agg = make_shared<MetaDataAggregate>(comps);

    return agg;
  }

  void load_annotations(MetaData_YML_Backed * md,string annotation_file)
  {
    ifstream ifs(annotation_file);
    if(not ifs)
    {
      log_file << "warning: failed to load " << annotation_file << endl;
      return;
    }
    log_file << "loaded " << annotation_file << endl;
    LibHand_DefDepth_Keypoint_Mapping name_map;
    Rect handBB;
    while(ifs)
    {
      string line; std::getline(ifs,line);
      istringstream iss(line);
      string ignore, libhand_name; iss >> libhand_name >> ignore;
      double x; iss >> x;
      double y; iss >> y;
      double z; iss >> z;
      
      // update keypoint
      string dd_name = name_map.lh_to_dd[libhand_name];
      if(dd_name != "")
      {
	log_once(safe_printf("info: mapped % to %",libhand_name,dd_name));
	md->keypoint(dd_name, Point2d(x,y), true);
      }
      else
	log_once(safe_printf("warning: failed to map %",libhand_name));

      // update bb
      if(handBB == Rect())
	handBB = Rect(Point(x,y),Size(1,1));
      else
	handBB |= Rect(Point(x,y),Size(1,1));
    }
    handBB = clamp(md->load_im()->Z,handBB);
    md->set_HandBB(handBB);
    log_file << "load_annotations: " << handBB << endl;
  }

  vector<shared_ptr<MetaData> > egocentric_dir_data()
  {
    const string egocentric_train_dir = "/home/grogez/Synth-Aug2014/";    
    vector<string> gt_files = find_files(egocentric_train_dir,boost::regex(".*txt"));
    vector<shared_ptr<MetaData> > data;
    
    vector<int> frames;
    for(int iter = 0; iter < gt_files.size(); ++iter)
      frames.push_back(iter);
    frames = pseudorandom_shuffle(frames);

    for(int iter = 0; !frames.empty() and iter < params::max_examples();  ++iter)
    {
      int frameIter = frames.at(iter);
      const string&ectd = egocentric_train_dir;
      cout << gt_files[frameIter] << endl;
      
      string label_filename = safe_printf("%/labels_%.png",ectd,frameIter);
      Mat Labels = imread(label_filename);
      for(int yIter = 0; yIter < Labels.rows; ++yIter)
	for(int xIter = 0; xIter < Labels.cols; ++xIter)
	  if(Labels.at<Vec3b>(yIter,xIter) == Vec3b(255,255,255))
	    Labels.at<Vec3b>(yIter,xIter) = HandRegion::background_color();
      string z_filename = safe_printf("%/Depth_%.png",ectd,frameIter);
      Mat Z = imread(z_filename,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
      Z.convertTo(Z,DataType<float>::type);
      Z = .1 * Z;
      string im_name = label_filename;
      string md_name = "md" + im_name;

      if(!Labels.empty() and !Z.empty())
      {
	image_safe("vis",horizCat(Labels,imageeq("",Z,false,false)));

	auto datum = form_Metadatum(Mat(),Z,Mat(),im_name,md_name);
	datum->setSegmentation(Labels);
	datum->set_is_left_hand(false);
	load_annotations(datum->get_subdata_yml().at("right_hand"),safe_printf("%/pose_%.txt",ectd,frameIter));
	assert(datum->use_positives());
	data.push_back(datum);		
      }
    }

    return pseudorandom_shuffle(data);
  }
}
