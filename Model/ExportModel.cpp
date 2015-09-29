/**
 * Copyright 2015: James Steven Supancic III
 **/

#include "ExportModel.hpp"

namespace deformable_depth
{
  DetectionSet ExportModel::detect(const ImRGBZ&im,DetectionFilter filter) const
  {
    cout << "++ExportModel::detect" << endl;
    
    DetectionSet results;
    static atomic<long> nextId(0);
    
    auto cheat = filter.cheat_code.lock();
    if(cheat)
    {
      for(auto && sub_datum : cheat->get_subdata())
      {
	// get the hand BB
	auto poss = sub_datum.second->get_positives();
	bool has_hand_bb = (poss.find("HandBB") != poss.end());
	if(!has_hand_bb)
	  continue;
	Rect sub_datum_bb = rectResize(poss["HandBB"],1.4,1.4);
	Point2d sub_datum_center = rectCenter(sub_datum_bb);      
	if(sub_datum_bb == Rect())
	  continue;

	// check that we have all keypoints
	bool has_all_kp = true;
	for(auto && ep : essential_hand_positives())
	  if(poss.find(ep) == poss.end())
	    has_all_kp = false;
	if(!has_all_kp)
	  continue;

	// get depth?
	float handDepth = manifoldFn_default(im,sub_datum_bb)[0];
	
	// write the keypoints
	int id = nextId++;
	ofstream ofs(params::out_dir() + safe_printf("/labels_%.txt",id));
	for(auto && ep : essential_hand_positives())
	{
	  Point2d center = rectCenter(poss.at(ep));
	  ofs << (center.x - sub_datum_bb.x) << " " << (center.y - sub_datum_bb.y) << endl;
	}
	
	// write the roi
	Mat Z = imroi(im.Z,sub_datum_bb);
	Z = imclamp(Z - handDepth,-15,15)/15;
	string zfilename = params::out_dir() + safe_printf("/z_%.exr",id);
	imwrite(zfilename,Z);

	// write textra
	string visZ_filename = params::out_dir() + safe_printf("/vizZ_%.png",id);
	imwrite(visZ_filename,imageeq("",im.Z,false,false));
	Mat fullZ = im.Z.clone();
	fullZ *= 10;
	fullZ.convertTo(fullZ,DataType<uint16_t>::type);
	string fullZ_filename = params::out_dir() + safe_printf("/fullz_%.png",id);
	imwrite(fullZ_filename,fullZ);
	string fullRGB_filename = params::out_dir() + safe_printf("/fullRGB_%.png",id);	
	imwrite(fullRGB_filename,im.RGB);
      }
    }
    
    return results;
  }
  
  void ExportModel::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    // for(auto && datum : training_set)
    // {
    //   // load the image
    //   shared_ptr<const ImRGBZ> im = datum->load_im();
    //   Mat Zeq = imageeq("",im->Z,false,false);

    //   // crop the  handBB
    //   auto poss = datum.get_positives();
    //   Rect handBB = poss["HandBB"];
    //   cv::rectangle(viz,handBB.tl(),handBB.br(),Scalar(0,0,255));      
    // }
  }
  
  Mat ExportModel::show(const string&title)
  {
    return image_text("ExportModel");
  }
}
