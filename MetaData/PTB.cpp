/**
 * Copyright 2014: James Steven Supancic III
 *
 * Metadata for the Princeton Tracking Becnhmark
 **/

#include <jsoncpp/json/json.h>
#include "PTB.hpp"
#include "RegEx.hpp"

namespace deformable_depth
{
  PTB_Video::PTB_Video(string name) 
  {
    filename = name;
    path = string("/mnt/data/jsupanci/PTB-Eval/") + name;

    // find the image data
    vector<string> rgb_file_names   = find_files(path + "/rgb/", boost::regex(".*png"));
    vector<string> depth_file_names = find_files(path + "/depth/", boost::regex(".*png"));    
    for(string & s : rgb_file_names)
    {
      int frame = fromString<int>(regex_matches(s, boost::regex("\\d+")).back());
      rgb_files[frame] = s;
    }
    for(string & s : depth_file_names)
    {
      int frame = fromString<int>(regex_matches(s, boost::regex("\\d+")).back());
      depth_files[frame] = s;      
    }

    // find the camera configuration
    // float fovx, fovy, fx, fy, cx, cy;
    Json::Reader reader;
    Json::Value root,K,fov;
    string json_file = path + "/frames.json";
    log_once(safe_printf("parsing json = %",json_file));
    ifstream json_ifs(json_file);
    std::string json_string(
      (std::istreambuf_iterator<char>(json_ifs)),
       std::istreambuf_iterator<char>());
    
    if(!reader.parse(json_string , root ))
    {
       std::cout << "Failed to parse configuration\n"
		 << reader.getFormattedErrorMessages() << endl;
       std::cout << "str = " << json_string << endl;
       assert(false);
    }
    K = root["K"];
    fov = root["fov"];
    require_equal<int>(fov.size(),2);
    require_equal<int>(K.size(),3);
    fovx = (fov[Json::Value::UInt(0)].asDouble());
    fovy = (fov[1].asDouble());
    fx   = (K[Json::Value::UInt(0)][Json::Value::UInt(0)].asDouble())/10;
    fy   = (K[1][1].asDouble())/10;
    cx   = (K[Json::Value::UInt(0)][2].asDouble());
    cy   = (K[1][2].asDouble());

    // load the init BB
    ifstream init_ifs(path + "/init.txt"); 
    std::string init_string(
      (std::istreambuf_iterator<char>(init_ifs)),
       std::istreambuf_iterator<char>());
    vector<string> init_strings = regex_matches(init_string,boost::regex("\\d+"));
    int tl = fromString<int>(init_strings[0]);
    int br = fromString<int>(init_strings[1]);
    int wd = fromString<int>(init_strings[2]);
    int he = fromString<int>(init_strings[3]);
    init = Rect(Point(tl,br),Size(wd,he));
  }
  
  PTB_Video::~PTB_Video()
  {
  }
  
  shared_ptr<MetaData_YML_Backed> PTB_Video::getFrame(int index,bool read_only)
  {
    bool has_depth = depth_files.find(index) != depth_files.end();
    bool has_rgb   = rgb_files.find(index)   != rgb_files.end();
    if(has_depth && has_rgb)
    {
      string id = safe_printf("%//%",filename,index);
      
      // set image
      Mat RGB = imread(rgb_files.at(index));
      Mat Z = imread(depth_files.at(index),-1);
      assert(Z.type() == DataType<uint16_t>::type);
      for(int yIter = 0; yIter < Z.rows; ++yIter)
	for(int xIter = 0; xIter < Z.cols; ++xIter)
	{
	  uint16_t&v = Z.at<uint16_t>(yIter,xIter);
	  
	  //uint16_t top_mask    = 0xE000;
	  //uint16_t bottom_mask = 0x0007;
	  //uint16_t neither_mask = ~top_mask & ~bottom_mask;	  
	  // extract the top and bottom bits
	  //uint16_t top = top_mask & v;
	  //uint16_t bot = bottom_mask & v;
	  //v &= neither_mask;
	  // repack swapped
	  //top >> 13;
	  //bot << 13;
	  //v |= (top | bot);

	  // 
	  v = (v >> 3) | (v << (16 -3 ) );
	}
      Z.convertTo(Z,DataType<float>::type);
      for(int yIter = 0; yIter < Z.rows; ++yIter)
	for(int xIter = 0; xIter < Z.cols; ++xIter)
	{
	  float&z = Z.at<float>(yIter,xIter);
	  z = clamp<float>(0,z/10,2000);
	}
      //log_im_decay_freq("PTBLoaded",eq(Z));
      CustomCamera camera(rad2deg(fovx),rad2deg(fovy),RGB.cols,RGB.rows,fx,fy);
      shared_ptr<ImRGBZ> im = make_shared<ImRGBZ>(RGB,Z,id,camera);

      // set datum
      shared_ptr<Metadata_Simple> datum = make_shared<Metadata_Simple>(id,true,true,false);
      datum->setIm(*im);

      // set gt for first frame
      if(index == 1)
      {
	map<string,AnnotationBoundingBox> notes;
	notes["TARGET"] = AnnotationBoundingBox(init,true);
	datum->setPositives(notes);
      }
      
      return datum;      
    }
    return nullptr;
  }
  
  int PTB_Video::getNumberOfFrames()
  {
    return std::min<int>(rgb_files.rbegin()->first,depth_files.rbegin()->first);
  }
  
  string PTB_Video::get_name()
  {
    return filename;
  }
  
  bool PTB_Video::is_frame_annotated(int index)
  {
    return index == 1;
  }
}

