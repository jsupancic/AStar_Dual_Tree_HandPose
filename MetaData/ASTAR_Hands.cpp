/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ASTAR_Hands.hpp"

namespace deformable_depth
{
  string ASTAR_path = "/home/jsupanci/workspace/data/ASTAR-HANDS/";
  string ASTAR_data_path = ASTAR_path + "Test_Dataset/";
  int ASTAR_stride = 1;

  ASTAR_Video::ASTAR_Video(int id)
  {
    string subset;
    if(id == 0)
      subset = "Training_Dataset/";
    else if(id == 1)
      subset = "Test_Dataset/";
    else
      assert(false);
    vector<string> example_choices = find_files(ASTAR_path + subset,boost::regex(".*ini"),true);    
    for(int iter = 0; iter < example_choices.size(); iter += ASTAR_stride)
      examples.push_back(example_choices.at(iter));
  }

  ASTAR_Video::~ASTAR_Video()
  {
  }

  shared_ptr<MetaData_YML_Backed> ASTAR_Video::getFrame(int index,bool read_only)
  {
    string base = boost::regex_replace(examples.at(index),boost::regex("ini$"),"");
    string label_filename = base + "shand";
    string depth_filename = base + "skdepth";
    cout << "ASTAR-Video loading: " << depth_filename << endl;

    // load the depth data
    const int width = 320, height = 240;
    const int img_size = width * height;
    cv::Mat depth(height, width, CV_16SC1);
    ifstream filedepth_(depth_filename.c_str());
    filedepth_.read((char*) depth.data, img_size * sizeof(short));
    //depth.convertTo(D, CV_8UC1, 0.25, 0);
    //cv::imshow("depth", D);
    depth.convertTo(depth,DataType<float>::type);

    // load RGB
    Mat rgb = imageeq("",depth,false,false);

    // kinect camera    
    string metadata_filename = safe_printf("ASTAR_%",index);
    CustomCamera pxc_camera(params::H_RGB_FOV,params::V_RGB_FOV, qnan,qnan);
    shared_ptr<ImRGBZ> im(new ImRGBZ(rgb,depth,metadata_filename + "image",pxc_camera));

    // create the metadata object
    shared_ptr<Metadata_Simple> metadata = std::make_shared<Metadata_Simple>(metadata_filename+".yml",true,true,false);
    metadata->setIm(*im);

    // load the keypoints
    vector<Vec3d> keypoints;
    ifstream ifs(label_filename);
    if(ifs)
    {
      string line; getline(ifs,line);
      line = boost::regex_replace(line,boost::regex(",")," ");
      cout << "line = " << line << endl;
      istringstream iss(line);
      while(iss)
      {
	double x,y,z;
	iss >> z >> x >> y;
	x += width/2;
	y += height/2;
	keypoints.push_back(Vec3d(x,y,z));
      }

      cout << "loaded keypoint count = " << keypoints.size() << endl;

      // set the annotations
      Rect handBB;
      auto getKeypoint = [&](int num)
	{
	  Point2d pt(keypoints.at(num)[0],keypoints.at(num)[1]);
	  // compute the handBB from the keypoints.
	  if(handBB == Rect())
	    handBB = Rect(pt,Size(1,1));
	  else
	    handBB |= Rect(pt,Size(1,1));
	  return pt;
	};
      metadata->keypoint("carpals",getKeypoint(0),true);
      metadata->keypoint("Z_P0",getKeypoint(0),true);
      metadata->keypoint("Z_P1",getKeypoint(0),true);
      // thumb
      metadata->keypoint("Z_J53",getKeypoint(3),true);
      metadata->keypoint("Z_J52",getKeypoint(2),true);
      metadata->keypoint("Z_J51",getKeypoint(1),true);
      // index
      metadata->keypoint("Z_J43",getKeypoint(7),true);
      metadata->keypoint("Z_J42",getKeypoint(6),true);
      metadata->keypoint("Z_J41",getKeypoint(5),true);
      // mid
      metadata->keypoint("Z_J33",getKeypoint(11),true);
      metadata->keypoint("Z_J32",getKeypoint(10),true);
      metadata->keypoint("Z_J31",getKeypoint(9),true);
      // ring
      metadata->keypoint("Z_J23",getKeypoint(15),true);
      metadata->keypoint("Z_J22",getKeypoint(14),true);
      metadata->keypoint("Z_J21",getKeypoint(13),true);
      // pinky
      metadata->keypoint("Z_J13",getKeypoint(19),true);
      metadata->keypoint("Z_J12",getKeypoint(18),true);
      metadata->keypoint("Z_J11",getKeypoint(17),true);
      // set the hand bb
      metadata->set_HandBB(rectResize(handBB,1.8,1.8));
      metadata->set_is_left_hand(true);
    }

    return metadata;
  }

  int ASTAR_Video::getNumberOfFrames()
  {
    return examples.size();
  }

  string ASTAR_Video::get_name()
  {
    return "0.ASTAR_HANDS";
  }

  bool ASTAR_Video::is_frame_annotated(int index)
  {
    return true;
  }
}

