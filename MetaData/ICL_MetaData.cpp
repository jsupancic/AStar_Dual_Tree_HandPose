/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "ICL_MetaData.hpp"
#include "RegEx.hpp"
#include "MetaData_Pool.hpp"
#include "LibHandRenderer.hpp"

#include "boost/filesystem.hpp"
#include "boost/algorithm/string/trim.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: ICL MetaData interface class implementation
  ///
  
  string icl_base()
  {
    const string dir_base = "/home/jsupanci/workspace/data/ICL_HANDS2/";
    if(g_params.has_key("ICL_BASE"))
      return g_params.get_value("ICL_BASE");
    else
      return dir_base;
  }

  static string getDirectory(string id)
  {
    int num = fromString<int>(deformable_depth::regex_match(id,boost::regex(".*\\d.*"))[0]);
    return string(icl_base() + "/Testing/Depth/test_seq_") + std::to_string(num) + "/";
  }  

  static string getAnnotationFile(string id)
  {
    int num = fromString<int>(deformable_depth::regex_match(id,boost::regex(".*\\d.*"))[0]);
    return string(icl_base() + "/Testing/test_seq_") + std::to_string(num) + ".txt";    
  }

  ICL_Video::ICL_Video(string filename) : filename(filename)
  {
    // parse the annotation file
    ifstream ifs(getAnnotationFile(filename));
    for(int frameIter = 0; ifs; frameIter++)
    {
      string line; std::getline(ifs,line);
      frame_labels[frameIter] = (line);
    }
  }

  ICL_Video::~ICL_Video()
  {
  }

  static Rect  set_annotations(Metadata_Simple*metadata,const vector<double>&labels,shared_ptr<ImRGBZ>&im)
  {
    // Label description: 	
    //     Each line is corresponding to one image.
    //     Each line has 16x3 numbers, which indicates (x, y, z) of 16 joint locations. Note that these are joint CENTRE locations.
    //     Note that (x, y) are in pixels and z is in mm.
    //     The order of 16 joints is 0:Palm, 1Thumb root, 2Thumb mid, 3Thumb tip, 4Index root, 5Index mid, 6Index tip, 
    //         7Middle root, 8Middle mid, 9Middle tip, 10:Ring root, 11:Ring mid, 12:Ring tip, 13:Pinky root, 14:Pinky mid, 15:Pinky tip.
    //     We used Intel Creative depth sensor. Calibration parameters can be obtained as in Page 119 of SDK Manual:
    Rect handBB;
    auto getKeypoint = [&](int index)
    {
      Point2d pt(labels[0 + index*3],labels[1 + index*3]);
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
    metadata->keypoint("Z_J53",getKeypoint(1),true);
    metadata->keypoint("Z_J52",getKeypoint(2),true);
    metadata->keypoint("Z_J51",getKeypoint(3),true);
    // index
    metadata->keypoint("Z_J43",getKeypoint(4),true);
    metadata->keypoint("Z_J42",getKeypoint(5),true);
    metadata->keypoint("Z_J41",getKeypoint(6),true);
    // mid
    metadata->keypoint("Z_J33",getKeypoint(7),true);
    metadata->keypoint("Z_J32",getKeypoint(8),true);
    metadata->keypoint("Z_J31",getKeypoint(9),true);
    // ring
    metadata->keypoint("Z_J23",getKeypoint(10),true);
    metadata->keypoint("Z_J22",getKeypoint(11),true);
    metadata->keypoint("Z_J21",getKeypoint(12),true);
    // pinky
    metadata->keypoint("Z_J13",getKeypoint(13),true);
    metadata->keypoint("Z_J12",getKeypoint(14),true);
    metadata->keypoint("Z_J11",getKeypoint(15),true);
    // set the hand bb
    Mat Zeroded = imopen(im->Z,5); //cv::erode(im,result,Mat());
    handBB = bbWhere(Zeroded,[](Mat&Z,int y,int x)
		     {
		       return goodNumber(Z.at<float>(y,x)) and Z.at<float>(y,x) < params::MAX_Z();
		     });
    handBB = rectResize(handBB,2,2);
    metadata->set_HandBB(handBB);
    metadata->set_is_left_hand(true);

    // generate the segmentation
    if(g_params.get_value("SEGMENTATION") == "TRUE")
    {
      Mat seg = drawRegions(*metadata);
      metadata->setSegmentation(seg);
    }

    return handBB;
  }

  static void suppress_most_common_pixel_hack(Mat&z,Rect handBB)
  { 
    // hack, remove the most common pixel
    std::map<float,double> counts;
    Rect imBB(Point(0,0),z.size());

    for(int yIter = 0; yIter < z.rows; ++yIter)
      for(int xIter = 0; xIter < z.cols; ++xIter)
	if(!handBB.contains(Point(xIter,yIter)))
	{
	  float zz = z.at<float>(yIter,xIter);
	  if(goodNumber(zz) && params::MIN_Z() <= zz && zz <= params::MAX_Z())
	    counts[zz] ++;
	}

    // find the larget count
    float common_z, common_zs_count = -inf;
    for(auto & pair : counts)
      if(pair.second > common_zs_count)
      {
	common_z = pair.first;
	common_zs_count = pair.second;
      }
    log_once(safe_printf("note: suppressing % %",common_z,common_zs_count));	     

    // suppress!
    for(int yIter = 0; yIter < z.rows; ++yIter)
      for(int xIter = 0; xIter < z.cols; ++xIter)
	if(!handBB.contains(Point(xIter,yIter)))
	{
	  float&zz = z.at<float>(yIter,xIter);
	  if(zz == common_z)
	    zz = inf;
	}	  
  }

  static void suppress_flood_fill_hack(Mat&z,Point start)
  {
    vector<Point> frontier{start};

    // segment the hand
    Mat seg(z.rows,z.cols,DataType<int>::type,Scalar::all(0));
    while(!frontier.empty())
    {
      Point here = frontier.back();
      frontier.pop_back();
      seg.at<int>(here.y,here.x) = 255;
      float z_here = z.at<float>(here.y,here.x);

      auto tryNeighbor = [&](Point pt)
      {
	if(pt.x < 0 || z.cols <= pt.x || pt.y < 0 || z.rows <= pt.y)
	  return;
	if(seg.at<int>(pt.y,pt.x) > 0)
	  return;
	float z_pt = z.at<float>(pt.y,pt.x);
	if(std::abs(z_here - z_pt) > params::DEPTHS_SIMILAR_THRESAH)
	  return;

	frontier.push_back(pt);
      };
      // up down
      tryNeighbor(Point(here.x,here.y+1));
      tryNeighbor(Point(here.x,here.y-1));
      // left
      tryNeighbor(Point(here.x+1,here.y+1));
      tryNeighbor(Point(here.x+1,here.y));
      tryNeighbor(Point(here.x+1,here.y-1));
      // right
      tryNeighbor(Point(here.x-1,here.y+1));
      tryNeighbor(Point(here.x-1,here.y));
      tryNeighbor(Point(here.x-1,here.y-1));
    }

    // erase the background
    for(int yIter = 0; yIter < z.rows; ++yIter)
      for(int xIter = 0; xIter < z.cols; ++xIter)
	if(seg.at<int>(yIter,xIter) == 0)
	  z.at<float>(yIter,xIter) = inf;
  }
  
  MetaData_YML_Backed* load_ICL(string base_path,string annotation,bool training)
  {
    // parse the annotation
    vector<double> labels;
    istringstream iss(annotation);
    string filename; iss >> filename; // discard.
    cout << "annotation " << annotation << endl;
    cout << "filename " << filename << endl;
    while(iss)
    {
      double v; iss >> v;
      labels.push_back(v);
    }    

    // calc the name for this metadata    
    string metadata_filename = base_path + filename;

    // load the raw data
    float active_min = training?0:params::MIN_Z();
    string frame_file = base_path + filename;
    Mat depth = imread(frame_file,-1);
    assert(!depth.empty());
    CustomCamera pxc_camera(params::H_RGB_FOV,params::V_RGB_FOV, depth.cols,depth.rows);
    depth.convertTo(depth,DataType<float>::type);
    depth /= 10;
    for(int rIter = 0; rIter < depth.rows; ++rIter)
      for(int cIter = 0; cIter < depth.cols; ++cIter)
      {
	float & d = depth.at<float>(rIter,cIter);
	if(d <= active_min)
	  d = inf;
      }
    //depth = pointCloudToDepth(depth,pxc_camera);
    
    if(depth.empty())
    {
      log_once(printfpp("ICL_Video cannot load %s",metadata_filename.c_str()));
      log_once(printfpp("Couldn't open %s",frame_file.c_str()));
      return nullptr;
    }
    //Mat RGB (depth.rows,depth.cols,DataType<Vec3b>::type,Scalar(122,150,233));
    Mat RGB = imageeq("",depth,false,false);
    for(int rIter = 0; rIter < RGB.rows; rIter++)
      for(int cIter = 0; cIter < RGB.cols; cIter++)
	RGB.at<Vec3b>(rIter,cIter)[2] = 255;

    // create an image    
    shared_ptr<ImRGBZ> im =
      make_shared<ImRGBZ>(RGB,depth,metadata_filename + "image",pxc_camera);

    // create the metadata object
    Metadata_Simple* metadata = new Metadata_Simple(metadata_filename+".yml",true,true,false);
    metadata->setIm(*im);

    Point2d hand_root(labels[0 + 0*3],labels[1 + 0*3]);
    suppress_flood_fill_hack(im->Z,hand_root);
    metadata->setIm(*im);
    Rect handBB = set_annotations(metadata,labels,im);
    suppress_most_common_pixel_hack(im->Z,handBB);    
    metadata->setIm(*im);
    
    log_im_decay_freq("ICL_loaded",[&]()
		      {
			return image_datum(*metadata);
		      });

    return metadata;
  }

  shared_ptr<MetaData_YML_Backed> ICL_Video::getFrame(int index,bool read_only)
  {
    return shared_ptr<MetaData_YML_Backed>(load_ICL(
					     icl_base() + "/Testing/Depth/",
					     frame_labels[index],
					     false));
  }

  int ICL_Video::getNumberOfFrames()
  {   
    return find_files(getDirectory(filename),boost::regex(".*image.*")).size();
  }

  string ICL_Video::get_name()
  {
    return filename;
  }

  bool ICL_Video::is_frame_annotated(int index)
  {
    return true;
  }

  ///
  /// SECTION: Producing the ICL training set
  ///
  MetaData* load_ICL_training(string descriptor)
  {
    return load_ICL(icl_base() + "/Training/Depth/",descriptor.substr(3,string::npos),true);
  }
  
  vector<shared_ptr<MetaData> > ICL_training_set()
  {
    // read and randomzie the lines
    vector<string> lines;
    ifstream ifs("/home/jsupanci/workspace/data/ICL_HANDS2/Training/labels.txt");
    // can't add && lines.size() < params::max_examples() without biasing sample
    while(ifs )
    {
      string line; std::getline(ifs,line);
      istringstream iss(line);
      string file; iss >> file;
      string frame_file = icl_base() + "/Training/Depth/" + file;
      boost::algorithm::trim(file);
      boost::regex real_pattern(".*Depth/20140\\d*/.*");
      
      bool files_exist = boost::filesystem::exists(frame_file) and file != "";
      bool pre_rotated = not boost::regex_match(frame_file,real_pattern);
      if((not files_exist) or (!g_params.option_is_set("DATA_SOURCE_PRE_ROTATE") and pre_rotated))
      {
	//cout << "reject: " << line << endl;      
      }
      else
      {
	lines.push_back(line);
      }
    }
    lines = pseudorandom_shuffle(lines);

    // now key up metadata for the lines
    vector<shared_ptr<MetaData> > data;    
    for(int iter = 0; iter < lines.size() and data.size() < params::max_examples(); ++iter)
    {      
      string line = lines[iter];
      cout << "parsing: " << line << endl;
      data.push_back(metadata_build("ICL"+line,true,false));
    }
    return data;
  }

  void show_ICL_training_set()
  {
    cout << "+show_ICL_training_set" << endl;
    for(auto && datum : ICL_training_set())
    {
      image_datum(*datum,true);
    }
  }
}
