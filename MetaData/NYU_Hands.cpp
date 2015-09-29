/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "NYU_Hands.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: NYU Testing Set
  /// 
  int NYU_TESTING_STRIDE = 5; //40

  struct NYU_Labels
  {
  public:
    Mat train_us;
    Mat train_vs;
    Mat train_ds;
    Mat test_us;
    Mat test_vs;
    Mat test_ds;
    
    NYU_Labels()
    {
      test_us = read_csv_double(nyu_prefix() + "/joint_1_u.csv").t();
      test_vs = read_csv_double(nyu_prefix() + "/joint_1_v.csv").t();
      test_ds = read_csv_double(nyu_prefix() + "/joint_1_d.csv").t();
      string training_dir = nyu_base() + "/train/";
      train_us = read_csv_double(training_dir + "/train_u.csv").t();
      train_vs = read_csv_double(training_dir + "/train_v.csv").t();
      train_ds = read_csv_double(training_dir + "/train_d.csv").t();  
    }
  };
  
  static NYU_Labels*nyu_labels()
  {
    static mutex m; lock_guard<mutex> l(m);
    static NYU_Labels *singleton = nullptr;
    if(singleton == nullptr)
      singleton = new NYU_Labels();
    return singleton;
  }

  string nyu_base()
  {
    if(g_params.option_is_set("NYU_BASE"))
      return g_params.require("NYU_BASE");
    else
      return "/home/jsupanci/workspace/data/NYU-Hands-v2/";
  }

  string nyu_prefix()
  {
    return nyu_base() + "/test/";
  }

  NYU_Video::NYU_Video()
  {
    // calc frame count
    vector<string> files = find_files(nyu_prefix(), boost::regex(".*rgb_1_.*.png.*"));
    cout << "NYU_Video::getNumberOfFrames() = " << files.size() << " in " << nyu_prefix() << endl;
    frame_count = files.size() / NYU_TESTING_STRIDE;

    // load gt
    Mat XS = nyu_labels()->test_us;
    Mat YS = nyu_labels()->test_vs;
    Mat ZS = nyu_labels()->test_ds;

    for(int iter = 0; iter < XS.rows; iter++)
    {
      vector<Vec3d> hand_pose;
      for(int jter = 0; jter < XS.cols; ++jter)
      {	
	double u = XS.at<double>(iter,jter);
	double v = YS.at<double>(iter,jter);	
	double d = ZS.at<double>(iter,jter);
	hand_pose.push_back(Vec3d(u,v,d));
      } // read all numbers from line
      cout << "[";
      for(auto && vec : hand_pose)
	cout << "(" << vec[0] << "," << vec[1] << "," << vec[2] << ")";
      cout << "]" << endl;
      keypoints.push_back(hand_pose);
    }    
    log_file << "loaded hand poses: " << keypoints.size() << endl;
  }

  NYU_Video::~NYU_Video()
  {
  }

  MetaData_YML_Backed* NYU_Video_make_datum_raw(KeypointFn keypointFn,string metadata_filename,const ImRGBZ&im)
  {
    // create the metadata object
    Metadata_Simple* metadata = new Metadata_Simple(metadata_filename+".yml",true,true,false);
    metadata->setIm(im);

    Rect handBB;
    KeypointFn getKeypoint = [&](int index)
    {
      Point2d pt = keypointFn(index);
      // compute the handBB from the keypoints.
      if(handBB == Rect())
	handBB = Rect(pt,Size(1,1));
      else
	handBB |= Rect(pt,Size(1,1));
      return pt;
    };

    // set the keypoints    
    metadata->keypoint("carpals",getKeypoint(29),true);
    metadata->keypoint("Z_P0",getKeypoint(29),true);
    metadata->keypoint("Z_P1",getKeypoint(29),true);
    // thumb
    metadata->keypoint("Z_J53",getKeypoint(24),true);
    metadata->keypoint("Z_J52",getKeypoint(25),true);
    metadata->keypoint("Z_J51",getKeypoint(26),true);
    // index
    metadata->keypoint("Z_J41",getKeypoint(18),true);
    metadata->keypoint("Z_J42",getKeypoint(19),true);
    metadata->keypoint("Z_J43",getKeypoint(20),true);
    // mid
    metadata->keypoint("Z_J31",getKeypoint(12),true);
    metadata->keypoint("Z_J32",getKeypoint(13),true);
    metadata->keypoint("Z_J33",getKeypoint(14),true);
    // ring
    metadata->keypoint("Z_J21",getKeypoint(6),true);
    metadata->keypoint("Z_J22",getKeypoint(7),true);
    metadata->keypoint("Z_J23",getKeypoint(8),true);
    // pinky
    metadata->keypoint("Z_J11",getKeypoint(0),true);
    metadata->keypoint("Z_J12",getKeypoint(1),true);
    metadata->keypoint("Z_J13",getKeypoint(2),true);
    // set the hand bb
    metadata->set_HandBB(rectResize(handBB,1.1,1.1));
    metadata->set_is_left_hand(true);

    return metadata;
  }

  shared_ptr<MetaData_YML_Backed> NYU_Video_make_datum(KeypointFn keypointFn,string metadata_filename,const ImRGBZ&im)
  {
    return shared_ptr<MetaData_YML_Backed>(NYU_Video_make_datum_raw(keypointFn,metadata_filename,im));
  }

  shared_ptr<ImRGBZ> NYU_Make_Image(string direc, int index,string metadata_filename)
  {
    string rgb_filename = printfpp("%s/rgb_1_%07d.png",direc.c_str(),index);
    cout << "loading: " << rgb_filename << endl;
    Mat rgb   = imread(rgb_filename);
    Mat coded_depth = imread(printfpp("%s/depth_1_%07d.png",direc.c_str(),index));
    Mat depth(rgb.rows,rgb.cols,DataType<float>::type,Scalar::all(0));
    float max_depth = -inf;
    for(int rIter = 0; rIter < depth.rows; rIter++)
      for(int cIter = 0; cIter < depth.cols; cIter++)
      {
	// Note: In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue.
	Vec3b coded_pix = coded_depth.at<Vec3b>(rIter,cIter);
	float decoded_depth = 256*coded_pix[1] + coded_pix[0]; // take the green and blue
	float zf = static_cast<float>(decoded_depth)/10;	
	if(zf > 200)
	  zf = inf;
	depth.at<float>(rIter,cIter) = zf;
	max_depth = std::max<float>(max_depth,zf);
      }
    log_once(safe_printf("NYU max_depth = %",max_depth));

    if(rgb.empty() or depth.empty())
      return nullptr;
    depth = fillDepthHoles(depth,5);
    
    // dilate?
    //depth = imclose(depth);

    // kinect camera    
    CustomCamera pxc_camera(params::H_RGB_FOV,params::V_RGB_FOV, qnan,qnan);
    return make_shared<ImRGBZ>(rgb,depth,metadata_filename + "image",pxc_camera);
  }

  shared_ptr<MetaData_YML_Backed> NYU_Video::getFrame(int index,bool read_only)
  {       
    // load the data
    index *= NYU_TESTING_STRIDE;    
    index++;
    cout << "NYU_Video::getFrame " << index << endl;
    string metadata_filename = safe_printf("NYU%",index);
    shared_ptr<ImRGBZ> im = NYU_Make_Image(nyu_prefix(),index,metadata_filename);
    assert(im);

    KeypointFn getKeypoint = [&](int num)
    {
      auto record = keypoints.at(index-1);
      if(num < 0 or num >= record.size())
      {
	cout << "index = " << index << endl;
	cout << "num = " << num << endl;
	cout << "record.size = " << record.size() << endl;
	assert(false);
      }
      double u = record.at(num)[0];
      double v = record.at(num)[1];
      Point2d pt(u,v);
      return pt;
    };
    return NYU_Video_make_datum(getKeypoint,metadata_filename,*im);
  }

  int NYU_Video::getNumberOfFrames()
  {
    return frame_count;
  }

  string NYU_Video::get_name()
  {
    return "NYU_Hands";
  }

  bool NYU_Video::is_frame_annotated(int index)
  {
    return true;
  }

  ///
  /// SECTION: NYU Training Set
  /// 
  static vector<size_t> sort_indices_random()
  {
    int npos = fromString<int>(g_params.require("NPOS"));

    // get the training data
    string training_dir = nyu_base() + "/train/";
    Mat training_us = nyu_labels()->train_us; 
    Mat training_vs = nyu_labels()->train_vs; 
    Mat training_ds = nyu_labels()->train_ds; 

    // 
    std::vector<size_t> indexes(training_us.rows);
    std::iota(indexes.begin(), indexes.end(), 0);
    indexes = pseudorandom_shuffle(indexes);

    // select NPOS indices to return...
    vector<size_t> indexes_final;
    for(int iter = 0; iter < std::min(npos,training_us.rows); ++iter)
      indexes_final.push_back(indexes.at(iter));
    return indexes_final;
  }

  static vector<size_t> sort_indices_smart_xs()
  {
    return vector<size_t>{}; // TODO
  }

  // return the training indexes sorted by distances to a testing set
  static double dist(Mat us1, Mat vs1, Mat ds1, Mat us2, Mat vs2, Mat ds2)
  {
    double sq_dist = 0;
    vector<double> v1_u, v1_v, v1_d, v2_u, v2_v, v2_d;
    for(int cIter = 0; cIter < ds1.cols; cIter++)
    {
      // double u_diff = at(training_us,0,cIter) 
      // 	- at(testing_us,0,cIter);
      // double v_diff = at(training_vs,0,cIter) 
      // 	- at(testing_vs,0,cIter);
      // double d_diff = at(training_ds,0,cIter) 
      // 	- at(testing_ds,0,cIter);
      // sq_dist += u_diff*u_diff + v_diff*v_diff + d_diff*d_diff;
      v1_u.push_back(at(us1,0,cIter));
      v1_v.push_back(at(vs1,0,cIter));      
      v1_d.push_back(at(ds1,0,cIter));
      v2_u.push_back(at(us2,0,cIter));
      v2_v.push_back(at(vs2,0,cIter));
      v2_d.push_back(at(ds2,0,cIter));
    }

    auto do_dist = [](
      vector<double>&v1_u, 
      vector<double>&v1_v,
      vector<double>&v1_d,
      vector<double>&v2_u,
      vector<double>&v2_v,
      vector<double>&v2_d)
    {
      standardize(v1_u);
      standardize(v1_v);
      standardize(v1_d);
      standardize(v2_u);
      standardize(v2_v);
      standardize(v2_d);
      vector<double> delta_u = v1_u - v2_u;
      vector<double> delta_v = v1_v - v2_v;
      vector<double> delta_d = v1_d - v2_d;

      return dot_self(delta_u) + dot_self(delta_v) + dot_self(delta_d);
    };

    vector<double> v1_u_neg = - v1_u;
    return std::min<double>(do_dist( v1_u,v1_v,v1_d,v2_u,v2_v,v2_d),
			    do_dist(v1_u_neg,v1_v,v1_d,v2_u,v2_v,v2_d));
  }

  static vector<size_t> sort_indices_smart_thetas(
    Mat testing_us,Mat testing_vs,Mat testing_ds,
    Mat training_us,Mat training_vs,Mat training_ds)
  {
    // precompute the min dists
    vector<double> min_dists(training_ds.rows);
    //TaskBlock calc_min_dists("calc_min_dists");    
    for(int train_index = 0; train_index < training_ds.rows; train_index++)
    {
      //calc_min_dists.add_callee([&,train_index]()
				{
				  double min_sq_dist = inf;
				  
				  for(int test_iter = 0; 
				      test_iter < testing_us.rows; 
				      test_iter ++)
				  {
				    double sq_dist = dist(training_us.row(train_index),
							  training_vs.row(train_index),
							  training_ds.row(train_index),
							  testing_us.row(test_iter),
							  testing_vs.row(test_iter),
							  testing_ds.row(test_iter));
				    
				    if(sq_dist < min_sq_dist)
				      min_sq_dist = sq_dist;
				  }
				  
				  min_dists.at(train_index) = min_sq_dist;				  
				}//);
    }    
    //calc_min_dists.execute();

    // function to compute distance from a training instance to the test set
    auto dist = [&](size_t train_index)
    {
      return min_dists.at(train_index);
    };

    // sort the training indexes by their distance to a testing example
    std::vector<size_t> indexes(training_us.rows);
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(indexes.begin(),indexes.end(),[&](size_t a, size_t b)
	      { 	       		
		return dist(a) < dist(b);
	      });

    return indexes;
  }

  static vector<size_t> sort_indices_smart_thetas()
  {
    // load the annotations
    Mat testing_us = nyu_labels()->test_us;
    Mat testing_vs = nyu_labels()->test_vs;
    Mat testing_ds = nyu_labels()->test_ds;

    string training_dir = nyu_base() + "/train/";
    Mat training_us = nyu_labels()->train_us;
    Mat training_vs = nyu_labels()->train_vs;
    Mat training_ds = nyu_labels()->train_ds;
    assert(training_ds.cols == testing_ds.cols);

    vector<size_t> indexes;
    TaskBlock per_testing_iter("per_testing_iter");
    atomic<int> progress(0);
    int NYU_SMART_INDEX_STRIDE = 20;
    for(int testing_iter = 0; testing_iter < testing_us.rows; testing_iter += NYU_SMART_INDEX_STRIDE)
    {
      per_testing_iter.add_callee([&,testing_iter]()
				  {
				    vector<size_t> matches = sort_indices_smart_thetas(
				      testing_us.row(testing_iter),
				      testing_vs.row(testing_iter),
				      testing_ds.row(testing_iter),
				      training_us,
				      training_vs,
				      training_ds);				    
				    double K = 5;
				    double b = std::exp(std::log(matches.size()) / 5);

				    for(int iter = 0; iter < 5; ++iter)
				    {
				      int index = clamp<int>(0,std::pow(b,iter)-1,matches.size()-1); 
				      auto match = matches.at(index);

				      static mutex m; lock_guard<mutex> l(m);
				      indexes.push_back(match);
				    }

				    log_im_decay_freq("train-test-match-theta",[&]()
						      {
							vector<Mat> training_images;
							for(int iter = 0; iter < 5; ++iter)
							{
							  int index = clamp<int>(0,std::pow(b,iter)-1,matches.size()-1); 
							  auto match = matches.at(index);
							  shared_ptr<ImRGBZ> train_image = 
							    NYU_Make_Image(nyu_base()+"/train/", match,uuid());
							  training_images.push_back(vertCat(imageeq("",train_image->Z,false,false),
											    image_text(safe_printf("index = %",index))));
							}							

							shared_ptr<ImRGBZ> test_image  = 
							  NYU_Make_Image(nyu_prefix(), testing_iter+1, uuid());
							return horizCat(tileCat(training_images),
									imageeq("",test_image->Z,false,false));
						      });
				    
				    progressBars->set_progress("calcing_indexes",progress++,training_ds.rows/NYU_TESTING_STRIDE);
				  });
    }
    progressBars->set_progress("calcing_indexes",0,0);
    per_testing_iter.execute();

    // take NPOS iterators
    int npos = fromString<int>(g_params.require("NPOS"));
    vector<size_t> selected_indexes;
    for(int iter = 0; iter < clamp<int>(0,npos,indexes.size()); ++iter)
      selected_indexes.push_back(indexes.at(iter));
    return selected_indexes;
  }

  MetaData* NYU_training_datum(int recordIter)
  {
    string training_dir = nyu_base() + "/train/";
    string datum_name = safe_printf("nyu_training_%",recordIter);
    shared_ptr<ImRGBZ> im = NYU_Make_Image(training_dir, recordIter,datum_name); 
    if(not im)
    {
      log_once(safe_printf("warning, couldn't open % %",training_dir,datum_name));
      return nullptr;
    }
    KeypointFn getKeypoint = [&](int num)
      {
	double u = at(nyu_labels()->train_us,recordIter,num);
	double v = at(nyu_labels()->train_vs,recordIter,num);
	return Point2d(u,v);
      };
    return NYU_Video_make_datum_raw(getKeypoint,datum_name,*im);
  }

  vector<shared_ptr<MetaData> > NYU_training_set()
  {
    string training_dir = nyu_base() + "/train/";
    Mat us = nyu_labels()->train_us;
    Mat vs = nyu_labels()->train_vs;
    Mat ds = nyu_labels()->train_ds;

    vector<string> training_color_files = find_files(training_dir, boost::regex(".*rgb_1_.*.png.*"));
    cout << "NYU::train_color_files.size = " << training_color_files.size() << " in " << training_dir << endl;
    int frame_count = training_color_files.size();
    vector<shared_ptr<MetaData> > data_set;

    // get the indices
    string indices_option = g_params.require("NYU_INDICES_OPTION");
    vector<size_t> indexes;
    if(indices_option == "THETAS_SMART")
      indexes = sort_indices_smart_thetas();
    else if(indices_option == "RANDOM")
      indexes = sort_indices_random();
    else
      throw std::runtime_error("Bad Value for NYU_INDICES_OPTION");
    //vector<size_t> indexes = sort_indices_smart_xs();
    //vector<size_t> 

    TaskBlock load_nyu("load_nyu");
    atomic<int> loadedCount(0);   
    for(auto recordIter : indexes)
    {
      load_nyu.add_callee([&,recordIter]()
			  {
			    shared_ptr<MetaData> datum(NYU_training_datum(recordIter));
			    if(not datum)
			      return;
			    progressBars->set_progress("NYU_Training_Load", loadedCount++, indexes.size()); 

			    static mutex m; lock_guard<mutex> l(m);
			    data_set.push_back(datum);
			  });
    }
    load_nyu.execute();
    progressBars->set_progress("NYU_Training_Load", 1, 1); 

    return data_set;
  }
}
