/**
 * Copyright 2013: James Steven Supancic III
 **/
#include "InverseKinematics.hpp"
#include "LibHandSynth.hpp"
#include "Poselet.hpp"
#include "Eval.hpp"
#include "HornAbsOri.hpp"
#include "Orthography.hpp"
#include "Cache.hpp"
#include <fstream>

namespace deformable_depth 
{
  using namespace libhand;
  using params::PI;  
  using namespace std;

#ifdef DD_ENABLE_HAND_SYNTH    
  PoseRegressionPoint ik_regress(BaselineDetection&det)
  {
    // define the target
    vector<Point2d> vec_target;
    Rect handBB = det.bb;
    vec_target.push_back(handBB.tl());
    vec_target.push_back(handBB.br());
    for(auto&part : det.parts)
    {
      vec_target.push_back(part.second.bb.tl());
      vec_target.push_back(part.second.bb.br());
    }
    function<vector<Point2d> (PoseRegressionPoint&rp)> kp_extractor = 
    [&](PoseRegressionPoint&rp)
    {
      vector<Point2d> feat_vec;
      feat_vec.push_back(rp.parts["HandBB"].tl());
      feat_vec.push_back(rp.parts["HandBB"].br());
      for(auto & pair : det.parts)
      {
	feat_vec.push_back(rp.parts[pair.first].tl());
	feat_vec.push_back(rp.parts[pair.first].br());
      }
      return feat_vec;
    };
    PoseRegressionPoint ik_match = libhand_regress(vec_target,kp_extractor);      
    log_file << "ik_match.parts.size() = " << ik_match.parts.size() << endl; 
    return ik_match;
  }

#endif
  
  void gen_reg()
  {
#ifdef DD_ENABLE_HAND_SYNTH    
    // seed the random number generator.
    randomize_seed();
    LibHandSynthesizer synther;
    //FileStorage reg_store(string(params::out_dir())+uuid(),FileStorage::WRITE);
    ofstream ofs(params::out_dir()+uuid()+".bin",std::ofstream::binary);
    
    //
    int oldx_size = -1, oldy_size = -1;
    for(int iter = 0; iter < 50000; ++iter)
    {
      shared_ptr< MetaData > metadata;
      do// metadata == nullptr
      {
	synther.randomize_model(Size(params::depth_hRes,params::depth_vRes));
	//synther.get_cam_spec().phi = 0;
	//synther.get_cam_spec().theta = 0;
	//synther.get_cam_spec().tilt = 
	synther.read_joint_positions();
	//metadata = synther.synth_one(true);
      } while(false);
      //assert(metadata);
      PoseRegressionPoint pt{&synther.get_scene_spec()};
      pt.jointPositionMap = synther.getJointPositionMap();
      //pt.keypoints = Poselet::keypoints(*metadata);
      pt.cam_spec  = synther.get_cam_spec();
      pt.hand_pose = synther.get_hand_pose();
      pt.world_size_randomization_sf = synther.models.begin()->second->world_size_randomization_sf;
      //pt.parts = metadata->get_positives();
      
      // binary format will be much faster
      vector<double> x;
      for(auto & joint : pt.jointPositionMap)
      {
	for(int c = 0; c < 3; ++c)
	  x.push_back(joint.second[c]);
      }
      vector<double> y;
      y.push_back(pt.cam_spec.theta);
      y.push_back(pt.cam_spec.phi);
      y.push_back(pt.cam_spec.tilt);
      y.push_back(pt.cam_spec.r);
      y.push_back(pt.hand_pose.data_size());
      for(int joint = 0; joint < LibHandJointRanges::NUM_JOINTS; joint++)
      {
	y.push_back(pt.hand_pose.bend(joint));
	y.push_back(pt.hand_pose.side(joint));
	y.push_back(pt.hand_pose.twist(joint));
      }
      y.push_back(999);
      
      // write it
      int xsize = x.size()*sizeof(decltype(x)::value_type);
      if(oldx_size != -1)
	assert(oldx_size == xsize);
      oldx_size = xsize;
      ofs.write(reinterpret_cast<const char*>(x.data()),oldx_size);
      int ysize = y.size()*sizeof(decltype(y)::value_type);
      if(oldy_size != -1)
	assert(oldy_size == ysize);
      oldy_size = ysize;
      ofs.write(reinterpret_cast<const char*>(y.data()),ysize);
      log_file << "wrote " << xsize << " " << ysize << endl;
      //reg_store << yaml_fix_key("gen_reg_"+uuid()) << pt;
    }
    
    //reg_store.release();
    cout << "DONE" << endl;
    exit(0);
#else
    throw std::runtime_error("Unsupported Method");
#endif
  }  
  
  void test_regression()
  {
#ifdef DD_ENABLE_HAND_SYNTH
    // load a metadata 
    vector<shared_ptr<MetaData> > set  = metadata_build_all(default_train_dir());
    set = filter_for_pose(set,string("_") + g_params.get_value("POSE") + ("_"));
    set = filterRightOnly(set);
    
    // synthesize 50 examples of this pose
    LibHandSynthesizer synther(params::out_dir());
    for(int iter = 0; iter < 50; ++iter)
    {
      // choose a random element of the set for this pose
      random_shuffle(set.begin(),set.end());
      shared_ptr<MetaData> metadata = set[0];
      
      // extract the keypoints
      vector<Point2d> keypoints = Poselet::keypoints(*metadata);
      shared_ptr<ImRGBZ> im = metadata->load_im();
      
      // perform a regression to find a set of libhand parameters
      PoseRegressionPoint reg = libhand_regress(keypoints,
						[](PoseRegressionPoint&rp)
						{
						  return rp.keypoints;
						});
      
      // render a hand with libhand using said parameters
      Size bgSize(params::depth_hRes,params::depth_vRes);
      synther.set_model(reg,bgSize);
      shared_ptr<MetaData> match_metadata;
      while(!(match_metadata = synther.synth_one(true)))
	synther.randomize_background(bgSize);
      
      // create a visualization of the match
      Mat vis_query = im->RGB;
      Mat vis_answer = match_metadata->load_im()->RGB;
      log_im("reg_query_result",horizCat(vis_query,vis_answer));      
      
      // generate a perturbation
      synther.set_model(reg,bgSize);
      synther.perturb_model();
      synther.randomize_background(bgSize);
      shared_ptr<MetaData> perturbation_metadata;
      while(!(perturbation_metadata = synther.synth_one(false)))
	synther.randomize_background(bgSize);
    }
#else
    throw std::runtime_error("Unsupported Method");
#endif
  }    

  void write(cv::FileStorage&fs, std::string&, const libhand::HandCameraSpec&cam_spec)
  {
    fs << "{";
    fs << "r" << cam_spec.r;
    fs << "tilt" << cam_spec.tilt;
    fs << "theta" << cam_spec.theta;
    fs << "phi" << cam_spec.phi;
    fs << "}";
  }
  void read(const FileNode&node, HandCameraSpec&cam_spec, HandCameraSpec)
  {
    node["r"] >> cam_spec.r;
    node["tilt"] >> cam_spec.tilt;
    node["theta"] >> cam_spec.theta;
    node["phi"] >> cam_spec.phi;
  }
  void write(FileStorage& fs, string&, const PoseRegressionPoint& pt)
  {
    string s;
    
    fs << "{";
    fs << "keypoints" << pt.keypoints;
    fs << "cam_spec"; write(fs,s,pt.cam_spec);
    fs << "hand_pose" << pt.hand_pose;
    fs << "world_size_randomization_sf" << pt.world_size_randomization_sf;
    fs << "parts"; write(fs,pt.parts);
    fs << "jointPositionMap"; write(fs,pt.jointPositionMap);
    fs << "}";
  }
  void read(const FileNode& node, PoseRegressionPoint& pt, PoseRegressionPoint)
  {
    node["keypoints"] >> pt.keypoints;
    read(node["cam_spec"],pt.cam_spec,HandCameraSpec());
    node["hand_pose"] >> pt.hand_pose; 
    log_once(printfpp("read handpose with %d joints",(int)pt.hand_pose.num_joints()));
    node["world_size_randomization_sf"] >> pt.world_size_randomization_sf;
    read(node["parts"],pt.parts);
    // old files may not have this
    if(!node["jointPositionMap"].empty())
      read(node["jointPositionMap"],pt.jointPositionMap);
    //cout << "loaded a regression point" << endl;
  }

#ifdef DD_ENABLE_HAND_SYNTH                
  PoseRegressionPoint libhand_regress_yml_format(
    const vector<Point2d>&vec_target,
    function<vector<Point2d> (PoseRegressionPoint&rp)> extract_saved_keypoints)
  {
    PoseRegressionPoint best_pt;
    double min_dist = inf;
    
    // load the regression db
    static Cache<shared_ptr<FileStorage>> storage_cache;
    shared_ptr<FileStorage> reg_store_ptr = storage_cache.get("default",
      []()
      {
	//string regression_database = "/home/jsupanci/Dropbox/data/vis_reg/5k_wild_camera.yml";
	string regression_database = "/home/jsupanci/Dropbox/data/vis_reg_big/vis_reg_0.yaml";
	//string regression_database = "/home/jsupanci/Dropbox/data/vis_reg/vis_reg.yaml";
	cout << "loading reg_store" << endl;
	shared_ptr<FileStorage> reg_store = make_shared<FileStorage>(regression_database,FileStorage::READ);
	assert(reg_store->isOpened());
	log_file << "reg_store loaded" << endl;	
	return reg_store;
      });
    FileStorage&reg_store = *reg_store_ptr;
    
    //static mutex m; lock_guard<mutex> l(m);
    vector<double> weights(vec_target.size(),1);
    int num = 0;
    for(auto iter = reg_store.root().begin(); iter != reg_store.root().end(); ++iter, ++num)
    {
      const auto&node = *iter;
      //cout << "reading: " << node.name() << endl;
      PoseRegressionPoint cur_pt; node >> cur_pt;
      // check the parts
      Rect handBB = cur_pt.parts.at("HandBB");
      bool bad_pt = false;
      for(auto & pair : cur_pt.parts)
      {
	Rect partBB = pair.second;
	if(partBB.size().area() < 0 || partBB.size().area() > 320*240)
	  bad_pt = true;
	if(partBB.width > 10000 || partBB.height > 10000)
	  bad_pt = true;
	if(partBB.x <= -2.14748e+09)
	  bad_pt = true;
	if((partBB & handBB) != partBB)
	  bad_pt = true;
      }
      if(bad_pt)
	continue;
      
      // get the keypoint vectors from the semantically meaningful
      // keypoint structures.
      vector<Point2d> vec_cur = extract_saved_keypoints(cur_pt);
      
      // compute the distance
      double ssd;
      assert(vec_cur.size() == vec_target.size());
      ssd = Poselet::min_dist(vec_cur,vec_target,weights).min_ssd;
      
      //cout << printfpp("ssd: %f for %d",ssd,num) << endl;
      if(ssd <= min_dist)
      {
	//cout << "reduced min dist to: " << min_dist << endl;
	min_dist = ssd;
	best_pt = cur_pt;
      }
    }
    
    return best_pt;
  }
    
  PoseRegressionPoint libhand_regress(
    const vector<Point2d>&vec_target,
    function<vector<Point2d> (PoseRegressionPoint&rp)> extract_saved_keypoints)
  {
    return libhand_regress_yml_format(vec_target,extract_saved_keypoints);
  }  

#endif

  ///
  /// SECTION: 3D InverseKinematics
  /// 
#ifdef DD_ENABLE_HAND_SYNTH
  void IK_Unpack_maps(
    vector<Vec3d>&p1, vector<Vec3d>&p2,
    const map< string, Vec3d >&jp1,
    const map< string, Vec3d >&jp2,
    bool orthographic = true)
  {
    set<string> intersection;
    for(auto && pair : jp1)
      if(jp2.find(pair.first) != jp2.end())
	if(!boost::regex_match(pair.first,boost::regex(".*Bone_.*")))
	  intersection.insert(pair.first);
    
    DepthCamera camera;
    Size ortho_size(640,480);
      
    for(auto && kp_name : intersection)
    {
      if(orthographic)
      {
	p1.push_back(ortho2cm(jp1.at(kp_name),ortho_size,camera));
	p2.push_back(ortho2cm(jp2.at(kp_name),ortho_size,camera));
      }
      else
      {
	p1.push_back(jp1.at(kp_name));
	p2.push_back(jp2.at(kp_name));	
      }
    }    
  }
  
  PoseRegressionPoint IK_Match_nn(
    const map< string, Vec3d >& joint_positions,
    AbsoluteOrientation&abs_ori,LibHandSynthesizer&refiner)  
  {
    // load the regression db
    static FileStorage reg_store;
    if(!reg_store.isOpened())
    {
      mutex m; unique_lock<mutex> l(m);
      string regression_database = "/home/jsupanci/Dropbox/data/vis_reg/5k_solid_camera.yml";
      //string regression_database = "/home/jsupanci/Dropbox/data/vis_reg_3d/50k.yml";
      //string regression_database = "/home/jsupanci/Dropbox/data/small_rectified_reg_3d/small.yml";
      cout << "loading reg_store" << endl;
      reg_store.open(regression_database,FileStorage::READ);
      assert(reg_store.isOpened());
      cout << "reg_store loaded" << endl;
    }    
    
    // now find the closest match
    double min_dist = inf;
    PoseRegressionPoint best_pt;
    for(auto iter = reg_store.root().begin(); iter != reg_store.root().end(); ++iter)
    {
      // read the current point
      const auto&node = *iter;
      cout << "reading: " << node.name() << endl;
      PoseRegressionPoint cur_pt; node >> cur_pt;
      
      // compute the distance
      vector<Vec3d> p1, p2;
      IK_Unpack_maps(p1,p2,joint_positions,cur_pt.jointPositionMap);
      AbsoluteOrientation cur_abs_ori = distHornAO(p2,p1);
      
      // update the min
      if(cur_abs_ori.distance < min_dist)
      {
	min_dist = cur_abs_ori.distance;
	abs_ori = cur_abs_ori;
	best_pt = cur_pt;
      }
    }  
    
    return best_pt;
  }
    
  double IK_Viewport_distance(
    const map< string, Vec3d >& joint_positions,
    const map< string, Vec3d >& alternative
 			    )
  {
    vector<Vec3d> p1, p2;
    IK_Unpack_maps(p1,p2,joint_positions,alternative,false);
    //vector<Point2d> xs, ys;
    //for(Vec3d v : p1)
      //xs.push_back(Point2d(v[0],v[1]));
    //for(Vec3d v : p2)
      //ys.push_back(Point2d(v[0],v[1]));
    return Poselet::min_dist_simple(p2,p1);    
  }
    
  void IK_Match_Viewpoint(
    const map< string, Vec3d >& joint_positions,
    AbsoluteOrientation&abs_ori,
    LibHandSynthesizer&refiner,
    PoseRegressionPoint&best_pt)  
  {
    // load the regression db
    static FileStorage reg_store;
    if(!reg_store.isOpened())
    {
      mutex m; unique_lock<mutex> l(m);
      string regression_database = "/home/jsupanci/Dropbox/data/vis_reg/50k_wild_camera.yml";
      //string regression_database = "/home/jsupanci/Dropbox/data/vis_reg_3d/50k.yml";
      //string regression_database = "/home/jsupanci/Dropbox/data/small_rectified_reg_3d/small.yml";
      cout << "loading reg_store" << endl;
      reg_store.open(regression_database,FileStorage::READ);
      assert(reg_store.isOpened());
      cout << "reg_store loaded" << endl;
    }    
    
    // find the best matching orientation
    double min_dist = inf;
    vector<double> best_dir;
    for(auto iter = reg_store.root().begin(); iter != reg_store.root().end(); ++iter)
    {
      // read the current point
      const auto&node = *iter;
      cout << "reading: " << node.name() << endl;
      PoseRegressionPoint cur_pt; node >> cur_pt;
      
      // compute the distance
//       vector<double> sample_ori,query_ori;
//       try
//       {
// 	sample_ori = convert(
// 	  cur_pt.jointPositionMap.at("finger3joint1") - cur_pt.jointPositionMap.at("metacarpals"));
// 	query_ori = convert(
// 	  joint_positions.at("finger3joint1") - joint_positions.at("metacarpals"));
//       }
//       catch(std::out_of_range)
//       {
// 	for(auto&pair : cur_pt.jointPositionMap)
// 	  cout << pair.first << endl;
// 	for(auto&pair : joint_positions)
// 	  cout << pair.first << endl;
// 	assert(false);
//       }
//       double s_norm = norm_l2(sample_ori);sample_ori /= s_norm;
//       double q_norm = norm_l2(query_ori);query_ori  /= q_norm;
//       double dist = dot(sample_ori,query_ori);
      
      double dist = IK_Viewport_distance(joint_positions,cur_pt.jointPositionMap);
      
      // update the min
      if(dist < min_dist)
      {
	min_dist = dist;
	best_pt.cam_spec = cur_pt.cam_spec;
	//best_dir = sample_ori;
      }
    }  
    
    log_file << printfpp("Best Dist = %f BestCamSpec = %f %f %f",
		     min_dist, best_pt.cam_spec.theta,
		     best_pt.cam_spec.phi,
		     best_pt.cam_spec.tilt
		    ) << endl;
    //log_file << "best direction: " << best_dir << endl;
  }    
    
  void IK_Match_Viewpoint_local_search(
    const map< string, Vec3d >& joint_positions,
    AbsoluteOrientation&abs_ori,
    LibHandSynthesizer&refiner,
    PoseRegressionPoint&best_pt)      
  {
    double min_dist = inf;
    auto MAX_TIME = std::chrono::seconds(5);
    auto start = std::chrono::system_clock::now();
    int iter = 0;
    while(std::chrono::system_clock::now() - start < MAX_TIME)
    {
      ++iter;
      // chagne the camera
      libhand::HandCameraSpec cam_spec =  best_pt.cam_spec;
      
      if(iter > 1)
      {
	cam_spec.theta += (rand()%2)?.05:-.05;
	cam_spec.phi += (rand()%2)?.05:-.05;
	cam_spec.tilt += (rand()%2)?.05:-.05;
      }
      
      // render
      Size bgSize(params::depth_hRes,params::depth_vRes);
      refiner.set_cam_spec(cam_spec);
      refiner.set_model(best_pt,bgSize,
			LibHandSynthesizer::COMP_FLAG_WORLD_SIZE|LibHandSynthesizer::COMP_FLAG_HAND_POSE);
      refiner.read_joint_positions();
      
      // comptue the dist
      double dist = IK_Viewport_distance(joint_positions,refiner.getJointPositionMap());
      if(dist < min_dist)
      {
	log_file << printfpp("viewpoint dist update: %f => %f",min_dist,dist) << endl;
	min_dist = dist;
	best_pt.cam_spec = cam_spec;
      }
    }
  }
  
  void IK_Match_local_search(
    const map< string, Vec3d >& joint_positions,
    AbsoluteOrientation&abs_ori,
    LibHandSynthesizer&refiner,
    PoseRegressionPoint&best_pt)  
  {
    // now, try to do some incremental refinement.
    auto MAX_TIME = std::chrono::seconds(5);
    auto start = std::chrono::system_clock::now();
    while(std::chrono::system_clock::now() - start < MAX_TIME)
    {
      // copy and permute
      libhand::FullHandPose hand_pose = best_pt.hand_pose;
      int joint_num = rand() % LibHandSynthesizer::IDX_MAX_VALID_FINGER_JOINTS; //hand_pose.num_joints();
      function<float& (int)> change_joint;
      string op;
      switch(rand()%1)
      {
	case 0:
	  change_joint = [&](int index)->float&{return hand_pose.bend(index); };
	  op = "bend";
	  break;
	case 1:
	  change_joint = [&](int index)->float&{return hand_pose.side(index); };
	  op = "side";
	  break;
	case 2:
	  change_joint = [&](int index)->float&{return hand_pose.twist(index); };
	  op = "twist";
	  break;
	default:
	  assert(false);
      }
      if(rand()%2)
	change_joint(joint_num) = clamp<float>(-PI/3,change_joint(joint_num)+.05,PI/8);
      else
	change_joint(joint_num) = clamp<float>(-PI/3,change_joint(joint_num)-.05,PI/8);
      
      // check distance
      Size bgSize(params::depth_hRes,params::depth_vRes);
      refiner.set_hand_pose(hand_pose);
      refiner.set_model(best_pt,bgSize,
			LibHandSynthesizer::COMP_FLAG_CAM|LibHandSynthesizer::COMP_FLAG_WORLD_SIZE);
      refiner.read_joint_positions();
      //log_im("possibleRefinement",refiner.render_only());
      //refiner.render_only();
      vector<Vec3d> p1, p2;
      IK_Unpack_maps(p1,p2,joint_positions,refiner.getJointPositionMap());
      AbsoluteOrientation cur_abs_ori = distHornAO(p2,p1);
      
      // update if better
      if(cur_abs_ori.distance < abs_ori.distance)
      {
	string message = printfpp(
	  "Local Search Updated joint(%s) = %d cost: (%f => %f)",
	  op.c_str(),joint_num,abs_ori.distance,cur_abs_ori.distance);	
	log_file << "q_old = " << abs_ori.quaternion << endl;
	log_file << "q_new = " << cur_abs_ori.quaternion << endl;
	
	abs_ori = cur_abs_ori;
	best_pt.hand_pose = hand_pose;

	cout << message << endl;
	log_file << message << endl;
      }
    }
  }
  
  PoseRegressionPoint IK_Match(
    const map< string, Vec3d >& joint_positions,
    AbsoluteOrientation&abs_ori,LibHandSynthesizer&refiner)
  {
    refiner.set_flip_lr(false);
    
    PoseRegressionPoint best_pt = IK_Match_nn(joint_positions,abs_ori,refiner);
    IK_Match_local_search(joint_positions,abs_ori,refiner,best_pt);
    IK_Match_Viewpoint(joint_positions,abs_ori,refiner,best_pt);
    IK_Match_Viewpoint_local_search(joint_positions,abs_ori,refiner,best_pt);
    return best_pt;
  }
  
  bool IKBinaryDatabase::hasNext()
  {
    return ifs;
  }

  IKBinaryDatabase::IKBinaryDatabase(LibHandRenderer& renderer,string database_filename) : 
    ifs(database_filename,ifstream::binary),
    renderer(renderer)
  {
    static mutex m; lock_guard<mutex> l(m);
    init_jpm = renderer.get_jointPositionMap();
  }

  IKBinaryDatabase::XYZ_TYR_Pair IKBinaryDatabase::next_record()
  {
    XYZ_TYR_Pair record;
    record.valid = false;

    // read the candidate
    // read x
    vector<double> x(init_jpm.size()*3);
    int xsize = x.size()*sizeof(decltype(x)::value_type);
    ifs.read(reinterpret_cast<char*>(x.data()),xsize);
    // read y
    vector<double> y(6 + 3*LibHandJointRanges::NUM_JOINTS);
    int ysize = y.size()*sizeof(decltype(y)::value_type);
    ifs.read(reinterpret_cast<char*>(y.data()),ysize);
    record.joint_position_map = init_jpm;
    int xPtr = 0;
    for(auto && jp : record.joint_position_map)
      for(int c = 0; c < 3; ++c)
      {
	jp.second[c] = x[xPtr++];
	if(renderer.get_flip_lr() && c == 0)
	  jp.second[c] = params::hRes - jp.second[c];
      }
    //cout << "read bytes : " << xsize << " " << ysize << endl;
    if(not ifs)
      return record;

    // read the angles
    record.cam_spec = renderer.get_cam_spec();
    record.hand_pose = renderer.get_hand_pose();
    record.cam_spec.theta = y[0];
    record.cam_spec.phi = y[1];
    record.cam_spec.tilt = y[2];
    record.cam_spec.r = y[3];
    if(y[4] != record.hand_pose.data_size())
      return record;
    require_equal<size_t>(y[4],record.hand_pose.data_size());
    int dIter = 0;
    for(int joint = 0; joint < LibHandJointRanges::NUM_JOINTS; joint++)
    {
      record.hand_pose.bend(joint) = y[5 + dIter++];
      record.hand_pose.side(joint) = y[5 + dIter++];
      record.hand_pose.twist(joint) = y[5 + dIter++];
      //cout << "Joint " << joint << " " << best_full_hand_pose.bend(joint) << " " 
      //   << best_full_hand_pose.side(joint) << " " << best_full_hand_pose.twist(joint) << endl;
      assert(5+dIter -1 < y.size());
    }
    assert(y[5 + dIter] == 999);

    record.valid = true;
    return record;
  }

  ///
  /// SECTUIN: incremental 2D IK
  /// 
  static vector<string> get_ik_databases()
  {
    return find_files("data/bin_ik/", boost::regex(".*bin"));
  }

  vector<PoseRegressionPoint> ik_regress_thresh(LibHandRenderer& renderer,
						BaselineDetection&det,double max_cm_thresh)
  {
    vector<PoseRegressionPoint> matches;
    
    TaskBlock ik_regress("ik_regress");
    vector< string > databases = get_ik_databases();
    atomic<long> candidates(0);
    for(string database_filename : databases)
    {
      ik_regress.add_callee([&,database_filename]()
      {
	IKBinaryDatabase ik_db(renderer,database_filename);
	for(int iter = 0; ik_db.hasNext(); ++iter)
	{
	  IKBinaryDatabase::XYZ_TYR_Pair record = ik_db.next_record();
	  if(not record.valid)
	    continue;
	  candidates++;

	  vector<Point2d> src_pts, dst_pts;
	  vector<double> weights;
	  for(auto && part : det.parts)
	  {
	    throw std::runtime_error("TODO: unsupported :-(");
	    // src_pts.push_back(rectCenter(part.second));
	    // pair<string,string> pt_names = joints4bone(part.first);
	    // Vec3d dst1 = record.joint_position_map.at();
	    // Vec3d dst2 = record.joint_position_map.at();
	    // dst_pts.push_back(Point2d((dst1.x + dst2.x)/2,(dst1.y + dst2.y)/2));
	    // weights.push_back(1);
	  }

	  // compute cost
	  Poselet::Procrustean2D cost = Poselet::min_dist(src_pts,dst_pts,weights);

	  if(cost.max_se < max_cm_thresh) 
	  {
	    PoseRegressionPoint pt;
	    pt.jointPositionMap = record.joint_position_map;
	    pt.cam_spec = record.cam_spec;
	    pt.hand_pose = record.hand_pose;
	    matches.push_back(pt);
	  }
	}	
      });
    }
    ik_regress.execute();

    log_file << safe_printf("--ik_regress_thresh selected % of % candidates",matches.size(),candidates++) << endl;

    return matches;
  }

  Poselet::Procrustean2D incremental2DIk_global(LibHandRenderer& renderer, const map< string, Vec3d >& joint_positions)
  {
    libhand::HandCameraSpec best_cam_spec = renderer.get_cam_spec();
    libhand::FullHandPose best_full_hand_pose = renderer.get_hand_pose();
    //int best_lr_flip = renderer.get_flip_lr();
    renderer.render();    
    auto init_jpm = renderer.get_jointPositionMap();
    Poselet::Procrustean2D min_cost = Poselet::min_dist(joint_positions,
							init_jpm);
    log_file << "init min_cost = " << min_cost.min_ssd << endl;
    vector< string > databases = get_ik_databases();
    
    TaskBlock ik_regress("ik_regress");
    for(string database_filename : databases)
    {
      ik_regress.add_callee([&,database_filename]()
      {
	IKBinaryDatabase ik_db(renderer,database_filename);
	for(int iter = 0; ik_db.hasNext(); ++iter)
	{
	  IKBinaryDatabase::XYZ_TYR_Pair record = ik_db.next_record();
	  if(not record.valid)
	    continue;

	  // compute cost
	  Poselet::Procrustean2D cost = Poselet::min_dist(
	    joint_positions,record.joint_position_map);
	  if(is_egocentric(record.joint_position_map))
	    cost.min_ssd = inf;
	  log_file << "new min_cost: " << cost.min_ssd << endl;
	  if(cost < min_cost && 1/8.0 < cost.s && cost.s < 8) 
	  {
	    static mutex m; lock_guard<mutex> l(m);
	    if(cost < min_cost)
	    {
	      log_file << printfpp("min_cost %f => %f",min_cost,cost) << endl;
	      min_cost = cost;
	      best_cam_spec = record.cam_spec;
	      best_full_hand_pose = record.hand_pose;
	    }
	  }
	}	
      });
    }
    ik_regress.execute();
    
    // return the best
    renderer.get_hand_pose() = best_full_hand_pose;
    renderer.get_cam_spec() = best_cam_spec;
    //renderer.set_flip_lr(best_lr_flip);
    renderer.render();
    renderer.read_joint_positions();
    min_cost = Poselet::min_dist(
	joint_positions,renderer.get_jointPositionMap());
    //min_cost.s = clamp<double>(.3,min_cost.s,3);
    return min_cost;
  }
  
  struct IKParticle
  {
    libhand::HandCameraSpec cur_cam_spec;
    libhand::FullHandPose cur_pose_spec;
    libhand::HandCameraSpec best_cam_spec;
    libhand::FullHandPose best_pose_spec;
    vector<double> velocity;
    double best_dist, cur_dist;
    
    float& get_i(int index,libhand::HandCameraSpec&cam,libhand::FullHandPose&hand)
    {
      if(index == 0)
	return cam.phi;
      else if(index == 1)
	return cam.theta;
      else if(index == 2)
	return cam.tilt;
      else if(index == 3)
	return cam.r;
      else
	return *(hand.begin() + index - 4);
    }
    
    float& get_cur_i(int index)
    {
      return get_i(index,cur_cam_spec,cur_pose_spec);
    }
    
    float& get_best_i(int index)
    {
      return get_i(index,best_cam_spec,best_pose_spec);
    }
  };  
  
  static constexpr double omega = .5, phi_p = .01, phi_g = .01;
  
  void update_particle(
    IKParticle&particle,
    LibHandRenderer& renderer, 
    const map< string, Vec3d >& joint_positions,
    IKParticle&g,double dist0)
  {
    // take a step/move the particle
    double rp = sample_in_range(0,1);
    double rg = sample_in_range(0,1);
    for(int comp_iter = 0; comp_iter < particle.velocity.size(); ++comp_iter)
    {
      float&pid = particle.get_best_i(comp_iter);
      float&gid = g.get_best_i(comp_iter);
      float&xid = particle.get_cur_i(comp_iter);
      double&vel_comp = particle.velocity[comp_iter];
      
      // move
      vel_comp = omega * vel_comp + phi_g * rp * (pid - xid) + phi_g * rg * (gid - xid);
      xid += vel_comp;
    }
    
    // compute the score
    renderer.get_hand_pose() = particle.cur_pose_spec;
    renderer.get_cam_spec() = particle.cur_cam_spec;
    renderer.read_joint_positions();
    particle.cur_dist = Poselet::min_dist(joint_positions,renderer.get_jointPositionMap()).min_ssd;
    
    if(particle.cur_dist < particle.best_dist)
    {
      particle.best_cam_spec = particle.cur_cam_spec;
      particle.best_pose_spec = particle.cur_pose_spec;
      static mutex m; lock_guard<mutex> l(m);
      if(particle.cur_dist < g.best_dist)
      {
	g.best_cam_spec = g.cur_cam_spec = particle.cur_cam_spec;
	g.best_pose_spec = g.cur_pose_spec = particle.cur_pose_spec;
	g.best_dist = g.cur_dist = particle.cur_dist;
	cout << printfpp("incremental2DIk_local %f => %f",dist0,g.best_dist) << endl;
      }	
    }    
  }
  
  Poselet::Procrustean2D incremental2DIk_local(LibHandRenderer& renderer, const map< string, Vec3d >& joint_positions)
  {
    double dist0 = Poselet::min_dist(joint_positions,renderer.get_jointPositionMap()).min_ssd;
    
    // initialize the swarm
    IKParticle p0{
      renderer.get_cam_spec(),renderer.get_hand_pose(),
      renderer.get_cam_spec(),renderer.get_hand_pose(),
      vector<double>(4 + renderer.get_hand_pose().data_size()),
      dist0,dist0
    };
    IKParticle g = p0; // swarm optimum
    vector<IKParticle> particles(500,p0);
    for(auto & particle : particles)
    {
      for(auto & vel_comp : particle.velocity)
	vel_comp = sample_in_range(0,.1*params::PI);
    }
    
    for(int pso_iter = 0; pso_iter < 100; ++pso_iter)
    {
      //TaskBlock pso("pso");
      int group_size = particles.size()/params::cpu_count();
      for(int group_iter = 0; group_iter < params::cpu_count(); ++group_iter)
      {
	//pso.add_callee([&,group_iter]()
	{
	  int first_particle = group_iter * group_size;
	  int last_particle = (group_iter+1) * group_size;
	  for(int particle_iter = first_particle; 
	      particle_iter < last_particle; ++particle_iter)
	  {
	    auto & particle = particles[particle_iter];
	    update_particle(particle,renderer, joint_positions,g,dist0);
	  }	  
	}//);
      }
      //pso.execute();
    }
    
    renderer.get_hand_pose() = g.best_pose_spec;
    renderer.get_cam_spec() = g.best_cam_spec;
    renderer.render();
    auto md = Poselet::min_dist(joint_positions,renderer.get_jointPositionMap());
    log_file << printfpp("incremental2DIk_local %f => %f",dist0,md.min_ssd) << endl;
    return md;
  }
  
  Poselet::Procrustean2D incremental2DIk(LibHandRenderer& renderer, const map< string, Vec3d >& joint_positions)
  {
    Poselet::Procrustean2D gbl_dist = incremental2DIk_global(renderer, joint_positions);
    Poselet::Procrustean2D lcl_dist = incremental2DIk_local(renderer, joint_positions);
    return lcl_dist;
  }
#endif
}

#ifndef DD_ENABLE_HAND_SYNTH
#include <opencv2/opencv.hpp>

namespace libhand
{
  using namespace std;
  using namespace cv;

  // provide implementations of these functions when libhand isn't available (eg on a cluster)
  void read(const cv::FileNode&fn, libhand::FullHandPose&hand_pose, libhand::FullHandPose)
  {
    fn["data_"] >> hand_pose.data_;
    if(!fn["num_joints_"].empty())
      fn["num_joints_"] >> hand_pose.num_joints_;
    else
      hand_pose.num_joints_ = 18;
    //cout << "loaded hand_pose with num_joints_ = " << hand_pose.num_joints_ << endl;
  }
  void write(cv::FileStorage&fs, string&, const FullHandPose&hand_pose)
  {
    fs << "{";
    fs << "data_" << hand_pose.data_;
    fs << "num_joints_" << hand_pose.num_joints_;
    fs << "}";
  }
}
#endif
