/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "MetaDataKITTI.hpp"
#include "util.hpp"
#include "Velodyne.hpp"
#include "ThreadPoolCXX11.hpp"
#include <functional>
#include "YML_Data.hpp"
#include <boost/algorithm/string/trim.hpp>

namespace deformable_depth
{
  static Mat padTo4x4(Mat m, bool eye = true)
  {
    assert(m.rows <= 4 && m.cols <= 4);
    Mat m4x4 = eye?Mat::eye(4,4,m.type()):Mat::zeros(4,4,m.type());
    
    m.copyTo(m4x4(Rect(Point2i(0,0),m.size())));
    //cout << m4x4.size() << endl;
    
    return m4x4;
  }  
  
  Mat KITTI_Calibration::PMat(int cam) const
  {
    Mat P;
    if(cam == 2)
      P = P2;
    else if(cam == 1)
      P = P1;
    else
      assert(false);
    return P;
  }

  Vec3d KITTI_Calibration::unproject(Vec3d pt, int cam) const
  {
    Mat P = PMat(cam);
    Mat registration_matrix = P * R0_Rect * Tr_velo_to_cam;
    Mat unreg_matrix = padTo4x4(registration_matrix).inv();
    double z = pt[2]/100;
    Mat unproj = (unreg_matrix * Mat(Vec4d(pt[0]*z,pt[1]*z,z,1)));    
    cout << "unproj = " << unproj << endl;
    
    Vec3d world = unhomo4(unproj);
    return world;
  }
  
  Vec3d KITTI_Calibration::project(Vec3d pt,int cam) const
  {
    Mat P = PMat(cam);
    
    Mat registration_matrix = P * R0_Rect * Tr_velo_to_cam;
    Mat proj = (registration_matrix * Mat(Vec4d(pt[0],pt[1],pt[2],1)));
    assert(proj.type() == DataType<double>::type);
    assert(proj.size().area() == 3);    
    //double x = .25*proj.at<double>(0) / proj.at<double>(3) + depth.cols/2;
    //double y = .25*proj.at<double>(1) / proj.at<double>(3) + depth.rows/2;
    double z = proj.at<double>(2);
    double x = proj.at<double>(0) / z;
    double y = proj.at<double>(1) / z;
    return Vec3d(x,y,100*z);
  }
  
  Mat formDepth_lidar(
    KITTI_Calibration&calib,VelodyneData&point_cloud,Mat&RGB1,Mat&RGB2,bool fillHoles)
  {
    Mat depth(RGB1.rows,RGB1.cols,DataType<float>::type,Scalar::all(inf));
    Mat isInvalid(RGB1.rows,RGB1.cols,DataType<float>::type,1);
    
    // project the velodyne data into the depth image
    // x = P2 * R0_rect * Tr_velo_to_cam * y
    
    //cout << "registration_matrix: " << registration_matrix << endl;
    for(int iter = 0; iter < point_cloud.getPoints().size(); ++iter)
    {
      // read the record
      const Vec3d& pt = point_cloud.getPoints()[iter];
      double r = point_cloud.getReflectances()[iter];
      
      Vec3d proj = calib.project(pt);
      double x = proj[0];
      double y = proj[1];
      double z = proj[2];
      
      if(0 <= x && x < RGB1.cols && 0 <= y && y < RGB1.rows && z > 0)
      {
	//cout << "projected: " << pt << " => " << Vec3d(x,y,z) << " r = " << r << endl;	
	depth.at<float>(y,x) = z;
	isInvalid.at<float>(y,x) = 0;
      }
    }
    //cout << "P2 " << calib.P2 << endl;
    //cout << "R0_Rect " << calib.R0_Rect << endl;
    //cout << "Tr_velo_to_cam " << calib.Tr_velo_to_cam << endl;
    //cout << "registration_matrix " << registration_matrix << endl;
    
    //cout << "resolution: " << depth.size() << endl;
    //imageeq("raw depth",depth,true,false);
    if(fillHoles)
    {
      depth = fillDepthHoles(depth,isInvalid,10);
    }
    return depth;
  }
    
  map<string,AnnotationBoundingBox> KITTI_GT(string filename)
  {
    ifstream input_file;
    {
      static mutex m; lock_guard<mutex> l(m);
      //log_once(printfpp("KITTI_GT Opening \"%s\"",filename.c_str()));
      input_file.open(filename);
      if(!input_file.is_open())
      {
	cout << "couldn't open: " << filename << endl;
	log_once(strerror(errno));
	assert(false);
      }
    }
    map<string,AnnotationBoundingBox> annotations;
    
    int parsed_lines = 0;
    while(input_file)
    {
      // get and parse the line
      string line; getline(input_file,line);
      boost::trim(line);
      if(line == "")
	continue;
      istringstream line_stream(line);
      // read type and generate name
      string name; line_stream >> name;
      name += uuid();
      // read trunction, occlusion and alpha
      float truncation; line_stream >> truncation;
      int occluded; line_stream >> occluded;
      float alpha; line_stream >> alpha;
      // read bounding box
      float left, top, right, bottom;
      line_stream >> left >> top >> right >> bottom;
      parsed_lines++;
      annotations[name] = 
	AnnotationBoundingBox(Rect(Point(left,top),Point(right,bottom)),
			      std::min<float>(1.0-occluded/2.0,1-truncation));
      // read the dimensions
      float height, width, length;
      line_stream >> height >> width >> length;
      // read the location
      float x,y,z;
      line_stream >> x >> y >> z;
      // read the rotation
      float rotation_y;
      line_stream >> rotation_y;
      // read the score
      float score;
      if(!line_stream)
      {
	cout << "failed to parse : " << line << endl;
	assert(false);
      }
      line_stream >> score;
      annotations[name].confidence = score;
    }
    
    require_equal<size_t>(parsed_lines,annotations.size());
    ostringstream oss;
    oss << "KITTI got " << annotations.size() << " annotations from " << filename;
    log_once(oss.str().c_str());
    
    return annotations;
  }
  
  void KITTI_Demo()
  {
    TaskBlock KITTIDemoTasks("KITTIDemoTasks");
    for(int iter = 30; iter < 200; ++iter)
      KITTIDemoTasks.add_callee([&,iter]()
      {
	new MetaDataKITTI(iter,true);
      });
    KITTIDemoTasks.execute();
  }  
  
  template<bool training, int min_frame, int max_frame>
  vector< shared_ptr< MetaData > > KITTI_Datasubset()
  {
    // critical section
    static mutex m; unique_lock<mutex> l(m);
    
    // try to return the already generated data
    static vector<shared_ptr< MetaData >> all_data;
    if(!all_data.empty())
      return all_data;
    
    // generate the data
    for(int iter = min_frame; iter <= max_frame; ++iter)
      all_data.push_back(shared_ptr<MetaData>(new MetaDataKITTI(iter,training)));
    
    // "randomize" it 
    std::mt19937 prng(1986); // use known seed to keep the same sequence
    std::shuffle(all_data.begin(),all_data.end(),prng);
    
    return all_data;
  }  
  
  vector< shared_ptr< MetaData > > KITTI_default_data()
  {
    return KITTI_Datasubset<true,0,7480>();
  }
  
  vector<shared_ptr<MetaData> > KITTI_validation_data()
  {
    constexpr int min_frame = 0, max_frame = 7517;
    vector<shared_ptr<MetaData> > data = KITTI_Datasubset<false,min_frame,max_frame>();
    log_once(printfpp("KITTI_validation_data = %d",(int)data.size()));
    require_equal<size_t>(data.size(),max_frame - min_frame + 1);
    return data;
  }
  
  vector< shared_ptr< MetaData > > KITTI_default_test_data()
  {
    vector< shared_ptr< MetaData > > all_data = KITTI_default_data();
    return vector< shared_ptr< MetaData > >(all_data.begin()+6001,all_data.end());    
  }

  vector< shared_ptr< MetaData > > KITTI_default_train_data()
  {
    vector< shared_ptr< MetaData > > all_data = KITTI_default_data();
    return vector< shared_ptr< MetaData > >(all_data.begin(),all_data.begin()+6000);
  }
  
  ///
  /// SECTION: KITTI_Calibration
  ///
  static Mat readMatrixLine(string name, ifstream&ifs,int rows, int cols)
  {
    // check the header name matches to catch parse errors
    string header; ifs >> header;
    require_equal<string>(header,name + ":");
    
    // read the data elements
    vector<double> data(cols*rows,qnan);
    for(int iter = 0; iter < cols*rows; ++iter)
    {
      ifs >> data[iter];
      assert(goodNumber(data[iter]));
    }
    //cout << name << data << endl;
    
    //return Mat(data).reshape(0,rows);
    return Mat(data,true).reshape(0,rows);
  }
    
  KITTI_Calibration::KITTI_Calibration(string filename)
  {
    ifstream ifs(filename);
    assert(ifs.is_open());
    P0 = readMatrixLine("P0",ifs,3,4);
    P1 = readMatrixLine("P1",ifs,3,4);
    P2 = readMatrixLine("P2",ifs,3,4);
    P3 = readMatrixLine("P3",ifs,3,4);    
    R0_Rect = padTo4x4(readMatrixLine("R0_rect",ifs,3,3));
    Tr_velo_to_cam = padTo4x4(readMatrixLine("Tr_velo_to_cam",ifs,3,4));
    Tr_imu_to_velo = padTo4x4(readMatrixLine("Tr_imu_to_velo",ifs,3,4));
    
    // define camera extrinsics
    T1 = Mat(vector<double>{-5.370000e-01, 4.822061e-03, -1.252488e-02},true);
    T2 = Mat(vector<double>{-4.731050e-01, 5.551470e-03, -5.250882e-03},true);
    assert(T1.rows == 3);
    
    // try to compute the focal length
    double X = 50, Y = 25, Z = 100;
    Mat p = P1 * Mat(Vec4d(X,Y,Z,1));
    double z = p.at<double>(2);
    double x = p.at<double>(0) / z;
    double y = p.at<double>(1) / z;
    fx = x*Z/X;
    fy = y*Z/Y;
    Mat Tdiff = T1-T2;
    Mat mT1_2 = (Tdiff).t()*(Tdiff);
    T1_2 = mT1_2.at<double>(0);
    cout << "focal lengths: " << fx << " " << fy << endl;
    
    // try to compute the field of view.
    Vec3d world_tl = unproject(Vec3d(0,0,fx));
    Vec3d world_br = unproject(Vec3d(fx,hRes,vRes));
    fovY = angle(Vec3d(world_tl[0],0,world_tl[2]),Vec3d(world_br[0],0,world_br[2]));
    fovX = angle(Vec3d(0,world_tl[1],world_tl[2]),Vec3d(0,world_br[1],world_br[2]));
    ostringstream oss;
    oss << filename << "world_tl = " << world_tl << "world_br = " << world_br << " fovx = " << fovX << "fovy = " << fovY << endl;
    log_once(oss.str());
    
    // we know z = d2z * 1/d
    // d2z = z*d
    Vec3d world_pos(1,1,5);
    Vec3d p1 = project(world_pos,1);
    Vec3d p2 = project(world_pos,2);
    d2z = world_pos[2]*(p2[0]-p1[0]);
    cout << "KITTI_Calibration: d2z = " << d2z << endl;
  }
  
  float KITTI_Calibration::disp2depth(float disp) const
  {
    if(!goodNumber(disp))
      return disp;
    
    return d2z/disp;
  }
  
  float KITTI_Calibration::depth2disp(float depth) const
  {
    if(!goodNumber(depth))
      return depth;
    
    // depth = 100*100*fx*T1_2/disp;
    // disp  = 100*100*fx*T1_2/depth
    return d2z/depth;
  }
  
  ///
  /// SECTION: Meatdata KITTI
  ///
  MetaDataKITTI::MetaDataKITTI(int id, bool training) : id(id), training(training)
  {
  }

  MetaDataKITTI::~MetaDataKITTI()
  {

  }
  
  map< string, AnnotationBoundingBox > MetaDataKITTI::get_positives()
  {
    lock_guard<mutex> l(exclusion);
    
    if(training && annotations.empty())
    {
      string annotation_filename = params::KITTI_dir() + "object/" + ((training)?"training":"testing") + 
	"/label_2/" + printfpp("%06d",id) + ".txt";
      annotations = KITTI_GT(annotation_filename);
    }
    return annotations;
  }
  
  shared_ptr< ImRGBZ > MetaDataKITTI::load_im_do() const
  {
    // Load the image pair
    //cout << "Loading KITTI Image Pair" << endl;
    string im_filename_1 = 
      params::KITTI_dir() + "object/" + ((training)?"training":"testing") + 
      "/image_2/" + printfpp("%06d",id) + ".png";
    string im_filename_2 = 
      params::KITTI_dir() + "object/" + ((training)?"training":"testing") + 
      "/image_3/" + printfpp("%06d",id) + ".png";      
    Mat rgb1 = imread(im_filename_1);
    Mat rgb2 = imread(im_filename_2);
    //image_safe("rgb1",rgb1,false);
    //image_safe("rgb2",rgb2,false);
      
    // Load the Velodyne data
    //cout << "Loading Velodyne LIDAR Data" << endl;
    string velodyne_filename = 
      params::KITTI_dir() + "object/" + ((training)?"training":"testing") + 
      "/velodyne/" + printfpp("%06d",id) + ".bin";     
    VelodyneData velodyne_cloud(velodyne_filename);
    //cout << "Loaded " << velodyne_cloud.getPoints().size() << " velodyne points" << endl;
    
    // Load the Calibration data
    //cout << "Loading Camera Calibration" << endl;
    string calib_filename = 
      params::KITTI_dir() + "object/" + ((training)?"training":"testing") + 
      "/calib/" + printfpp("%06d",id) + ".txt";      
    KITTI_Calibration calib(calib_filename);
      
    // compute the depth image estimate
    // my conclusion is that the stereo-LiDAR fusion process is unjustifiably slow
    Mat depth_lidar = formDepth_lidar(calib,velodyne_cloud,rgb1,rgb2,true);
    
    // construct the image
    CustomCamera kitti_cam(calib.fovX,calib.fovY,calib.hRes,calib.vRes);
    image.reset(new ImRGBZ(rgb1,depth_lidar,get_filename(),kitti_cam));    
    
    assert(image);
    return image;
  }
  
  shared_ptr< ImRGBZ > MetaDataKITTI::load_im()
  {
    unique_lock<mutex> l(exclusion);
    
    if(!image)
    {
      return load_im_do();
    }
    else
      return image;
  }

  std::shared_ptr< const ImRGBZ > MetaDataKITTI::load_im() const
  {
    unique_lock<mutex> l(exclusion);
    
    if(!image)
      return load_im_do();
    else
      return image;    
  }
  
  DetectionSet MetaDataKITTI::filter(DetectionSet src)
  {
    return src;
  }

  string MetaDataKITTI::get_filename() const
  {
    return printfpp("KITTI %d %d",(int)id,(int)training);
  }
  
  string MetaDataKITTI::get_pose_name()
  {
    return "NA";
  }
  
  bool MetaDataKITTI::leftP() const
  {
    return false;
  }
  
  bool MetaDataKITTI::use_negatives() const
  {
      return true;
  }

  bool MetaDataKITTI::use_positives() const
  {
      return true;
  }
  
  int MetaDataKITTI::getId() const
  {
    return id;
  }
}
