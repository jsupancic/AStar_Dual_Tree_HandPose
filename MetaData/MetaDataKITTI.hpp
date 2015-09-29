/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#ifndef DD_KITTI
#define DD_KITTI

#include "MetaData.hpp"
#include "Velodyne.hpp"

namespace deformable_depth
{
  class KITTI_Calibration
  {
  public:
    KITTI_Calibration(string filename);
    Mat P0, P1, P2, P3, R0_Rect, Tr_velo_to_cam, Tr_imu_to_velo;
    // camera parameters
    Mat T1,T2; // camera translation matrices
    double T1_2; // distance between cameras
    double fx, fy; // focal length
    double fovX, fovY;
    // derived parameters.
    static constexpr int hRes = 1224;
    static constexpr int vRes = 370;
    // constant factor for disp2depth/depth2disp
    double d2z;
     
    Vec3d project(Vec3d,int cam = 2) const;
    Vec3d unproject(Vec3d, int cam = 2) const;
    float disp2depth(float disp ) const;
    float depth2disp(float depth) const;
    
  protected:
    Mat PMat(int cam) const;
  };
  
  class MetaDataKITTI : public MetaDataNoKeypoints
  {
  public:
    // core functions
    MetaDataKITTI(int id, bool training = true);
    virtual ~MetaDataKITTI();

    // abstract methods
    virtual map<string,AnnotationBoundingBox > get_positives() ;
    virtual std::shared_ptr<ImRGBZ> load_im();    
    virtual std::shared_ptr<const ImRGBZ> load_im() const;
    virtual string get_pose_name();
    virtual string get_filename() const ;
    virtual bool leftP() const ;
    virtual DetectionSet filter(DetectionSet src);   
    virtual bool use_negatives() const;
    virtual bool use_positives() const;
    
    int getId() const;
  
  protected:
    int id;
    bool training;
    mutable mutex exclusion;
    mutable shared_ptr<ImRGBZ> image;
    map< string, AnnotationBoundingBox > annotations;
    
    virtual std::shared_ptr<ImRGBZ> load_im_do() const;  
  };
  
  vector<shared_ptr<MetaData> > KITTI_default_data();
  vector<shared_ptr<MetaData> > KITTI_default_train_data();
  vector<shared_ptr<MetaData> > KITTI_default_test_data();
  vector<shared_ptr<MetaData> > KITTI_validation_data();
  Mat formDepth_lidar(
    KITTI_Calibration&calib,VelodyneData&point_cloud,Mat&RGB1,Mat&RGB2,bool fillHoles = false);
  
  map<string,AnnotationBoundingBox> KITTI_GT(string filename);
  void KITTI_Demo();
}

#endif
