/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_METADATA
#define DD_METADATA

#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include "util_mat.hpp"
#include "util_file.hpp"
#include <functional>
#include "ThreadCompat.hpp"

namespace deformable_depth
{
  // virtualify Rect..
  class VirtualRect : public Rect_<double>
  {
  public:
    VirtualRect(){};
    VirtualRect(Rect_<double> bb) : Rect_<double>(bb) {};
  private:
    virtual void f() {};
  };

  // class which represents and annotated bounding box
  class AnnotationBoundingBox : public VirtualRect
  {
  public:
    AnnotationBoundingBox(Rect_<double> bb, float visible);
    AnnotationBoundingBox();
    AnnotationBoundingBox& write(const VirtualRect&update_bb);
    //AnnotationBoundingBox& operator=(const VirtualRect&update_bb);
    
    double confidence;
    double depth;
    float visible;
    Vec3d up, normal;
  };  
  void write(cv::FileStorage&, std::string, const AnnotationBoundingBox&);
  void read(cv::FileNode, AnnotationBoundingBox&, 
	    AnnotationBoundingBox);
  std::ostream& operator<< (std::ostream &out, AnnotationBoundingBox &bb);
struct PerformanceRecord;
}

#include "Detection.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  template<typename T>
  struct Maybe
  {
    T value;
    bool initialized;
    Maybe() : initialized(false) {};
  };
    
  ///
  /// Class which represents an image and assocaited annotation information.
  ///
  struct BaselineDetection;  
  class MetaData
  {
  public:
    MetaData();
    // core functions
    virtual ~MetaData();

    // abstract methods
    virtual map<string,AnnotationBoundingBox > get_positives() = 0;
    virtual std::shared_ptr<ImRGBZ> load_im() = 0;    
    virtual std::shared_ptr<const ImRGBZ> load_im() const = 0;
    virtual string get_pose_name() = 0;
    virtual string get_filename() const = 0;
    virtual bool leftP() const = 0;
    virtual DetectionSet filter(DetectionSet src) = 0;
    // keypoint functions
    virtual pair<Point3d,bool> keypoint(string name) = 0;
    virtual int keypoint() = 0;
    virtual bool hasKeypoint(string name) = 0;
    virtual vector<string> keypoint_names() = 0;    
    
    // concreate methods    
    virtual vector<shared_ptr<BaselineDetection> > ground_truths() const;    
    virtual map<string,MetaData* > get_subdata() const; 
    const map<string,AnnotationBoundingBox>& get_positives_c();
    virtual Mat load_raw_RGB();
    virtual Mat load_raw_depth();
    virtual bool use_negatives() const;
    virtual bool use_positives() const;
    virtual Mat getSemanticSegmentation() const;
    virtual void drawSkeletons(Mat&target,Rect boundings) const;
    // export the annotations in a more friendly format for other people.
    virtual void export_keypoints(string filename);
    virtual void export_annotations(string filename);
    virtual void export_annotations();
    virtual void export_one_line_per_hand(string filename);
    bool operator< (const MetaData& other) const;
    const map<string,AnnotationBoundingBox >& default_training_positives();
    void putAnnotation(string key, string value);
    bool hasAnnotation(string key);
    string getAnnotation(string key);
    virtual bool loaded() const {return true;};
    string leveldbKeyRoot();

  protected:
    // real (non-abstract) utility functions    
    Rect bbForKeypoints(Point p1, Point p2, const ImRGBZ&im,double side_len = 5);
    map< string, AnnotationBoundingBox > get_essential_positives(Rect_<double> handBB);
    mutable map<string,AnnotationBoundingBox> training_positives_cache;
    mutable map<string,AnnotationBoundingBox> all_positives_cache;
    mutable bool training_positives_cache_ready, all_positives_cache_ready;
    typedef recursive_mutex ExclusionMutexType;
    mutable ExclusionMutexType exclusion;
  };
  void write(cv::FileStorage&, std::string, const std::shared_ptr<deformable_depth::MetaData>&);
  void read(cv::FileNode, std::shared_ptr<deformable_depth::MetaData>&, 
	    std::shared_ptr<deformable_depth::MetaData>);
  set<string> essential_hand_positives();
  pair<string,string> joints4bone(const string&bone_name);
  
  class MetaDataNoKeypoints : public MetaData
  {
  public:
    virtual pair<Point3d,bool> keypoint(string name);
    virtual int keypoint();
    virtual bool hasKeypoint(string name);
    virtual vector<string> keypoint_names();    
  };
  
  class MetaData_YML_Backed;
  
  class MetaData_Editable : public MetaData
  {
  public:
    virtual void setPose_name(string newName) = 0;
    virtual void change_filename(string filename) = 0;
    virtual string naked_pose() const = 0;
  };  
  
  class MetaData_LR_Flip : public MetaData
  {
  public:
    MetaData_LR_Flip(shared_ptr<MetaData> backend);
    virtual ~MetaData_LR_Flip();
    virtual map<string,AnnotationBoundingBox> get_positives();
    virtual std::shared_ptr<ImRGBZ> load_im();  
    virtual std::shared_ptr<const ImRGBZ> load_im() const;
    virtual string get_pose_name();
    virtual string get_pose_name_naked();
    virtual string get_filename() const;
    // keypoint functions
    virtual pair<Point3d,bool> keypoint(string name);
    virtual int keypoint();
    virtual bool hasKeypoint(string name);
    virtual vector<string> keypoint_names();    
    virtual DetectionSet filter(DetectionSet src);
    virtual bool leftP() const;
    
    virtual bool use_negatives() const;
    virtual bool use_positives() const;
  protected:
    void build_affine_transform();
    
    void ensure_im_cached() const;
    mutable mutex monitor;
    mutable shared_ptr<ImRGBZ> im_cache;
    int width, height;
    Mat affine;
    shared_ptr<MetaData> backend;
  };
      
  vector<string> filenames(vector<shared_ptr<MetaData> >&metadatas);
  void export_depth_exr();
  void do_imageeq();
 
  shared_ptr< MetaData > metadata_build(string filename, 
					bool read_only = false, bool only_valid = true);  
  // build training set, insert Left-Right flips.
  vector<shared_ptr< MetaData > > metadata_build_training_set(
    vector< string > train_files,
    bool read_only = false,
    function<bool(MetaData&)> filter = [](MetaData&){return true;});
  vector<shared_ptr<MetaData>> metadata_build_all(
    string dir, bool only_valid = true,bool read_only = true);
  vector<shared_ptr<MetaData>> metadata_build_all(
    vector<string> dirs,bool read_only = true, bool only_valid = true);
  void metadata_insert_lr_flips(vector<shared_ptr<MetaData> >&examples);
  
  // filter functions
  vector<shared_ptr<MetaData> > filterLeftOnly(vector<shared_ptr<MetaData> >&training_set);
  vector<shared_ptr<MetaData> > filterRightOnly(vector<shared_ptr<MetaData> >&training_set);
  vector<shared_ptr<MetaData> > filter_for_pose(
    const vector<shared_ptr<MetaData> >&metadata,
    const string& pose);
  shared_ptr< MetaData > metadata_build_with_cache(string filename, bool read_only);
  void split_pos_neg(
    const vector<shared_ptr<MetaData> >&training_set,
    vector<shared_ptr<MetaData> > & positive_set,
    vector<shared_ptr<MetaData> > & negtive_set);
  struct Orientation
  {
    double phi, theta, psi;
  };
}

#include "YML_Data.hpp"

#endif
