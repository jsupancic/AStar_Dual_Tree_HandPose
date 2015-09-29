/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "MetaData.hpp"

#ifndef DD_DETECTION
#define DD_DETECTION

#include <memory>
#include <opencv2/opencv.hpp>
#include "util_real.hpp"
#include "vec.hpp"
#include "FeatPyr.hpp"
#include <limits>
#include "ManifoldSelection.hpp"

namespace deformable_depth
{
  using std::shared_ptr;
  using namespace cv;
  
  struct Detection;
  typedef shared_ptr<Detection> DetPtr;
  typedef DetPtr DetectorResult;
  
  class IFeatPyr;
  class MetaData;

  struct EPM_Statistics
  {
  protected:
    map<string,long> counters;
    mutex monitor;
    string title;
    
  public:
    EPM_Statistics(string title = "");
    virtual ~EPM_Statistics();
    void count(string id, int increment = 1);
    void add(const EPM_Statistics&other);
    vector<string> toLines();
    void print();
    long getCounter(string name) const;
  };    
  typedef EPM_Statistics Statistics;
  
  struct Detection
  {
  protected:
    std::unique_ptr<map<string/*part name?*/,vector<Detection> > > parts;    
    std::unique_ptr<set<string> > m_part_names; 
    
  public:
    // data
#ifndef WIN32
    std::function<SparseVector ()> feature;
#endif
    // external info
    Rect_<double> BB; // the BB of the object in world coords
    RotatedRect rawBB;
    // the depth of the object in cm
    float depth;
    // the size of the object in the z axis.
    float z_size;
    float resp; // the detector responce (higher is better)
    Mat blob; // floating point?
    string pose;
    string src_filename;
    int lr_flips; // defaults to zero
    // number between 0 and 1 representing the %
    // of the object which is occluded
    float occlusion; 
    // number between 0 and 1 representing percent which is 
    // "real", that is, neither background nor occlusion
    float real;
    // what scale factor was the detection at?
    float scale_factor;
    // the exmplar used for this configuration.
    shared_ptr<MetaData> exemplar;
    // why/how was this rejected?
    shared_ptr<EPM_Statistics> pw_debug_stats;
    // associated latent space representation
    vector<double> latent_space_position;
    // is the detection supressed?
    bool supressed;
    // why skipped? 
    int jump_code;
    // was an in-plane rotation applied?
    float in_plane_rotation;
    // links to parent windows in a resolution pyramid.
    vector<DetectorResult> pyramid_parent_windows;
    // how many levels of down-pyramiding was required to get to this window?
    int down_pyramid_count;
    
    /// methods
    Detection() : 
      resp(-inf), depth(qnan), lr_flips(0), 
      occlusion(0), real(1), scale_factor(1), supressed(false),z_size(qnan), in_plane_rotation(qnan),down_pyramid_count(0), jump_code(0) {};
    Detection(const Detection& other);
      
	string toString();
    bool operator<(const Detection&other) const;
    Detection& operator= (const Detection&copyFrom);
    // return true if this is a black example
    // against the given set of positives and assuming the 
    // root has the given type
    bool is_black_against(
      map<string,AnnotationBoundingBox>&positives,double ol_thresh = .5) const;
    // return the keypoints for this detection.
    vector<Point2d> keypoints(bool toplevel = true) const;
    // return a list of all parts
    set<string> parts_flat() const;
    double getDepth(const ImRGBZ&im) const;
    bool is_occluded() const;
    string print_resp_computation() const;
    void applyAffineTransform(Mat&affine);
    void tighten_bb();
    
    const set<string>&part_names() const;
    void emplace_part(string part_name,const Detection&part_detection,bool keep_old = false);
    void set_parts(map<string,AnnotationBoundingBox > parts);
    const Detection& getPartCloseset(string part_name, Rect_<double> bb) const;
    const Detection& getPart(string part_name) const;      
    Detection& getPartCloseset(string part_name, Rect_<double> bb);
    Detection& getPart(string part_name);    
    
    friend void write(cv::FileStorage&, std::string, const deformable_depth::Detection&);
    friend void read(const cv::FileNode&, deformable_depth::Detection&, deformable_depth::Detection);
  };  
  void write(cv::FileStorage&, std::string, const deformable_depth::Detection&);
  void read(const cv::FileNode&, deformable_depth::Detection&, deformable_depth::Detection);

  class DetectionSet : public vector<DetectorResult>
  {
  public:
    DetectionSet() = default;
    DetectionSet(const vector<DetectorResult>&cpy);
    // used to debug part based models
    map<string/*part_name*/,DetectionSet > part_candidates;
  };    
  
  struct DetectionFilter
  {
    // only return detections above thresh
    float thresh;
    // return at most nmax detections
    int nmax;
    // HINT: The ground truth has pose label $pose_hint
    // 		this is to be used for diagnostics
    string pose_hint;
    // HINT: for debugging
    map<string,AnnotationBoundingBox> gt_bbs_hint;
    // option
    bool supress_feature; // default = false
    bool sort; // default = true
    bool verbose_log; // default = false
    // require the responce to be an unscaled (raw) dot product.
    bool require_dot_resp; // default = false
    bool allow_occlusion; // default = true
    bool testing_mode; // default = false
    bool is_root_template; // default = true
    bool use_skin_filter;
    bool slide_window_only; // default = false ; does not do any detection or feature extraction
    ManifoldFn manifoldFn;
    
    DetectionFilter(float thresh = inf, int nmax = std::numeric_limits<int>::max(), 
		    string pose_hint = "");
    void apply(DetectionSet&detections);
    shared_ptr<IFeatPyr> feat_pyr;
    std::weak_ptr<MetaData> cheat_code;
  };      
  
  DetectionSet sort(DetectionSet src);
  DetectionSet nms_w_list(DetectionSet src, double overlap);
  DetectionSet nms_apx_w_array(const DetectionSet&src, double nmax, Size imSize);
  void translate(DetectionSet&detections,Vec2d offset);
  DetectorResult nearest_neighbour(const DetectionSet&dets,Rect_<double> bb);
  DetectionSet removeTruncations(const ImRGBZ&im,DetectionSet src);
  DetectionSet fromMetadata(MetaData&datum);
}

#endif
