/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_PARAMS
#define DD_PARAMS

#include <functional>
#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <set>
#include <sstream>

namespace deformable_depth
{
  using cv::Size;
  using std::string;
  using std::map;
  using std::set;
  
  // parses options of the form key=value
  // from the command line and makes them available
  class Params 
  {
  public:
    void parse(int argc, char**argv);
    bool has_key(const string key) const;
    std::string get_value(const string key, string def = "") const;
    std::string require(const string key) const;
    void log_params();
    map<string,string> matching_keys(string RE);
    set<string> matching_values(string RE);
    
    bool option_is_set(string name) const;
  protected:
    void parse_token(string token);
    
    map<string,string> m_params;
    std::set<string> included_files;
  };
  
  class AnnotationBoundingBox;
  typedef std::function<map<string,AnnotationBoundingBox>
	(const map<string,AnnotationBoundingBox>&all_pos)> TrainingBBSelectFn;
	
  extern Params g_params;
  class Model_Builder;
  
  namespace params
  {
    TrainingBBSelectFn makeSelectFn(string regex);
    TrainingBBSelectFn defaultSelectFn();
    string KITTI_dir();
    string out_dir();
    string cache_dir();
    int cpu_count();
    double finger_correct_distance_threshold();
    int video_annotation_stride();
    string synthetic_directory();
    bool use_skin_filter();
    string target_category();
    double pyramid_sharpness();

#ifdef DD_CXX11
    // for skin detection histograms.
    extern int channels[];
    extern float colRange[];
    const extern float*ranges[];
    const float DEPTH_EDGE_THRESH = .2; // cm?
    constexpr int histSz_1 = 16;
    constexpr int histSz[] = {histSz_1,histSz_1,histSz_1};
    constexpr float DEPTHS_SIMILAR_THRESAH = 10;
    
    // training parameters
    constexpr double HARD_NEG_PASSES = 5;
#endif
    // operating resolutions
    const float vRes = 480, hRes = 640;
    const float depth_vRes = 240, depth_hRes = 320;
    
    // physical properties of the ASUS camera (far range)
    //const float vFov = 45, hFov = 58;
    // physical properties for the Intel camera (near range)
    const float V_Z_FOV = 58, H_Z_FOV = 74;
    const float V_RGB_FOV = 54, H_RGB_FOV = 65;
    const float vFov = V_Z_FOV, hFov = H_Z_FOV;
    
    // mathematical constants
    const float PI = 3.14159265;

    // 
    const float MIN_Z_STEP = 2; // in cm
    
#ifdef DD_CXX11
    // detector paramters
    constexpr float world_area_variance = 2; // 1.5 is the "best"
    constexpr int 
	RESP_ORTHO_X_RES = 160, 
	RESP_ORTHO_Y_RES = 120,
	RESP_ORTHO_Z_RES = 10;
#endif

    // HOSR data
    const Size TSize(20,20);
    
    // low conf value
    const double depth_low_conf = 32002/125;
    
    // operating range
#ifndef DD_CXX11
	#define constexpr const
#endif

    // for Intel's PXC/Creative Gesture Camera
    float MAX_X(); // X
    float MIN_X();
    float MAX_Y(); // Y     
    float MIN_Y();
    float MAX_Z(); // Z
    float MIN_Z();
    
    float min_image_area();
    int max_examples();

    long long constexpr GB4 = 4294967296;
    long long constexpr BIG_MEM = GB4;
    
    // Number of Orientation bins
    const int ORI_BINS = 18;

    // unit vectors used to compute gradient orientation
    constexpr static float uu[ORI_BINS] = 
    {
      1.0000,
      0.9397,
      0.7660,
      0.5000,
      0.1736,
      -0.1736,
      -0.5000,
      -0.7660,
      -0.9397,
      -1.0000,
      -0.9397,
      -0.7660,
      -0.5000,
      -0.1736,
      0.1736,
      0.5000,
      0.7660,
      0.9397
    };

    constexpr static float vv[ORI_BINS] = 
    {
      0,
      0.3420,
      0.6428,
      0.8660,
      0.9848,
      0.9848,
      0.8660,
      0.6428,
      0.3420,
      0,
      -0.3420,
      -0.6428,
      -0.8660,
      -0.9848,
      -0.9848,
      -0.8660,
      -0.6428,
      -0.3420
    };
    
    // derivative filters
    extern float dervFilter[];
    extern cv::Mat dxFilter;
    extern cv::Mat dyFilter;
    extern cv::Mat dxFilter01;
    extern cv::Mat dyFilter01;

    // return configured C value for SVMs
    double C();

    // get the configured object depth
    float obj_depth();

    // cv::INTER_AREA; cv::INTER_NEAREST;
    constexpr static int DEPTH_INTER_STRATEGY = cv::INTER_AREA;

    //
    const deformable_depth::Model_Builder& model_builder();
  }

  template<typename D>
  D fromString(const string input)
  {
    std::istringstream iss(input);
    D d;
    iss >> d;
    return d;
  }

  enum Arguments
  {
    SYNTH_CLUSTERS,
    SYNTH_EX_PER_CLUSTER,
    SYNTH_PERTURB_R_MIN,
    SYNTH_PERTURB_R_MAX
  };

  template<typename T,Arguments arg>
  T get_param(const string&name)
  {
    static T value;
    static bool value_set = false;
    if(not value_set)
    {
      value = fromString<T>(g_params.require(name));
      value_set = true;
    }

    return value;
  }
}

#endif
