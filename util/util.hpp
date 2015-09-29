/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_UTIL
#define DD_UTIL

#include <string>

namespace cv
{
  class FileStorage;
  class String;
  class FileNode;
  
  template<typename T>
  void write(cv::FileStorage&fs, cv::String&cv_str, T&v);

  template<typename T>
  void write(cv::FileStorage&fs, cv::String&cv_str, const T&v);

  static void read(const cv::FileNode&, std::string&, std::string);
}

#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include <iomanip>

#define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))

namespace cv
{
  template<typename T>
  void write(cv::FileStorage&fs, cv::String&cv_str, T&v)
  {
    std::string std_str = cv_str.operator std::string();
    write(fs,std_str,v);
  }
  
  template<typename T>
  void write(cv::FileStorage&fs, cv::String&cv_str, const T&v)
  {
    std::string std_str = cv_str.operator std::string();
    write(fs,std_str,v);
  }

  void read(const cv::FileNode&node, std::string&s, std::string)
  {
    cv::String cvStr;
    read(node,cvStr,cv::String());
    s = cvStr;
  }
}

#include "util_real.hpp"
#include "util_depth.hpp"
#include "util_file.hpp"
#include "util_mat.hpp"
#include "util_rect.hpp"
#include "util_graph.hpp"
#include "util_vis.hpp"
#include "DepthFeatures.hpp"
#include "Log.hpp"

#ifndef WIN32
#include "hashMat.hpp"
#endif

namespace deformable_depth
{
  using cv::Mat;
  using cv::Mat_;
  using cv::Rect;
  using cv::Point2i;
  using cv::Point;
  
  using std::vector;
  using std::numeric_limits;
  using std::string;
  
  template<typename T>
  void require_equal(T&v1,T&v2)
  {
    if(v1 != v2)
    {
      cout << v1 << " != " << v2 << endl;
      log_file << v1 << " != " << v2 << endl;
      assert(false);
    }
  } 
    
  template<typename T>
  void require_equal(T v1,T v2)
  {
    if(v1 != v2)
    {
      cout << v1 << " != " << v2 << endl;
      log_file << v1 << " != " << v2 << endl;
      assert(false);
    }
  }  
  
  template<typename T>
  void require_not_equal(const T&v1,const T&v2)
  {
    if(v1 == v2)
    {
      cout << v1 << " == " << v2 << endl;
      log_file << v1 << " == " << v2 << endl;
      assert(false);
    }
  }     
  
  template<typename T>
  void require_gt(const T&v1, const T&v2) // >
  {
    if(v1 <= v2)
    {
      cout << v1 << " <= " << v2 << endl;
      log_file << v1 << " <= " << v2 << endl;      
      assert(false);
    }
  }
  
  template<typename T>
  void require_in_range(const T l, const T v, const T u)
  {
    if(v < l || u < v)
    {
      cout << v << " not in [" << l << ", " << u << "]" << endl;
      assert(false);
    }
  }

  template<typename T>
  void require_true(bool is_true, const T v, const T u)
  {
    if(not is_true)
    {
      cout << "v = " << v << endl;
      cout << "u = " << u << endl;
      assert(false);
    }
  }

  void require(bool must_be_true,string error);
  
  void map_blockToCell4(int blockX, int blockY, int cellN, int&cell_cell_X, int&cell_cell_Y);
  Mat autoCanny(Mat image);
  string printfpp(const char*format,...);
  string current_time_string();
  void pause();
  string varName(string str);

  // implements a typesafe printf statement  
  // base cases
  string safe_printf(const string&format);
  string safe_printf(const char*);
  // recursive case
  template<typename FT, typename... RT>
  string safe_printf(const string&format,FT first, RT... rest)
  {
    ostringstream oss;
    
    size_t pos = format.find("%");
    if(pos == string::npos)
    {
      // no positional specifier found,
      // just append...
      oss << format << " " << first << " " << safe_printf("",rest...);
    }
    else
    {
      // split the string and insert
      string before(format.begin(),format.begin()+pos);
      string after(format.begin()+pos+1,format.end());
      oss << before << first << safe_printf(after,rest...);
    }

    return oss.str();
  }
  
  enum Direction
  {
    LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3, ALL = 4,
    left = LEFT, right = RIGHT, up = UP, down = DOWN
  };  
  vector<Direction> card_dirs();
  Direction opp_dir(Direction dir);
  Vec2i dir2vec(deformable_depth::Direction dir);
  
#ifdef DD_CXX11
  Rect bbWhere(const Mat&im,std::function<bool(Mat&im,int y, int x)> predicate);
#endif
      
  bool merge(const Mat rndRGB, const Mat rndZ, 
	     const Mat capRGB, const Mat capZ, 
	     Mat&outRGB, Mat&outZ,
	     float&rndOcc, float&capOcc
	    );
    
  template<typename D>
  string toString(const D&output)
  {
    ostringstream oss;
    oss << output;
    return oss.str();
  }
  template<typename T>
  string to_string(T value,int leading_zeros)
  {
    ostringstream oss;
    oss << setfill('0') << setw(leading_zeros) << value;
    return oss.str();
  }
    
  // generalized distance transform
  // LC : Local Cost
  // Z  : Depth
  void GDT(Mat_<float> LC, Mat_<float> Z,Mat_<float>&newLC, 
	   float w1 = 0, float w2 = 1);
  
  // see also: 
  //     vector<T> random_sample_w_replacement(const vector<T>&domain,size_t n, int seed = -1);
  template<typename T>
  vector<T> pseudorandom_shuffle(const vector<T>&src)
  {
    vector<T> result(src.begin(),src.end());
    std::mt19937 sample_seq;
    sample_seq.seed(19860);
    std::shuffle(result.begin(),result.end(),sample_seq);
    return result;
  }
  
  bool prompt_yes_no(string message);
  // returns colors form a determinisitic but
  // psudo-random sequence.
  Vec3b getColor(unsigned int index);
  Scalar getColorScalar(unsigned int index);
  void randomize_seed();
  int thread_rand();
  
  void dump_heap_profile();
  void do_command(const char*format,...);
  void breakpoint();

  template<typename T>
  Point_<T> take2(const Point3_<T>&pt)
  {
    return Point_<T>(pt.x,pt.y);
  }

  template <typename T> 
  T rol(T val) 
  {
    return (val << 1) | (val >> (sizeof(T)*CHAR_BIT-1));
  }

  class Timer
  {
  protected:
    std::chrono::time_point<std::chrono::system_clock> start;

  public:
    Timer();
    void tic();
    long toc();
  };

  int longestCommonSubstring(const string&str1, const string&str2);
  string alpha_substring(const string&input);

  template<typename... AT>
  Mat image_format(const string&format,AT... args)
  {
    return image_text(safe_printf(format,args...));
  }

  float im_mean(const Mat&m);
}

#endif

