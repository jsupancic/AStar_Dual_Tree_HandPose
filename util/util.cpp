/**
 * Copyright 2012: James Steven Supancic III
 **/

#include <iostream>
#include "util.hpp"
#include <opencv2/opencv.hpp>
#include <queue>
#include "params.hpp"
#include "vec.hpp"
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <stdarg.h>
#include "ThreadCompat.hpp"
#include "Semaphore.hpp"
#include "Log.hpp"
#ifdef DD_CXX11
#include <google/heap-profiler.h>
#endif

namespace deformable_depth
{
  using std::cout;
  using std::endl;
  using cv::Scalar;
  using std::numeric_limits;
  
  using namespace cv;
  using namespace std;
  using namespace boost::filesystem;
  
  void dump_heap_profile()
  {
#ifdef DD_CXX11
    if(g_params.has_key("HEAP_PROF"))
    {
      static mutex m; lock_guard<mutex> l(m);
      HeapProfilerDump("dump");
    }
#else
	  assert(false);
#endif
  }
  
  Mat autoCanny(Mat image)
  {
    if(image.type() == DataType<float>::type)
    {
      image = imagesc("",image,false,false);
      cv::cvtColor(image,image,CV_RGB2GRAY);
      image.convertTo(image,CV_8UC1);
    }
    else
      image.convertTo(image,CV_8UC1);
    
    // for depth
    // 1 is to low
    Mat out;
    Scalar mean = cv::mean(image);
    Scalar low = .66*mean;
    Scalar high = 1.33*mean;
    
    //double hi = high[0], lo = low[0];
    double lo = 10, hi = 10;
    cout << "autocanny: low = " << lo << " high = " << hi << endl;
    //cout << image;
    Canny(image, out, lo, hi);
    return out;
  }
  
  void map_blockToCell4(int blockX, int blockY, int cellN, int& cell_cell_X, int& cell_cell_Y)
  {
    cell_cell_X = blockX;
    cell_cell_Y = blockY;
    // 0: x + 0; y + 0
    // 1: x + 0; y + 1
    if(cellN == 1)
      cell_cell_Y++;
    // 2: x + 1; y + 0
    else if(cellN == 2)
      cell_cell_X++;
    // 3: x + 1; y + 1
    else if(cellN == 3)
    {
      cell_cell_X++; cell_cell_Y++;
    }
  }
                  
  cv::Rect bbWhere(const Mat&im,std::function<bool(cv::Mat&im,int y, int x)> predicate)
  {
    int x1 = im.cols-1;
    int y1 = im.rows-1;
    int x2 = 0;
    int y2 = 0;
    
    for(int yIter = 0; yIter < im.rows; yIter++)
      for(int xIter = 0; xIter < im.cols; xIter++)
	if(predicate((Mat&)im,yIter,xIter))
	{
	  x1 = std::min(x1,xIter);
	  y1 = std::min(y1,yIter);
	  x2 = std::max(x2,xIter);
	  y2 = std::max(y2,yIter);
	}
    return cv::Rect(cv::Point2i(x1,y1),cv::Point2i(x2,y2));
  }
  
  bool merge(const Mat rndRGB, const Mat rndZ, 
	     const Mat capRGB, const Mat capZ, 
	     cv::Mat&outRGB, cv::Mat&outZ,
	     float&rndOcc, float&capOcc)
  {
    outRGB = rndRGB.clone();
    outZ = rndZ.clone();
    rndOcc = 0;
    capOcc = 0;
    bool noOcclusion = true;
    
    assert(!rndZ.empty());
    assert(!capZ.empty());
    
    for(int yIter = 0; yIter < outRGB.rows; yIter++)
      for(int xIter = 0; xIter < outRGB.cols; xIter++)
	if(rndZ.at<float>(yIter,xIter) < capZ.at<float>(yIter,xIter))
	{
	  // use rnd
	  rndOcc++;
	  outZ.at<float>(yIter,xIter) = rndZ.at<float>(yIter,xIter);
	  outRGB.at<cv::Vec3b>(yIter,xIter) = rndRGB.at<cv::Vec3b>(yIter,xIter);
	}
	else
	{
	  // use cap
	  Vec3b rColor = rndRGB.at<Vec3b>(yIter,xIter);
	  if(rColor != Vec3b(0,0,0))
	    noOcclusion = false;
	    
	  capOcc++;
	  outZ.at<float>(yIter,xIter) = capZ.at<float>(yIter,xIter);
	  outRGB.at<cv::Vec3b>(yIter,xIter) = capRGB.at<cv::Vec3b>(yIter,xIter);
	}
	
    return noOcclusion;
  }
   
#ifndef WIN32
  void GDT(
    Mat_<float> LC, Mat_<float> Z, Mat_<float>&newLC,
    float w1, float w2
  )
  {
    float p = tan(params::hFov*params::PI/180);
    float q = tan(params::vFov*params::PI/180);
    auto dist = [p,q](float x1, float x2, float x3, float y1, float y2, float y3)
    {
      float z1 = x3;
      float z2 = y3;
      // x and y coords need to be untransformed from proj to world coordinates.
      // essentially, scale, [0,640] into [-P/2,P/2]
      x1 /= params::hRes;
      y1 /= params::vRes;
      x2 /= params::hRes;
      y2 /= params::vRes;
      x1 *= p*z1 - p*z1/2;
      y1 *= q*z1 - q*z1/2;
      x2 *= p*z2 - p*z2/2;
      y2 *= q*z2 - q*z2/2;
      
      // 
      float d1 = x1 - y1;
      float d2 = x2 - y2;
      float d3 = x3 - y3;
      return std::sqrt(d1*d1 + d2*d2 + d3*d3);
    };
    
    int MAX_OFFSET = 50;
    
    printf("util::GDT Begin\n");
    newLC = LC.clone();
    
    // maybe add openMP here?
    //  then each (yIter1, xIter1) is in its own thread.
    active_worker_thread.V();
    #pragma omp parallel for
    for(int yIter1 = 0; yIter1 < Z.rows; yIter1++)
    {
      active_worker_thread.P();
      printf("Computing DT for Row = %d\n",yIter1);
      for(int xIter1 = 0; xIter1 < Z.cols; xIter1++)
	// the inner loop...
	// but (yIter2, xIter2) should be in the same thraed.
	for(int yIter2 = std::max<int>(0,yIter1 - MAX_OFFSET); 
	    yIter2 < std::min<int>(yIter1 + MAX_OFFSET,Z.rows); yIter2++)
	  for(int xIter2 = std::max<int>(0,xIter1 - MAX_OFFSET); 
	      xIter2 < std::min<int>(xIter1 + MAX_OFFSET,Z.cols); xIter2++)
	  {
	    float Z2 = Z.at<float>(yIter2,xIter2);
	    float Z1 = Z.at<float>(yIter1,xIter1);
	    
	    float d = dist(xIter2,yIter2,Z2,xIter1,yIter1,Z1);
	    
	    float alt_cost = 
	      LC.at<float>(yIter2,xIter2) + w1*d + w2*d*d;
	    newLC.at<float>(yIter1,xIter1) = 
	      std::min<float>(newLC.at<float>(yIter1,xIter1),alt_cost);
	  }
      active_worker_thread.V();
    }
    active_worker_thread.P();
	
    printf("util::GDT End\n");
  }
#endif
  
  vector<Direction> card_dirs()
  {
    vector<Direction> dirs;
    dirs.push_back(LEFT);
    dirs.push_back(RIGHT);
    dirs.push_back(UP);
    dirs.push_back(DOWN);
    return dirs;
  }  
  
  Direction opp_dir(Direction dir)
  {
    if(dir == LEFT)
      return RIGHT;
    if(dir == RIGHT)
      return LEFT;
    if(dir == UP)
      return DOWN;
    if(dir == DOWN)
      return UP;
    assert(false);
	return RIGHT;
  }
  
  Vec2i dir2vec(Direction dir)
  {
    if(dir == Direction::LEFT)
      return Vec2i(0,-1);
    else if(dir == Direction::RIGHT)
      return Vec2i(0,+1);
    else if(dir == Direction::UP)
      return Vec2i(-1,0);
    else if(dir == Direction::DOWN)
      return Vec2i(+1,0);
    else
      throw std::exception();
  }  
  
  string printfpp(const char* format, ... )
  {
    int size = 128;
    char*char_store = (char*)malloc(size);
    while(true)
    {
      // try to print
      va_list ap;
      va_start(ap,format);
      int n = vsnprintf(char_store,size,format,ap);
      va_end(ap);
      
      // if it worked, return
      if(n > -1 && n < size)
      {
	string result(char_store);
	free(char_store);
	return result;
      }
      
      // otherwise, try again with more space
      if(n > -1)
	size = n+1;
      else
	size *= 2;
      // realloc
      char_store = (char*)realloc(char_store,size);
    }
  }
  
  void do_command(const char* format, ...)
  {
    va_list ap;
    va_start(ap,format);
    string command = printfpp(format,ap);
    va_end(ap);
    
    cout << "do_command: " << command << endl;
    assert(system(command.c_str()) == 0);
  }
  
  string current_time_string()
  {
    time_t cur_time;
    time(&cur_time);
    
    tm* time_info = localtime(&cur_time);
    
    char buffer[180];
    strftime(buffer,180,"%Y-%m-%d\t%H:%M:%S",time_info);
    
    return printfpp("%s",buffer);
  }
  
  bool prompt_yes_no(string message)
  {
    cout << message;
    char input = ' ';
    while(true)
    {
      cin >> input;
      if(input == 'y')
	return true;
      if(input == 'n')
	return false;
      assert(!cin.fail());
    }
    cout << endl;
  }
   
#ifndef WIN32
  Vec3b getColor(unsigned int index)
  {
    std::mt19937 sample_seq;
    sample_seq.seed(19860);
    uniform_int_distribution<int> col_dist(0,255);
    int R,G,B;
    for(int iter = 0; iter < index + 1; iter++)
    {
      R = col_dist(sample_seq);
      G = col_dist(sample_seq);
      B = col_dist(sample_seq);
    }
    return Vec3b(R,G,B);
  }
  
  Scalar getColorScalar(unsigned int index)
  {
    Vec3b color = getColor(index);
    return Scalar(color[0], color[1], color[2]);
  }
#endif
  
  void pause()
  {
    cout << "Press ENTER to continue" << endl;
    cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
  }
  
  // http://www.concentric.net/~Ttwang/tech/inthash.htm
  static unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
  {
      a=a-b;  a=a-c;  a=a^(c >> 13);
      b=b-c;  b=b-a;  b=b^(a << 8);
      c=c-a;  c=c-b;  c=c^(b >> 13);
      a=a-b;  a=a-c;  a=a^(c >> 12);
      b=b-c;  b=b-a;  b=b^(a << 16);
      c=c-a;  c=c-b;  c=c^(b >> 5);
      a=a-b;  a=a-c;  a=a^(c >> 3);
      b=b-c;  b=b-a;  b=b^(a << 10);
      c=c-a;  c=c-b;  c=c^(b >> 15);
      return c;
  }
  
#ifndef WIN32
  void randomize_seed()
  {
    unsigned long seed = mix(clock(), time(NULL), getpid());
    srand(seed);
  }
#endif

  string varName(string str)
  {
    boost::regex bad_re("[-\\s/]+");
    return boost::regex_replace(str,bad_re,"");
  }
 
  void breakpoint()
  {
    asm volatile ("int3;");
  }

  int thread_rand()
  {
    static thread_local std::mt19937 sample_seq(19860);
    std::uniform_int_distribution<> dist(0,std::numeric_limits<int>::max());
    return dist(sample_seq);
  }

  string safe_printf(const string&format)
  {
    return format;
  }

  string safe_printf(const char* v)
  {
    return v;
  }

  ///
  /// SECTION: Timer 
  ///
  Timer::Timer()
  {
    tic();
  }

  void Timer::tic()
  {
    start = std::chrono::system_clock::now();
  }

  long Timer::toc()
  {
    auto stop = std::chrono::system_clock::now();
    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    return duration.count();
  }

  int longestCommonSubstring(const string&str1, const string&str2)
  {
    if(str1.empty() || str2.empty())
    {
      return 0;
    }
 
    int *curr = new int [str2.size()];
    int *prev = new int [str2.size()];
    int *swap = nullptr;
    int maxSubstr = 0;
 
    for(int i = 0; i<str1.size(); ++i)
    {
      for(int j = 0; j<str2.size(); ++j)
      {
	if(str1[i] != str2[j])
	{
	  curr[j] = 0;
	}
	else
	{
	  if(i == 0 || j == 0)
	  {
	    curr[j] = 1;
	  }
	  else
	  {
	    curr[j] = 1 + prev[j-1];
	  }
	  //The next if can be replaced with:
	  //maxSubstr = max(maxSubstr, curr[j]);
	  //(You need algorithm.h library for using max())
	  if(maxSubstr < curr[j])
	  {
	    maxSubstr = curr[j];
	  }
	}
      }
      swap=curr;
      curr=prev;
      prev=swap;
    }
    delete [] curr;
    delete [] prev;
    return maxSubstr;
  }    

  void require(bool must_be_true,string error)
  {
    if(not must_be_true)
    {
      cout << error << endl;
      assert(false);
    }
  }

  string alpha_substring(const string&input)
  {
    ostringstream fixed;
    for(char c : input)
      if(isalpha(c))
	fixed << c;
    return fixed.str();
  }
}
