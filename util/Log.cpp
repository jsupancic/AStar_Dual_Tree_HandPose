/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Log.hpp"
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <ostream>
#include <sstream>
#include "ThreadCompat.hpp"
#include "params.hpp"
#include <set>
#include "util_vis.hpp"

namespace deformable_depth
{
  using namespace std;
  
  ofstream log_file;
  ofstream log_file_locked;
  
  string uuid()
  {
    // lock
    static mutex m; unique_lock<mutex> l(m);
    
    // generate
    static boost::uuids::basic_random_generator<boost::mt19937> gen;
    boost::uuids::uuid uuid = gen();
    
    // convert
    ostringstream oss;
    oss << uuid;
    
    return oss.str();
  }
  
  void ensure(bool condition_must_be_true)
  {
    if(!condition_must_be_true)
      throw std::runtime_error("Failed an assurance!");
  }
  
  string log_im(string prefix, Mat im)
  {
    string uuid = deformable_depth::uuid();
    std::ostringstream oss;
    oss << params::out_dir();
    oss << prefix;
    oss << uuid;
    oss << ".png";
    
    log_file << "log: writing image: " << oss.str() << endl;
    if(im.type() == cv::DataType<float>::type)
      im = imageeq("",im,false,false);
    imwrite(oss.str(),im);
    return oss.str();
  }
  
  void log_once(string message,string extra_unconsidered)
  {
    // lock because set isn't thread safe
    static mutex m; unique_lock<mutex> l(m);
    static set<string> have_logged;  
    
    // if we haven't already printed this message...
    if(have_logged.find(message) == have_logged.end())
    {
      have_logged.insert(message);
      
      // log it to stdout and to the file
      log_file << "log_once: " << message << " " << extra_unconsidered << endl;
      cout << "log_once: " << message << " " << extra_unconsidered << endl;
    }    
  }
  
  void log_im_decay_freq(string prefix, Mat im)
  {
    log_im_decay_freq(prefix,[&](){return im;});
  }

  bool is_power_of_two(long count)
  {
    bool powOf2 = !(count == 0) && !(count & (count - 1));
    return powOf2;
  }

  void log_decay_freq(string prefix, std::function<void (void)> fn)
  {
    static mutex m; unique_lock<mutex> l(m);
    static map<string,long> have_logged;   
    if(have_logged.find(prefix) == have_logged.end())
      have_logged[prefix] = 1;
    else
      have_logged[prefix]++;
    
    long count = have_logged[prefix];
    bool powOf2 = is_power_of_two(count);
    if(powOf2)
      fn();
  }
  
  void log_im_decay_freq(string prefix, std::function<Mat (void)> im)
  {    
    log_decay_freq(prefix,[&](){log_im(prefix,im());});
  }

  void log_text_decay_freq(string counter, std::function<string (void)> text)
  {
    log_decay_freq(counter,[&](){log_file << text() << endl;});
  }
 
  void log_locked(string message)
  {
    static mutex m; lock_guard<decltype(m)> l(m);
    log_file_locked << message << endl;
  }
  
  void init_logs(const string&command)
  {
    ostringstream oss;
    oss << params::out_dir() << "log_" << command << "_" << g_params.get_value("LOG_PREFIX") << ".txt";
    deformable_depth::log_file.open(oss.str());
    oss << ".locked";
    deformable_depth::log_file_locked.open(oss.str());
    g_params.log_params();
  }
}
