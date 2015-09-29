/**
 * Copyright 2012: James Steven Supancic III
 **/
 
#include "util.hpp"
#include "ThreadCompat.hpp"

namespace deformable_depth
{
  void repaint_background_thread()
  {
    if(!have_display())
      return;
    
    timespec sleep_duration, remainder;
    sleep_duration.tv_sec = 0;
    sleep_duration.tv_nsec = 10000000;
    
    while(true)
    {
      waitKey_safe(10);
      nanosleep(&sleep_duration,&remainder);
    }
  }  

  static bool is_launched = false;
  
  bool background_repainter_running()
  {
    return is_launched;
  }

  bool launch_background_repainter()
  {
    if(g_params.option_is_set("NO_BACKGROUND_REPAINTER"))
      return false;

    {
      static mutex m; lock_guard<mutex> l(m);
      
      if(!is_launched)
	std::thread(repaint_background_thread).detach();
      is_launched = true;
    }
    return true;
  }
  
  string get_hostname()
  {
    static string hostname = "";
    if(hostname == "")
    {
      static mutex m; lock_guard<mutex> l(m);
      char hn[128];
      gethostname(hn,127);
      hostname = string(hn);
    }
    return hostname;
  }
}
