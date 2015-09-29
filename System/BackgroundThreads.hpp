/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_BG_THREADS
#define DD_BG_THREADS

namespace deformable_depth
{
  bool background_repainter_running();
  bool launch_background_repainter();
  string get_hostname();
}

#endif
