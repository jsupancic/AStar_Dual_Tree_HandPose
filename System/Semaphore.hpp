/*********************************************************************** 
 visTrack: Copyright (C) 2013 - James Steven Supancic III
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
 ***********************************************************************/

#include "ThreadCompat.hpp"

#ifndef VISTRACK__Semaphore_H
#define VISTRACK__Semaphore_H

#include <vector>

namespace deformable_depth
{
  using std::vector;
  
  class Semaphore
  {
  public:
    void P();
    void V();
    void set(int val);
    Semaphore(int init);
#ifdef CXX11
    Semaphore(const Semaphore&) = delete;
#endif
  private:
    int S;
    condition_variable SGtZero;
    mutex exclusion;
  };  
  
  class Semaphore_LIFO
  {
  public:
    void P();
    void V();
    void set(int val);
    Semaphore_LIFO(int init);
#ifdef CXX11
    Semaphore_LIFO(const Semaphore_LIFO&) = delete;
#endif
  private:
    int S;
    condition_variable SGtZero;
    mutex exclusion;
    vector<thread::id> lifo;
  };
  
  extern Semaphore_LIFO active_worker_thread;
}

#endif
