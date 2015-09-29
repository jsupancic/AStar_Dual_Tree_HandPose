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

#include "Semaphore.hpp"
#include "params.hpp"
#include "ThreadCompat.hpp"

namespace deformable_depth
{
  Semaphore_LIFO active_worker_thread(params::cpu_count());
  
  /// SECTION: LIFO Semaphore
  Semaphore_LIFO::Semaphore_LIFO(int init) : 
    S(init)
  {
  }
  
  void Semaphore_LIFO::V()
  {
    lock_guard<mutex> lock(exclusion);
    S++; // atomic
    SGtZero.notify_all();
  }

  void Semaphore_LIFO::P()
  {
    unique_lock<mutex> lock(exclusion);
#ifdef DD_CXX11
    lifo.push_back(std::this_thread::get_id());
    while(S <= 0 || lifo.back() != std::this_thread::get_id())
#else
    lifo.push_back(this_thread::get_id());
    while(S <= 0 || lifo.back() != this_thread::get_id())      
#endif
      SGtZero.wait(lock);
    lifo.pop_back();
    // we have the mutex
    
    // go
    S--;
  }
  
  void Semaphore_LIFO::set(int val)
  {
    S = val;
  }
  
  /// SECTION: Regular Semaphore
  
  Semaphore::Semaphore(int init)
  {
    S = init;
  }

  void Semaphore::V()
  {
    lock_guard<mutex> lock(exclusion);
    // we have the lock
    
    S++;
    SGtZero.notify_one();
    
    // lock released automatically upon return.
  }


  void Semaphore::P()
  {
    // wait until we can go
    unique_lock<mutex> lock(exclusion);
    while(S <= 0)
      SGtZero.wait(lock);
    // we have the mutex
    
    // go
    S--;
    
    // mutex is released in unique_lock dtor.
  }

  void Semaphore::set(int val)
  {
    S = val;
  }
}
