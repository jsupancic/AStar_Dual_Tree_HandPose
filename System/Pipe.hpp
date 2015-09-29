/**
 * Copyright 2012: http://programmersbook.com/page/14/C0x-thread-safe-queue/
  * Copyright 2012: James Supancic III
 **/

#include <queue>
#include <thread>
#include "Semaphore.hpp"
#include <assert.h>

namespace deformable_depth
{
  using std::queue;
  using std::mutex;
  using std::lock_guard;
  
  template<typename T>
  class Pipe
  {
  private:
    queue<T> Q;
    mutex lock;
    Semaphore s;
  public:
    Pipe() : s(0)
    {
    }
    
    void push(const T&t)
    {
      lock_guard<mutex> l(lock);
      Q.push(t);
      s.V();
    }

    T pull(bool clear = false)
    {
      s.P();
      lock_guard<mutex> l(lock);
      assert(!Q.empty());
      if(clear)
      {
	T t = Q.front();
	Q = queue<T>();
	s.set(0);
	return t;
      }
      else
      {
	T t = Q.front();
	Q.pop();
	return t;
      }
    }

    bool empty()
    {
      lock_guard<mutex> l(lock);
      return Q.empty();
    }
  };
}
