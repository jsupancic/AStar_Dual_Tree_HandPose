/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_THREAD_COMPAT
#define DD_THREAD_COMPAT

#ifdef WIN32
///
/// SECTION: Visual Studio 2010 on Windows
///
#include <boost/thread.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/atomic.hpp>
using boost::recursive_mutex;
using boost::thread;
using boost::unique_lock;
using boost::atomic;
using boost::packaged_task;
using boost::lock_guard;
using boost::condition_variable;
#define this_thread boost::this_thread
using boost::mutex;
namespace std
{
	inline bool isnan(double v)
	{
		return _isnan(v);
	}
}
#else
///
/// SECTION: C++11 on Linux
///

#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace deformable_depth
{
  using std::mutex;
  using std::condition_variable;
  using std::atomic;
  using std::packaged_task;
  using std::unique_lock;
  using std::thread;
  using std::lock_guard;
}
#endif

namespace deformable_depth
{
}

#endif
