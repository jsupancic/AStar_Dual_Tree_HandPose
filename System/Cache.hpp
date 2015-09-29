/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_CACHE
#define DD_CACHE

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <memory>

namespace deformable_depth
{
  using std::unordered_map;
  using std::unordered_set;
  using std::string;
  using std::mutex;
  using std::unique_lock;
  using std::function;
  using std::condition_variable;
  using std::shared_ptr;
  
  template<typename T>
  class Cache
  {
  protected:
    condition_variable result_ready;
    unordered_map<string,T> cache;
    unordered_set<string> started_computation;
    mutable std::mutex monitor;
    // used for the LRU caching strategy
    long capacity;    
    atomic<long> virtual_time;
    unordered_map<string,long> id_2_last_used;
    map<long,string> last_used_2_id;
    
  public:
    typedef function<T ()> ComputeFn;
    
    Cache(long capacity = std::numeric_limits<long>::max()) : 
      virtual_time(0),
      capacity(capacity) {}

    size_t size() const
    {
      unique_lock<mutex> exclusion(this->monitor);
      return cache.size();
    }
    
    T get(string key, ComputeFn computeFn)
    {
      // only allow one thread
      unique_lock<mutex> exclusion(this->monitor);
      
      // three possibilities
      if(cache.find(key) != cache.end())
      {
	// we have the value cached, just return it
      }
      else if(started_computation.find(key) != started_computation.end())
      {
	// another thread is computing the value, so we 
	// must wait...
	while(cache.find(key) == cache.end())
	  result_ready.wait(exclusion);
	//result_ready.at(key)->notify_one();
      }
      else
      {
	// tell other threads to wait for our computation
	started_computation.insert(key);
	
	// release the lock...
	exclusion.unlock();
	
	// compute the value
	T value = computeFn();
	
	// now we must re-lock the exclusion, update the map, and notify waiting threads
	exclusion.lock();
	cache.insert(std::pair<string,T>(key,value));
      }  

      // update the use times
      long time = virtual_time++;
      if(id_2_last_used.find(key) != id_2_last_used.end())
      {
	long prev_time = id_2_last_used.at(key);
	auto erase_at = last_used_2_id.find(prev_time);
	if(erase_at != last_used_2_id.end())	  
	  last_used_2_id.erase(erase_at);
      }
      id_2_last_used[key] = time;
      last_used_2_id[time] = key;

      // 
      string remove_key;
      if(cache.size() > capacity)
      {
	// free the least recently used item.
	remove_key = last_used_2_id.begin()->second;
	cache.erase(remove_key);
	started_computation.erase(remove_key);
      }
      
      // release the lock
      T result = cache.at(key);
      exclusion.unlock();
      result_ready.notify_all();
      return result;
    }
  };
}

#endif
