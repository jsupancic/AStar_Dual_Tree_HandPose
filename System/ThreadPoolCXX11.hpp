/**
 * Copyright 2013: James Steven Supancic III
 **/


#ifndef DD_THREAD_POOL_CXX11
#define DD_THREAD_POOL_CXX11

#include <memory>
#include <vector>
#include <functional>
#include "ThreadCompat.hpp"

#include "params.hpp"

namespace deformable_depth
{  
  using std::vector;
  using std::function;
  using std::shared_ptr;
  
  class ThreadPool
  {
  public:
    struct TaskType
    {
    public:
      packaged_task<void ()> task;
      string account;
      TaskType(function<void ()> call,string account);
    };
    
  public:
    ThreadPool(int thread_count);
    virtual ~ThreadPool();
    void schedule_task(shared_ptr<TaskType> task);
    void assist_until(function<bool ()> done);
    void shutdown();
    void print_accounts();
  protected:
    void work();
    void do_job(TaskType&task,unique_lock<mutex>&exclusion);
    shared_ptr<TaskType> takeNextJob();
    
    atomic<bool> is_shutdown;
    mutex monitor;
    condition_variable work_queue_changed;
    condition_variable work_available;
    vector<std::thread> worker_threads;
    map<int,vector< shared_ptr<TaskType > > > tasks;
    map<string,long> accounts; // track how many seconds each account used
  };
  
  using std::unique_ptr;  
  extern unique_ptr<ThreadPool> default_pool;    
  extern unique_ptr<ThreadPool> IO_Pool;
  extern unique_ptr<ThreadPool> empty_pool;

  class TaskBlock
  {
  public:
    void add_callee(function<void ()> f);
    void execute(ThreadPool&pool = *default_pool);
    bool done();
    TaskBlock(string account_id);
  protected:
    vector<shared_ptr<ThreadPool::TaskType > > tasks;
    vector<int> complete;
    string account_id;
  };    
  
  // map vector<T> to vector<U>
  template<typename U,typename T,typename InputIterator>
  vector<U> par_map(
    InputIterator iter,
    InputIterator end,
    function<U (T&)> mapper)
  {
    // store map results
    vector<U> map_results;
    TaskBlock map_tasks;
    // schedule the maps
    for(int idx = 0; iter != end; idx++)
    {
      T&t = *iter;
      map_results.push_back(U());
      U&u = map_results.back();
      map_tasks.add_callee(
	[mapper,&t,&u]()
	{
	  u = mapper(t);
	});
      iter++;
    }
    
    // wait until the tasks have completed
    map_tasks.execute(*default_pool);
    
    return map_results;
  }
}

#endif
