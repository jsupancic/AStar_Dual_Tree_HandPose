/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "ThreadPool.hpp"
#include "util.hpp"
#include "Log.hpp"

namespace deformable_depth
{
  using namespace std;
  
  unique_ptr<ThreadPool> default_pool; 
  unique_ptr<ThreadPool> IO_Pool;
  unique_ptr<ThreadPool> empty_pool;
  
  static __thread int thread_depth = 0;
  
  /// SECTION: ThreadPool Implementation
  ThreadPool::ThreadPool(int thread_count) :
    is_shutdown(false)
  {
    for(int iter = 0; iter < thread_count; iter++)
      worker_threads.push_back(std::thread([&](){work();}));
  }
  
  ThreadPool::~ThreadPool()
  {
    print_accounts();
    shutdown();
  }
  
  void ThreadPool::schedule_task(shared_ptr<ThreadPool::TaskType> task)
  {
    lock_guard<mutex> exclusion(monitor);
    tasks[thread_depth+1].push_back(task);
    work_queue_changed.notify_all();
    work_available.notify_all();
  }
  
  void ThreadPool::shutdown()
  {
    // notify the workers that it's time to shutdown
    {
      unique_lock<mutex> exclusion(monitor);
      // can't double shutdown...
      if(is_shutdown)
	return;
      
      is_shutdown = true;
      // wakeup everyone
      work_available.notify_all();
      work_queue_changed.notify_all();
      exclusion.unlock();
    }
    
    // wait for all workers to joinup
    for(std::thread & t : worker_threads)
      t.join();    
  }
  
  shared_ptr< ThreadPool::TaskType > ThreadPool::takeNextJob()
  {
    if(tasks.empty())
      return nullptr;
    
    // find the right level queue and make sure it isn't empty
    vector< shared_ptr<TaskType > > & level_queue = tasks.rbegin()->second;
    if(level_queue.empty())
    {
      tasks.erase(--tasks.rbegin().base());
      return takeNextJob();
    }
    
    // take a job from the queue
    thread_depth = tasks.rbegin()->first;
    shared_ptr<TaskType> job = level_queue.back();
    level_queue.pop_back();
    if(level_queue.empty())
      tasks.erase(--tasks.rbegin().base());
    
    return job;
  }
  
  void ThreadPool::work()
  {
    auto time_to_die = [&](){return is_shutdown && tasks.empty();};
    
    while(!time_to_die())
    {
      // wait for work
      std::unique_lock<mutex> exclusion(monitor);
      while(tasks.empty())
      {
	if(time_to_die())
	  return;
	work_available.wait(exclusion);
	if(time_to_die())
	  return;
      }
      
      // get a job
      shared_ptr<TaskType> job = takeNextJob();
      
      // release the lock and do it
      do_job(*job,exclusion);
    }
  }
  
  void ThreadPool::print_accounts()
  {
    log_file << "======== THREAD POOL ACCONTS ========" << endl;
    unique_lock< mutex > exclusion(monitor);
    for(auto&& account : accounts)
      log_file << printfpp("%s %d",account.first.c_str(),account.second) << endl;
  }
  
  void ThreadPool::do_job(ThreadPool::TaskType& task, unique_lock< mutex >& exclusion)
  {
    exclusion.unlock();
    //__sync_synchronize();
    auto start = std::chrono::system_clock::now();
    string account = task.account;
    auto task_future = task.task.get_future();
    task.task();
    try
    {
      task_future.get();
    }
    catch(std::exception&e)
    {
      log_file << "Exception thrown in job: " << e.what() << endl;
      breakpoint();
    }
    catch(...)
    {
      log_file << "Exception thrown... but it's a mystery... results undefined" << endl;
      breakpoint();
    }
    auto end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>
                          (end-start).count();
    //atomic_thread_fence(std::memory_order_seq_cst);
    //__sync_synchronize();
    exclusion.lock();
    accounts[account] += elapsed_seconds;
    work_queue_changed.notify_all();
  }
  
  // try to do a job, if we can't wait until all jobs are complete
  void ThreadPool::assist_until(function<bool ()> done)
  {
    while(true)
    {
      std::unique_lock<mutex> exclusion(monitor);
      if(done())
	return;
      else if(tasks.empty())
      {
	chrono::seconds watchdog_timeout(30);
	work_queue_changed.wait_for(exclusion,watchdog_timeout);
      }
      else
      {
	// get a job and do it
	shared_ptr<TaskType> job = takeNextJob();
	do_job(*job,exclusion);
      }
    }
  }
  
  ThreadPool::TaskType::TaskType(function< void ()> call, string account) : 
    task(call), account(account)
  {

  }
  
  /// SECTION: TaskBlock implementation
  TaskBlock::TaskBlock(string account_id) : account_id(account_id)
  {
  }
  
  void TaskBlock::add_callee(function< void() > f)
  {
    int id = complete.size();
    tasks.push_back(shared_ptr<ThreadPool::TaskType>( 
    new ThreadPool::TaskType(
      [this,f,id]()
      {
	f();
	this->complete[id] = true;
      },account_id)));
    complete.push_back(false);
  }
  
  void TaskBlock::execute(ThreadPool& pool)
  {
    for(auto&&task : tasks)
      pool.schedule_task(task);
    
    pool.assist_until([this](){return done();});
    //pool.print_accounts();
  }
  
  bool TaskBlock::done()
  {
    for(bool && b : complete)
      if(!b)
	return false;
    return true;
  }
}
