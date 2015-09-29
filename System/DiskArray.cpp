/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "DiskArray.hpp"
#include "Log.hpp"
#include "util.hpp"
#include "ThreadPool.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <assert.h>
#include <thread>
#include <atomic>

namespace deformable_depth
{
  using namespace std;
  
  static atomic<unsigned long> array_id(0);
  
  void DiskArray::write(shared_ptr< vector< float > > data)
  {
    // dump the data to the disk
    fd = alloc_file_atomic_unique(
      "/scratch/jsupanci/dd_feats%lu.dat",filename,array_id);
    FILE*fp = fdopen(fd,"w");
    if(fp == NULL)
      perror("DiskArray failed to convert fd");
    assert(fp != NULL);
    size_t num_writen = fwrite((char*)&(*data)[0],sizeof(float),data->size(),fp);
    if(num_writen != data->size())
    {
      cout << "num_writen: " << num_writen << endl;
      cout << "data->size: " << data->size() << endl;
    }
    assert(num_writen == data->size());
    fclose(fp);
    unique_lock<mutex> exclusion(ensure_stored_exclusion);
    stored = true;
    //atomic_thread_fence(std::memory_order_seq_cst);
    __sync_synchronize();
    write_complete.notify_all();
  }
  
  void DiskArray::map()
  {
    {
      lock_guard<mutex> exclusion(map_unmap_exclusion);
      
      mapped++;
      if(mapped == 1)
      { 
	ensure_stored();
	
	// now, map the disk into virtual memory
	fd = open(filename.c_str(),O_RDONLY);
	if(fd == -1) perror("DiskArray::write_and_map failed open for reading");
	assert(fd != -1);
	
	do
	{
	  start = (float*)mmap(0,numel*sizeof(float),PROT_READ,MAP_SHARED|MAP_NORESERVE,fd,0);
	  if(start == MAP_FAILED) 
	  {
	    perror("DiskArray::write_and_map failed!");
	    cout << "filename" << filename << endl;
	    cout << "size = " << size() << endl;
	    sleep(1);
	  }
	} while(start == MAP_FAILED);
	assert(start != MAP_FAILED);
      }
    }
  }

  void DiskArray::unmap()
  {
    {
      lock_guard<mutex> exclusion(map_unmap_exclusion);
      
      mapped--;
      if(mapped == 0)
      {
	munmap(start,numel*sizeof(float));
	close(fd);
      }
    }
  }
  
  void DiskArray::ensure_stored() 
  {
    unique_lock<mutex> exclusion(ensure_stored_exclusion);
    while(!stored)
    {
      chrono::seconds watchdog_timeout(10);
      write_complete.wait_for(exclusion,watchdog_timeout);
    }
  }
  
  DiskArray::DiskArray(shared_ptr<vector< float > > data) : 
    stored(false), mapped(0), 
    write_task(shared_ptr<ThreadPool::TaskType>(
      new ThreadPool::TaskType([this,data](){this->write(data);},"DiskArray::DiskArray")))
  {
    //log_file << "warning: DiskArray uses deprecated threading system" << endl;
    numel = data->size();
    
    IO_Pool->schedule_task(write_task);
  }

  float DiskArray::operator[](size_t idx) 
  {
    return start[idx];
  }

  size_t DiskArray::size() const
  {
    return numel;
  }

  DiskArray::~DiskArray()
  {
    ensure_stored();
    remove(filename.c_str());
  }
  
  
  // SECTION: Feature Image
  FeatIm::FeatIm()
  {
    //log_file << "FeatIm " << this << dec << " of size " << size() << " allocated" << endl;
    //cout << "FeatIm " << this << dec << " of size " << size() << " allocated" << endl;
  }

  FeatIm::~FeatIm()
  {
    //log_file << "FeatIm " << this << dec << " of size " << size() << " destroyed" << endl;
    //cout << "FeatIm " << this << dec << " of size " << size() << " destroyed" << endl;
  }  
}
