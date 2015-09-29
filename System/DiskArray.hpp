/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_DISK_ARRAY
#define DD_DISK_ARRAY

#include <stddef.h>
#include <vector>
#include <memory>
#include <string>
#include "ThreadCompat.hpp"
#include "ThreadPool.hpp"

namespace deformable_depth
{
  using std::vector;

#ifdef DD_CXX11
  using std::shared_ptr;
  using std::thread;
  using std::mutex;
  using std::string;
  using std::condition_variable;
  using std::atomic;
  
  class DiskArray
  {
  protected:
    string filename;
    float* start;
    size_t numel;
    int fd;
    atomic<bool> stored;
    int mapped;
    mutex map_unmap_exclusion;
    mutex ensure_stored_exclusion;
    condition_variable write_complete;
    shared_ptr<ThreadPool::TaskType> write_task;
  public:
    DiskArray(shared_ptr<vector<float> > data);
    virtual ~DiskArray();
    float operator[](size_t idx);
    size_t size() const;
    void map();
    void unmap();
  protected:
    void write(shared_ptr<vector<float> > data);
    void ensure_stored();
  };

#endif

  class FeatIm : public vector<float>
  {
  public:
    FeatIm();
    virtual ~FeatIm();
  };  
}
#endif
