/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_PTRS
#define DD_PTRS

namespace deformable_depth
{
  template<typename T>
  class valid_ptr
  {
  protected:
    T*m_ptr;

  public:
    valid_ptr(T*ptr) : m_ptr(ptr) 
    {
      if(m_ptr == nullptr)
	throw std::runtime_error("nullpointer");
    };
    bool operator==(const valid_ptr<T>&other)
    {
      return m_ptr == other.m_ptr;
    }
    bool operator< (const valid_ptr<T>&other)
    {
      return m_ptr < other.m_ptr;
    }   
    T* get() const
    {
      return m_ptr;
    }
    T& operator*() const
    {
      return *m_ptr;
    }
    T* operator->() const
    {
      return get();
    }
  };
  template<typename T>
  void write(cv::FileStorage&fs, std::string s, const deformable_depth::valid_ptr<T>&vp)
  {    
    write(fs,s,*vp);
  }
}

namespace std
{
  template<typename T>
  class hash<deformable_depth::valid_ptr<T> >
  {
  public:
    size_t operator()(const deformable_depth::valid_ptr<T>&v) const
    {
      return std::hash<T*>(v.get());
    }
  };
}

#endif


