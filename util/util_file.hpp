/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_UTIL_FILE
#define DD_UTIL_FILE

#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include <boost/regex.hpp>
#include <string>
#include "ThreadCompat.hpp"
#include <set>
#include "MetaData.hpp"
#include "Log.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
  
#ifdef DD_CXX11
  static constexpr bool DEBUG_YML_IO = false;
#else
#define DEBUG_YML_IO false
#endif

  void read(const cv::FileNode&, Point2d&,Point2d default_Value);
  void read(const cv::FileNode&, Point2f&,Point2f default_value);
  void read(const cv::FileNode&, Rect&, Rect default_Value = Rect());

  template<typename T>
  void read(const cv::FileNode& node, cv::Size_<T>& size, cv::Size_<T> = cv::Size_<T>())
  {
    vector<T> value;
    node >> value;
    assert(value.size() == 2);
    size.width = value[0];
    size.height = value[1];
  }
  
  ///
  /// SECTION: Read/Write cv::Vec<T,D>
  ///
  
  template<typename T>
  void read(const cv::FileNode&node, Vec<T,3>& vec2i, Vec<T,3> = Vec<T,3>())
  {
    vector<int> value;
    node >> value;
    assert(value.size() == 3);
    vec2i[0] = value[0];
    vec2i[1] = value[1];
    vec2i[2] = value[2];
  }    
  template<typename T>
  void read(const cv::FileNode&node, Vec<T,2>& vec2i, Vec<T,2> = Vec<T,2>())
  {
    vector<int> value;
    node >> value;
    assert(value.size() == 2);
    vec2i[0] = value[0];
    vec2i[1] = value[1];
  }  
  template void read(const cv::FileNode&node, Vec<int,2>& vec2i, Vec<int,2> );
  
  ///
  /// SECTION: Read/Write std::map
  ///
  // read/write a map using OpenCV
  
  template<typename V>
  void read_in_map ( const cv::FileNode& node, V &result )
  {
    if(DEBUG_YML_IO)
      cout << "reading a value" << endl;
    read(node,result,V());
  }  
  
  template<typename V>
  void read_in_map ( const cv::FileNode& node, map<string,V>& result )
  {
    if(DEBUG_YML_IO)
      cout << "reading a map" << endl;
    read(node,result);
  }    
  
  template<>
  void read_in_map( const FileNode& node, cv::Point_<double> & p);  
  template<>
  void read_in_map ( const cv::FileNode& node, Vec2i &result);
  template<>
  void read_in_map ( const cv::FileNode& node, cv::Vec3d&v3d);
  template<>
  void read_in_map( const cv::FileNode& node, AnnotationBoundingBox&abb);

  template<typename V>
  void read ( const cv::FileNode& node, map< string, V >&result)
  {
    bool node_type_ok = (node.type() & FileNode::MAP) > 0;
    if(!node_type_ok)
    {
      cout << node.type() << endl;
      cout << (node.type() & FileNode::MAP) << endl;
      cout << ((node.type() & FileNode::MAP) > 0) << endl;
      assert(node_type_ok);
    }
    
    for(FileNodeIterator iter = node.begin(); iter != node.end(); ++iter)
    {
      string node_name = (*iter).name();
      V value;
      //iter >> value; 
      deformable_depth::read_in_map(*iter,value);
      result[node_name] = value;
    }
  }

  //
  // section write_in_map
  //
    
  template<typename V>
  void write_in_map ( FileStorage& fs, string&s, const V& v)
  {
    //cout << "writing a value " << fs.state << endl;
    //fs << "{";
    //write(fs,s,v);
    fs << v;
    //fs << "}";
    //cout << "wrote the value" << endl;
  }

  template<>
  void write_in_map ( FileStorage& fs, string&s, const cv::Point_<double> & p);  
  template<>
  void write_in_map ( FileStorage& fs, string&s, const cv::Vec<int,2> & v);
  template<>
  void write_in_map ( FileStorage& fs, string&s, const cv::Vec3d&v3d);
  
  template<typename V>
  void write_in_map ( FileStorage& fs, string&s, const vector<V>& vec)
  {
    //cout << "writing the vector" << endl;
    fs << "[";
    for(auto&&value : vec)
      fs << value;
    fs << "]";
    //cout << "wrote the vector" << endl;
  }

  // for sanitizing keys for YAML's tastes  
  string yaml_fix_key(const string&orig);
  
  template<typename V>
  void write_seq_as_map(FileStorage& fs, string&s, const vector<V>& vec)
  {
    fs << "{";
    for(auto && elem : vec)
      fs << yaml_fix_key(uuid()) << elem;
    fs << "}";
  }

  template<typename V>
  void write_in_map ( FileStorage& fs, string&s, const map<string,V>& v)
  {
    //cout << "writing the map" << endl;
    write(fs,s,v);
    //cout << "wrote the map" << endl;
  }      
      
  template<typename V>
  void write ( FileStorage& fs, string&s, const map< string, V >& m )
  {
    fs << "{";
    for(typename map<string,V>::const_iterator iter = m.begin();
	iter != m.end(); ++iter)
    {
      // sanitize iter->first w/ RE
      string key_name = yaml_fix_key(iter->first);
      
      // write to key_name
      //cout << "key: " << key_name << endl;
      string s;
      fs << key_name; write_in_map(fs,s,(const V&)iter->second);
    }
    fs << "}";
  }

  template<typename V>
  void write ( FileStorage& fs, const map< string, V >& m )
  {
    string s;
    deformable_depth::write(fs,s,m);
  }    
      
  ///
  /// SECTION: Read/Write std::set
  /// 
  void write(cv::FileStorage&fs, std::string&, const std::set<std::basic_string<char> >& set);
  void read(cv::FileNode fn, std::set<std::string >&);
  
  // allocate a file with a given pattern, ensuring 
  // it is unique (avoid race conditions with other programs).
  int alloc_file_atomic_unique(string pattern,string&filename,atomic<unsigned long>&array_id);
  string alloc_unique_filename(string pattern,atomic<unsigned long>&array_id);
  
  vector<string> allStems(string dir, boost::regex regex, string pose = "");
  vector<string> allStems(string dir, string ext, string pose = "");
  vector<string> filesWithExt(string dir, string ext);
  vector<string> find_files(string dir, boost::regex regex, bool recursive = false);
  vector<string> find_dirs(string dir, boost::regex regex);
  string randomStem(string dir, string ext);
  ///
  /// SECTION: Rectangle IO
  ///
  void write(cv::FileStorage&, std::string&, const cv::RotatedRect&);
  void read(const cv::FileNode&, cv::RotatedRect&, cv::RotatedRect = cv::RotatedRect());
  Rect loadRect(cv::FileStorage&fs,string name);
  template<typename T>
  void read(const cv::FileNode& fn, cv::Rect_<T>&rect_out, cv::Rect_<T>)
  {
    vector<T> bbVec; fn >> bbVec; // assumes: x y w h
    rect_out.x = bbVec[0];
    rect_out.y = bbVec[1];
    rect_out.width = bbVec[2];
    rect_out.height = bbVec[3];
  }
  template<typename T>
  void read_rect(const cv::FileNode& fn, cv::Rect_<T>&rect_out, cv::Rect_<T>)
  {
    vector<T> bbVec; fn >> bbVec; // assumes: x y w h
    rect_out.x = bbVec[0];
    rect_out.y = bbVec[1];
    rect_out.width = bbVec[2];
    rect_out.height = bbVec[3];
  }
  void read(const cv::FileNode&, cv::Rect_<int>&, cv::Rect_<int>);
  void read(const cv::FileNode&, cv::Rect_<double>&, cv::Rect_<double>);
  
  void loadRandomRGBZ(Mat&RGB,Mat_<float>&Z,string dir,string*stem_path=nullptr);
  string get_path_seperator();
  string convert(const wstring&from);
  string convert(const string &from);
  vector<string> parse_lines(const string&filename);  

  Mat read_csv_double(const string&filename);
  void write_csv_double(const string&filename,const Mat&m);

  void write_depth_mm(const string&filename,const Mat&m);
  Mat  read_depth_mm(const string&filename);
}

#endif
