/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "util_file.hpp"
#include "params.hpp"
#ifndef WIN32
#include "MetaData.hpp"
#include <sys/mman.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string>
#include "util.hpp"
#include <boost/filesystem.hpp>

namespace deformable_depth
{
  using namespace boost::filesystem;
  using namespace std;

#ifndef WIN32
  int alloc_file_atomic_unique(
    string pattern, 
    string&filename,
    atomic<unsigned long>&array_id)
  {
    int fd = -1;
    while(fd == -1)
    {
      unsigned long file_id = array_id++;
      filename = printfpp(pattern.c_str(),file_id);
      assert(filename.size() > 0);
      fd = open(filename.c_str(), O_CREAT|O_EXCL|O_WRONLY|O_TRUNC,S_IRWXU);
      if(fd == -1) 
	perror(printfpp("warning: allocated file (%s) exists!",filename.c_str()).c_str());
    }
    return fd;
  }

  string alloc_unique_filename(string pattern,atomic<unsigned long>&array_id)
  {
    string filename;
    int fd = alloc_file_atomic_unique(pattern,filename,array_id);
    close(fd);
    return filename;
  }
#endif

  template<>
  void read_in_map( const cv::FileNode& node, AnnotationBoundingBox&abb)
  {
    deformable_depth::read(node,abb,AnnotationBoundingBox());
  }

  template<>
  void read_in_map( const FileNode& node, cv::Point_<double> & p)
  {
    if(!node.isMap())
    {
      deformable_depth::read(node, p,Point2d());
    }
    else
    {
      node["x"] >> p.x;
      node["y"] >> p.y;
    }
  }

  template<>
  void write_in_map ( FileStorage& fs, string&s, const cv::Point_<double> & p)
  {
    fs << "{";
    fs << "x" << p.x;
    fs << "y" << p.y;
    fs << "}";    
  }

  template<>
  void write_in_map ( FileStorage& fs, string&s, const cv::Vec<int,2> & v)
  {
    cout << "writing a Vec2i" << endl;
    fs << "{";
    fs << "x" << v[0];
    fs << "y" << v[1];
    fs << "}";
    cout << "wrote the Vec2i" << endl;
  }    
  
  template<>
  void write_in_map ( FileStorage& fs, string&s, const cv::Vec3d&v3d)
  {
    //cout << "writing a Vec2i" << endl;
    fs << "{";
    fs << "x" << v3d[0];
    fs << "y" << v3d[1];
    fs << "z" << v3d[2];
    fs << "}";
    //cout << "wrote the Vec2i" << endl;
  }    
  
  template<>
  void read_in_map(const FileNode& node, Vec2i& result)
  {
    node["x"] >> result[0];
    node["y"] >> result[1];
  }
  
  template<>
  void read_in_map(const FileNode& node, Vec3d& v3d)
  {
    if(!node.isMap())
    {
      vector<double> data;
      node >> data;
      v3d[0] = data[0];
      v3d[1] = data[1];
      v3d[2] = data[2];
#ifndef WIN32
      if(DEBUG_YML_IO)
		cout << "read_in_map<Vec3d>: " << v3d << endl;
#endif
    }
    else
    {
      node["x"] >> v3d[0];
      node["y"] >> v3d[1];
      node["z"] >> v3d[2];
    }
  }
  
  void read(const cv::FileNode&node, Point2d& pt,Point2d default_Value)
  {
    vector<double> value;
    node >> value;
    assert(value.size() == 2);
    pt.x = value[0];
    pt.y = value[1];
  }  

  void read(const cv::FileNode&node, Point2f&pt,Point2f default_value)
  {
    vector<float> value;
    node >> value;
    assert(value.size() == 2);
    pt.x = value[0];
    pt.y = value[1];    
  }
  
  void read(const cv::FileNode& fn, Rect& rect, Rect default_Value)
  {
#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4 
    vector<double> bb_vec; 
    read(fn,bb_vec);
    assert(bb_vec.size() == 4);
    rect.x = bb_vec[0];
    rect.y = bb_vec[1];
    rect.width = bb_vec[2];
    rect.height = bb_vec[3];
#else
	cout << "unimplemented" << endl;
	exit(1);
#endif
  }
    
  void read(const FileNode& fn, Rect_< double >& read_ref, Rect_< double > def)
  {
    return read_rect(fn,read_ref,def);
  }
    
  vector< string > allStems(string dir, boost::regex regex, string pose)
  {
    vector<string> files = find_files(dir,regex);
    vector<string> stems;
    
    for(int iter =0; iter < files.size(); ++iter)
    {
		string file = files[iter];
      boost::filesystem::path path(file);
      string stem = path.stem().string();
      
#ifndef WIN32
	if(pose == "" || pose == metadata_build(dir+"/"+stem)->get_pose_name())
#endif      
	{
	  stems.push_back(stem);
	}
    }
    
    return stems;
  }
  
  vector<string> find_dirs(string dir, boost::regex regex)
  {
    vector<string> files = find_files(dir,regex);
    vector<string> dirs;

    for(string & file : files)
      if(boost::filesystem::is_directory(file))
	dirs.push_back(file);

    return dirs;
  }

  vector< string > find_files(string dir, boost::regex regex,bool recursive)
  {
    printf("allStems %s\n",dir.c_str());
    vector<string> matches;
    path direc(dir);
    int file_count = 0;
    for(boost::filesystem::directory_iterator fIter(direc);
	fIter != boost::filesystem::directory_iterator();
	fIter++)    
    {
      file_count++;
      string file = fIter->path().string();	
      if(boost::regex_match(fIter->path().string(),regex))
      //if(fIter->path().extension().string() == ext)
      {	
	matches.push_back(file);
      }
      if(boost::filesystem::is_directory(file) and recursive)
      {
	vector<string> sub_matches = find_files(file,regex,recursive);
	matches.insert(matches.end(),sub_matches.begin(),sub_matches.end());
      }
    }
    std::sort(matches.begin(),matches.end());
    printf("allStems %d of %d\n",(int)matches.size(),file_count);
    return matches;
  }
    
  vector< string > allStems(std::string dir, std::string ext, std::string pose)
  {
    return allStems(dir,boost::regex(string(".*") + ext + "$"));
  }
  
  string randomStem(string dir, string ext)
  {
    vector<string> matches = allStems(dir,ext);
    int selIdx = rand()%matches.size();
    return matches[selIdx];
  }
 
  void loadRandomRGBZ(Mat&RGB,Mat_<float>&Z,string dir,string*stem_path)
  {
    // select negatives
    string stem = randomStem(dir, ".png");
    
    FileStorage bg_file(stem+".yml", FileStorage::READ);
    bg_file["bg_depth"] >> Z;
    bg_file.release();
    RGB = imread(stem+".png");    
    
    if(stem_path != nullptr)
    {
      *stem_path = stem;
    }
  }
  
  Rect loadRect(FileStorage&fs,string name)
  {
	  assert(fs.isOpened());
      vector<int> bbVec; fs[name] >> bbVec;
	  if(bbVec.size() < 4)
		  printf("err, loadRect got %d numbers from \n",(int)bbVec.size());
      return Rect(Point2i(bbVec[0],bbVec[1]),Size(bbVec[2],bbVec[3]));
  }

  string convert(const wstring& from)
  {
    return std::string(from.begin(),from.end());
  }
  
  string convert(const string& from)
  {
    return from;
  }
  
  string get_path_seperator()
  {
#ifdef WIN32
	boost::filesystem::path slash("/");
	std::wstring preferredSlash(slash.make_preferred().native().c_str());
	return convert(preferredSlash);
#else
	return "/";
#endif
  }
  
  string yaml_fix_key(const string& orig)
  {
    static boost::regex yaml_bad_re("[-\\.]");
    string updated = boost::regex_replace(orig,yaml_bad_re,string("_"));

    if(updated.length() == 0 or not std::isalpha(updated[0]))
      updated = string("affixed") + updated;

    return updated;
  }
  
  void write(cv::FileStorage&fs, std::string&, const std::set<std::basic_string<char> >& set)
  {
    fs << "{";
    for(auto iter = set.begin(); iter != set.end(); ++iter)
      fs << *iter << "1";
    fs << "}";
  }  
  
  void read(FileNode fn, set< string >& s)
  {
    for(FileNodeIterator iter = fn.begin(); iter != fn.end(); ++iter)
    {
      s.insert((*iter).name());
    }
  }

  vector<string> parse_lines(const string&filename)
  {
    vector<string> lines;

    ifstream ifs(filename);
    while(ifs)
    {
      string line; std::getline(ifs,line);
      lines.push_back(line);
    }

    return lines;
  }

  void write(cv::FileStorage&fs, std::string&, const cv::RotatedRect&rr)
  {
    fs << "{";
    fs << "center" << rr.center;
    fs << "size" << rr.size;
    fs << "angle" << rr.angle;
    fs << "}";
  }

  void read(const cv::FileNode& node, cv::RotatedRect&rr, cv::RotatedRect)
  {
    read_in_map(node["center"],rr.center);
    deformable_depth::read(node["size"],rr.size);
    node["angle"] >> rr.angle;
  }

  Mat read_csv_double(const string&filename)
  {
    log_file << "++read_csv_double: " << filename << endl;
    Mat m;
    ifstream ifs(filename);
    if(not ifs)
    {
      cout << "error: failed to open " << filename << endl;
      assert(false);
    }

    while(ifs)
    {
      string line;
      std::getline(ifs,line);
      istringstream iss(line);
      vector<double> line_values;
      while(iss)
      {
	double value;
	string svalue;
	getline(iss,svalue,',');
	value = fromString<double>(svalue);
	line_values.push_back(value);
      } // read all numbers from line
      //cout << "[";
      //for(auto && vec : line_values)
      //cout << "(" << vec << ")";
      //cout << "]" << endl;
      Mat row = Mat(line_values).t();
      if(m.cols == 0 or row.cols == m.cols)
	m.push_back(row);
      else
	break;
    }        

    log_file << "++read_csv_double: " << m.size() << endl;
    return m.t();
  }

  void write_csv_double(const string&filename,const Mat&m)
  {
    assert(m.type() == DataType<double>::type);
    ofstream ofs(filename);
    for(int yIter = 0; yIter < m.rows; ++yIter)
      for(int xIter = 0; xIter < m.cols; ++xIter)
      {
	double v = m.at<double>(yIter,xIter);
	ofs << v;
	if(xIter < m.cols - 1)
	  ofs << ",";
	else
	  ofs << endl;
      }
  }

  void write_depth_mm(const string&filename,const Mat&m)
  {
    assert(m.type() == DataType<float>::type);
    Mat Z(m.rows,m.cols,DataType<uint16_t>::type);
    for(int yIter = 0; yIter < Z.rows; ++yIter)
      for(int xIter = 0; xIter < Z.cols; ++xIter)
      {
	Z.at<uint16_t>(yIter,xIter) = 10*m.at<float>(yIter,xIter);
      }    
    imwrite(filename,Z);
  }
  
  Mat  read_depth_mm(const string&filename)
  {
    Mat Z = imread(filename,-1);
    Z.convertTo(Z,DataType<float>::type);
    for(int yIter = 0; yIter < Z.rows; ++yIter)
      for(int xIter = 0; xIter < Z.cols; ++xIter)
      {
	float&z = Z.at<float>(yIter,xIter);
	z = z/10;
      }
    return Z;
  }
}

