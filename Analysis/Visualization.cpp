/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Visualization.hpp"
#include <exiv2/exiv2.hpp>

#include "Log.hpp"
#include "util_vis.hpp"
#include "util_mat.hpp"
#include "RegEx.hpp"

using namespace cv;
using namespace std;

namespace deformable_depth
{
  Visualization::Visualization()
  {
  }

  Visualization::Visualization(const Mat&image,const string&title)
  {
    insert(image,title);
  }

  void Visualization::insert(const Mat&image,const string&title)
  {
    if(images.find(title) != images.end())
    {
      string unique_title = title + uuid();
      images[unique_title] = vertCat(image_text(unique_title),image);
    }
    else
    {
      images[title] = vertCat(image_text(title),image);
    }
    update_layout();
  }

  void Visualization::insert(const Visualization&&vis, const std::string&prefix)
  {
    for(auto && pair : vis.images)
      insert(pair.second,prefix + pair.first);
  }
  
  Visualization::Visualization(const Visualization&v1,const string&prefix1,const Visualization&v2,const string&prefix2)
  {
    for(auto && image : v1.images)
      images[prefix1 + image.first] = image.second;
    for(auto && image : v2.images)
      images[prefix2 + image.first] = image.second;
    update_layout();
  }

  void Visualization::update_layout()
  {
    // compute the geometry
    double N = images.size();
    int n_rows = ceil(std::sqrt(N));
    int n_cols = ceil(N/n_rows);

    auto nth_image = [&](int n)
    {
      auto iter = images.begin();
      for(int nIter = 0; nIter < n; ++nIter, ++iter)
	;
      return *iter;
    };

    layout.clear();
    int xCursor = 0, yCursor = 0;
    for(int rIter = 0; rIter < n_rows; ++rIter)
    {
      int max_y = 0;
      for(int cIter = 0; cIter < n_cols; ++cIter)
      {
	int n = rIter * n_cols + cIter;
	if(n < images.size())
	{
	  auto im = nth_image(n);
	  Rect bb(Point(xCursor,yCursor),im.second.size());
	  layout[im.first] = bb;
	  xCursor += bb.width;
	  max_y = std::max(max_y,bb.br().y);	
	}
      }
      yCursor = max_y;
      xCursor = 0;
    }
  }

  void Visualization::write(const string&prefix) const
  {
    static mutex m; lock_guard<mutex> l(m);

    // the tricky part!!!
    string filename = log_im(string("exiv2")+prefix,image());
    
    // register DD namespace
    Exiv2::XmpProperties::registerNs("deformable_depth/", "DD");

    // generate the data
    //Exiv2::ExifData exifData;
    Exiv2::XmpData data;
    data["Xmp.DD.Visualization"] = "TRUE";
    for(auto && im : layout)
      data[string("Xmp.DD.") + im.first] = toString(im.second);

    // commit to the file
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(filename);
    image->setXmpData(data);
    image->writeMetadata();
  }

  cv::Mat Visualization::at(const string&id)
  {
    if(images.find(id) == images.end())
      return Mat();
    else
      return images.at(id);
  }

  Visualization::Visualization(const string&filename)
  {
    // (0) load the layout
    log_file << "++Visualization::Visualization" << endl;
    Exiv2::Image::AutoPtr image = Exiv2::ImageFactory::open(filename);
    image->readMetadata();
    Exiv2::XmpData&data = image->xmpData();
    for(auto && iter : data)
    {      
      ostringstream oss; oss << iter.value();
      vector<string> num_strs = deformable_depth::regex_match(oss.str(),boost::regex("\\d+"));
      if(num_strs.size() == 4)
      {
	int x = fromString<int>(num_strs.at(2));
	int y = fromString<int>(num_strs.at(3));
	int w = fromString<int>(num_strs.at(0));
	int h = fromString<int>(num_strs.at(1));
	Rect bb(Point(x,y),Size(w,h));
	layout[iter.key()] = bb;
	log_file << "Visualization::Visualization (" << iter.key() << ", " << bb << ")" << endl;
      }
    }

    // (1) load the images
    Mat im = imread(filename);
    for(auto && imElem : layout)
    {
      images[imElem.first] = im(imElem.second);
    }
    log_file << "--Visualization::Visualization" << endl;
  }

  Mat Visualization::image() const
  {
    // get the total size.
    Rect bounds;    
    for(auto && bb : layout)
    {
      bounds |= bb.second;
    }

    // now draw the big visualization
    Mat viz(bounds.height,bounds.width,DataType<Vec3b>::type,Scalar::all(0));
    for(auto && image : images)
    {
      Rect bb = layout.at(image.first);
      const Mat &im = image.second;
      assert(im.type() == DataType<Vec3b>::type);
      im.copyTo(viz(bb));
    }
    return viz;
  }
}

