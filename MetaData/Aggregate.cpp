/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "Aggregate.hpp"
#include "RegEx.hpp"

namespace deformable_depth
{
  // construction and destruction
  MetaDataAggregate::MetaDataAggregate() : 
    MetaData_YML_Backed(uuid(),true)
  {
  }

  MetaDataAggregate::MetaDataAggregate(map<string,shared_ptr<MetaData_YML_Backed> > sub_data) : 
    sub_data{sub_data}, MetaData_YML_Backed(uuid(),true)
  {
    string new_name = "Aggregate:";
    for(auto && sub_datum : sub_data)
      new_name += sub_datum.second->get_filename();
    change_filename(new_name);
  }

  MetaDataAggregate::~MetaDataAggregate()
  {}
  
  // implementation of virtual methods
  void MetaDataAggregate::setSegmentation(Mat&segmentation)  
  {
    for(auto kv_pair : sub_data)
      kv_pair.second->setSegmentation(segmentation);
  }

  bool MetaDataAggregate::leftP() const 
  {
    bool has_left = false;
    bool has_right = false;
    for(auto kv_pair : sub_data)
    {
      if(kv_pair.second->leftP())
	has_left = true;
      else
	has_right = true;
    }

    if(has_left)
      return true;
    return false;
  }

  map<string,MetaData* > MetaDataAggregate::get_subdata()  const
  {
    map<string,MetaData* > data;
    for (auto && pair : sub_data)
      data.insert({pair.first,pair.second.get()});
    return data;
  }

  map<string,MetaData_YML_Backed* > MetaDataAggregate::get_subdata_yml()  
  {
    map<string,MetaData_YML_Backed* > data;
    for (auto && pair : sub_data)
      data.insert({pair.first,pair.second.get()});
    return data;
  }

  void MetaDataAggregate::set_HandBB(cv::Rect newHandBB)  
  {
    throw std::runtime_error("unsuported");
  }

  int MetaDataAggregate::keypoint()
  {
    int count = 0;
    for(auto && pair : sub_data)
      count += pair.second->keypoint();
    return count;
  }

  static void split_kp_name(const string&id,string&subdatum_id,string&name_in_subdatum)
  {
    std::vector<std::string> before = regex_matches(id, boost::regex("^(.*):"));
    std::vector<std::string> after  = regex_matches(id, boost::regex(":(.*)$"));
    assert(before.size() == 1 && after.size() == 1);
    subdatum_id = before.at(0);
    subdatum_id = subdatum_id.substr(0,subdatum_id.size()-1);
    name_in_subdatum = after.at(0);
    name_in_subdatum = name_in_subdatum.substr(1,name_in_subdatum.size()-1);
  }
  
  bool MetaDataAggregate::hasKeypoint(string name)
  {
    string subdatum_id,name_in_subdatum;
    split_kp_name(name,subdatum_id,name_in_subdatum);
    return sub_data.at(subdatum_id)->hasKeypoint(name_in_subdatum);
  }
  
  pair<Point3d,bool> MetaDataAggregate::keypoint(string name)
  {
    string subdatum_id,name_in_subdatum;
    split_kp_name(name,subdatum_id,name_in_subdatum);
    return sub_data.at(subdatum_id)->keypoint(name_in_subdatum);    
  }
  
  void MetaDataAggregate::keypoint(string name, Point3d value, bool vis)
  {
    string subdatum_id,name_in_subdatum;
    split_kp_name(name,subdatum_id,name_in_subdatum);
    return sub_data.at(subdatum_id)->keypoint(name_in_subdatum,value,vis);        
  }
  
  void MetaDataAggregate::keypoint(string name, Point2d value, bool vis)
  {
    string subdatum_id,name_in_subdatum;
    split_kp_name(name,subdatum_id,name_in_subdatum);
    return sub_data.at(subdatum_id)->keypoint(name_in_subdatum,value,vis);    
  }
  
  vector<string> MetaDataAggregate::keypoint_names()
  {
    vector<string> kp_names;

    for(auto && pair : sub_data)
    {
      vector<string> names = pair.second->keypoint_names();
      for(string & name : names)
	kp_names.push_back(pair.first + ":" + name);
    }
    return kp_names;
  }
  
  map<string,AnnotationBoundingBox > MetaDataAggregate::get_positives()  
  {
    map<string,AnnotationBoundingBox> poss;

    for(auto && pair : sub_data)
    {
      auto pos = pair.second->get_positives();
      for(auto && po : pos)
	if(po.second.size().area() > 0)
	  poss.insert({po.first,po.second});
    }

    return poss;
  }

  std::shared_ptr<ImRGBZ> MetaDataAggregate::load_im()  
  {
    return sub_data.begin()->second->load_im();
  }

  std::shared_ptr<const ImRGBZ> MetaDataAggregate::load_im() const  
  {
    const MetaData& c_subdata = *sub_data.begin()->second;
    return c_subdata.load_im();
  }

  Mat MetaDataAggregate::getSemanticSegmentation() const 
  {
    return sub_data.begin()->second->getSemanticSegmentation();
  }

  bool MetaDataAggregate::use_negatives() const 
  {
    return false;
  }

  bool MetaDataAggregate::use_positives() const 
  {
    return true;
  }
}
