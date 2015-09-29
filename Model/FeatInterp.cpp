/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "FeatInterp.hpp"

#include <vector>
#include <opencv2/opencv.hpp>
#include "Log.hpp"
#include "FauxLearner.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  // SECTION: FeatureInterpetation
  FeatureInterpretation::FeatureInterpretation() : total_length(0)
  {

  }
  
  string FeatureInterpretation::interp_at_pos(int start)
  {
    for(auto && interp : *this)
      if(interp.second[0] == start)
	return interp.first;
    return "NULL";
  }
  
  std::vector< double > FeatureInterpretation::select(string aspect, const std::vector< double >& w)
  {
    assert(this->find(aspect) != this->end());
    size_t first = (*this)[aspect][0];
    size_t last  = first + (*this)[aspect][1];
    return vector<double>(w.begin()+first,w.begin()+last);
  }
  
  void FeatureInterpretation::set(string aspect, const SparseVector&value, SparseVector& to) const
  {
    bool has_aspect = (this->find(aspect) != this->end());
    if(!has_aspect)
    {
      cout << "FeatureInterpretation::set missing aspect: " << aspect << endl;
      assert(false);
    }
    
    bool aspect_size_works = (*this).at(aspect)[1] == value.size();
    if(!aspect_size_works)
    {
      int expected = (*this).at(aspect)[1];
      int acutal = value.size();
      cout << printfpp("aspect size expected %d actual %d",expected,acutal) << endl;
      assert(aspect_size_works);
    }
    to.set((*this).at(aspect)[0],value);
  }  
  
  void FeatureInterpretation::init_include_append_model(ExposedLDAModel& model,string name)
  {
    int cur_length = model.getLearner().getFeatureLength();
    init_include_append_model(cur_length,name);
  }
  
  void FeatureInterpretation::init_include_append_model(int cur_length, string name)
  {
    
    log_file << "joint_interp, " << name << " length = " << cur_length << endl;
    (*this)[name] = Vec2i(total_length,cur_length);
    total_length += cur_length;    
  }

  string FeatureInterpretation::toString() const
  {
    ostringstream oss;
    
    for(auto&pair : *this)
      oss << " " << pair.first << " " << pair.second;
    
    return oss.str();
  }
  
  void FeatureInterpretation::flush(
    const vector<double>& joint_feat, std::map< string, SettableLDAModel& >& models)
  {
    for(const auto & model : models)
    {
      vector<double> part_wf = select(model.first,joint_feat);
      model.second.setLearner(new FauxLearner(part_wf,0.0f));
      model.second.update_model();      
    }
  }
    
  SparseVector FeatureInterpretation::pack(const map< string, SparseVector >& part_feats)
  {
    SparseVector feature(total_length);

    for(const auto & part_feat : part_feats)
    {
      if(part_feat.second.size() == 0)
	return vector<double>{};
      
      require_equal<size_t>(part_feat.second.size(),at(part_feat.first)[1]);
      feature.set(at(part_feat.first)[0],part_feat.second);
    }
    
    return feature;
  }
  
  void write(FileStorage& fs, string& str, const FeatureInterpretation& interp )
  {
    fs << "{";
    fs << "this"; write(fs,(std::map<string,Vec2i>&)interp);
    fs << "total_length" << (int)interp.total_length;
    fs << "}";
  }
  
  void write(FileStorage& fs, string& str, const shared_ptr< FeatureInterpretation >& interp)
  {
    write(fs,str,*interp);
  }
  
  void read(const FileNode& node, FeatureInterpretation& interp, FeatureInterpretation def)
  {
    int total_length;
    node["total_length"] >> total_length; 
    interp.total_length = total_length;
    std::map<string,Vec2i>&base_ref = interp;
    read(node["this"],base_ref);
  }
  

  void read(const FileNode& node, shared_ptr< FeatureInterpretation >& interp, 
	    shared_ptr< FeatureInterpretation > )
  {
    interp.reset(new FeatureInterpretation());
    read(node,*interp);
  }    
}
