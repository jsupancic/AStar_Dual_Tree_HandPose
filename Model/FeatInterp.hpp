/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_FEAT_INTERP
#define DD_FEAT_INTERP

#include "vec.hpp"
#include <map>
#include "Detector.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

namespace deformable_depth
{
  using cv::Vec2i;
  using std::vector;
  
  // class representing the interpreation of a joint feature vector
  class FeatureInterpretation : public map<string,Vec2i /*start,length*/> 
  {
  public:
    FeatureInterpretation();
    void set(string aspect, const SparseVector&value,SparseVector&to) const;
    vector<double> select(string aspect,const vector<double>&w);
    void init_include_append_model(ExposedLDAModel&model,string semantics);
    void init_include_append_model(int feat_size,string semantics);
    string interp_at_pos(int start);
    string toString() const;
    
    // code to simplify packing and unpacking the joint features
    void flush(const vector<double>& joint_feat,map<string,SettableLDAModel&>&models);
    SparseVector pack(const map<string,SparseVector>&part_feats);
  public:
    size_t total_length;
  public:
    friend void read(const FileNode&, shared_ptr<FeatureInterpretation>&, shared_ptr<FeatureInterpretation>);
    friend void write(cv::FileStorage&, std::string&, const std::shared_ptr<FeatureInterpretation>&);
  };
    
  void read(const FileNode&, FeatureInterpretation&,FeatureInterpretation def = FeatureInterpretation());
  void read(const FileNode&, shared_ptr<FeatureInterpretation>&, shared_ptr<FeatureInterpretation>);
  void write(cv::FileStorage&, std::string&, const std::shared_ptr<FeatureInterpretation>&);  
  void write(cv::FileStorage&, std::string&, const FeatureInterpretation&);  
}
  
#endif
