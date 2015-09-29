/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_HEURISTIC_TEMPLATES
#define DD_HEURISTIC_TEMPLATES

#include <boost/filesystem.hpp>
#include "boost/multi_array.hpp"
#include "opencv2/opencv.hpp"
#include "util_mat.hpp"
#include "MetaData.hpp"
#include "params.hpp"

#include "tbb/concurrent_unordered_set.h"
#include "tbb/concurrent_unordered_map.h"

namespace deformable_depth
{
  using cv::Mat;
  using cv::Vec3d;

  extern Size SpearTemplSize;
  static constexpr int NN_DEFAULT_ZRES = 30;

  template<typename T>
  class DynMultiArray : public T
  {
  public:
    DynMultiArray() : T() {};

    template<typename U>
    DynMultiArray(U u) : T(u) {};

    DynMultiArray& operator=(const DynMultiArray&other)
    {
      //boost::array<boost::multi_array::multi_array_base::index,D> set_shape(other.shape());      
      int width = other.shape()[0];
      int height= other.shape()[1];
      //typename T::extent_gen extents[width][height];
      this->resize(boost::extents[width][height]);
      static_cast<T&>(*this) = static_cast<const T&>(other);

      return *this;
    }
  };

  // assumes an ortographic projection
  // thus we can figure out the optimal alignment in closed form.
  struct VolumetricTemplate;
  class AutoAlignedTemplate
  {
  protected:
    Mat TVis, feat_near_envelope, feat_far_envelope;
    
  public:    
    AutoAlignedTemplate(const ImRGBZ&im);
    double cor(const AutoAlignedTemplate&other,float&aligned_depth);
    double cor(const VolumetricTemplate&other,Point2i p0, float&aligned_depth);
    Vec3i resolution() const;
    AutoAlignedTemplate pyramid_down() const;    
  };

  // represents a volumetric template which is discrete in the Z axis.
  class DiscreteVolumetricTemplate
  {
  protected:
    Mat TVis, feat_near_envelope, feat_far_envelope;

  public:   
    DiscreteVolumetricTemplate(const ImRGBZ&im);    
    Vec3i resolution() const;
  };
  
  struct VolumetricTemplate
  {
  protected:
    int XRES = 30;
    int YRES = 30;
    int ZRES = 30;
    float z_min;
    //typedef DynMultiArray<boost::multi_array<uint8_t, 3> , 3> Array3D;
    typedef DynMultiArray<boost::multi_array<uint8_t, 2> > Array2D;
    //Array3D feat;
    Array2D feat_near_envelope;
    Array2D feat_far_envelope;
    bool valid;
    double cluster_size = 1;

    Mat TVis;
    //Mat color;
    // foreground segmenetation, only exists for labeled templates.
    //Mat foreground;
    RotatedRect extracted_from;
    shared_ptr<MetaData> exemplar;
    
    double simple_cor(const VolumetricTemplate&other, double admissibility) const;
    double fg_cor(const VolumetricTemplate&other) const;
    void validate() const;    
    Mat asMat(const Array2D&array2d) const;
    void setFromMat(const Mat mat,Array2D&array2d);
    
  public:
    float get_ZMin() const;
    void incClusterSize(int inc_by);
    int getClusterSize();
    Vec3i resolution() const;
    bool is_valid() const;    
    virtual ~VolumetricTemplate();
    VolumetricTemplate(const ImRGBZ&im,float z,shared_ptr<MetaData> exemplar,RotatedRect extractedFrom);
    VolumetricTemplate(Vec3i size = Vec3i(SpearTemplSize.width,SpearTemplSize.height,NN_DEFAULT_ZRES));
    double cor(const VolumetricTemplate&other,double admissibility = 1) const;
    shared_ptr<MetaData> getMetadata() const;
    Mat getTIm() const;
    Mat getTNear() const;
    Mat getTFar() const;
    const Mat&getTRef() const;
    RotatedRect getExtractedFrom() const;
    double merge_cost(const VolumetricTemplate&other);
    VolumetricTemplate merge(const VolumetricTemplate&other) const;
    VolumetricTemplate pyramid_down() const;
    bool operator== (const VolumetricTemplate&other) const;
    Mat vis_high_res() const;

    // hashing
    friend class hash<deformable_depth::VolumetricTemplate>;
    operator size_t() const;

    // serialization
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::VolumetricTemplate&);
    friend void read(const cv::FileNode&, deformable_depth::VolumetricTemplate&, deformable_depth::VolumetricTemplate);
  };
}
namespace std
{
  template <>
  struct hash<deformable_depth::VolumetricTemplate>
  {
    std::size_t operator()(const deformable_depth::VolumetricTemplate k) const;
  };
}
namespace tbb
{
  
}
 
namespace deformable_depth
{ 
  struct SpearmanTemplate
  {
  public:
    SpearmanTemplate();
    SpearmanTemplate(Mat rawIm,float z,shared_ptr<MetaData> exemplar,RotatedRect extractedFrom);
    double cor(const vector<float>&T) const;
    double cor(const SpearmanTemplate&other) const;
    shared_ptr<MetaData> getMetadata() const;
    Mat getTIm() const;
    RotatedRect getExtractedFrom() const;
    SpearmanTemplate merge(const SpearmanTemplate&O);
    
  protected:
    RotatedRect extracted_from;
    shared_ptr<MetaData> exemplar;
    vector<float> T;
    Mat TIm;   
  };  
  
  // Template which combines RGB and Depth data
  class CombinedTemplate
  {
  protected:
    SpearmanTemplate rgb_templ;
    VolumetricTemplate depth_templ;
    
  public:
    CombinedTemplate();
    CombinedTemplate(const ImRGBZ&im,float z,shared_ptr<MetaData> exemplar,RotatedRect extractedFrom);
    double cor(const CombinedTemplate&other) const;
    shared_ptr<MetaData> getMetadata() const;
    Mat getTIm() const;
    RotatedRect getExtractedFrom() const;    
  };
 
  typedef VolumetricTemplate NNTemplateType;

}

#endif

