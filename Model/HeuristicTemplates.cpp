/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "util_file.hpp"
#include "HeuristicTemplates.hpp"
#include <queue>
#include <opencv2/opencv.hpp>
#include "Orthography.hpp"
#include "util.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv; 
  Mat calc_spearman_template(const Mat&mat,float z_min, const Size&size,int&nBackground,int&nForeground);
  Size SpearTemplSize(40,40);

  ///
  /// SECTION: Spearman Template
  /// 
  double SpearmanTemplate::cor(const SpearmanTemplate& other) const
  {
    // normalize: each template contains 0*0 + 1*1 + 2*2 + ... + (N-1)*(N-1)
    double n = other.T.size();
    double nf = std::pow(n-1,3)/3 + std::pow(n-1,2)/2 + (n-1)/6;
    return dot(other.T,this->T)/nf;
  }
  
  shared_ptr< MetaData > SpearmanTemplate::getMetadata() const
  {
    return exemplar;
  }

  double SpearmanTemplate::cor(const vector< float >& T) const
  {
    return dot(this->T,T);
  }

  static bool check_padded(const Mat&padded,float z_max)
  {
    float center_z = padded.at<float>(padded.rows/2,padded.cols/2);
    if(center_z > z_max)
    {
      //imageeq("padded",padded,true,true);
      log_file << safe_printf("warning: bad center % vs %",center_z,z_max) << endl;
      return false;  
    }
    
    double valid_count = 0;
    for(int rIter = 0; rIter < padded.rows; rIter++)
      for(int cIter = 0; cIter < padded.cols; cIter++)
      {
	const float & z_here = padded.at<float>(rIter,cIter);
	if(z_here < z_max and z_here  < params::MAX_Z() and goodNumber(z_here))
	  valid_count++;
      }
    double validity_ratio = static_cast<double>(valid_count)/static_cast<double>(padded.size().area());
    
    if(std::sqrt(validity_ratio) < .3)
    {
      log_file << "warning: validity_ratio = " << validity_ratio << endl;
      return false; 
    }
    
    return true;
  }

  SpearmanTemplate::SpearmanTemplate()
  {
  }

  SpearmanTemplate::SpearmanTemplate(Mat rawIm,float z,
				     shared_ptr<MetaData> exemplar, RotatedRect extracted_from) : 
    exemplar(exemplar), extracted_from(extracted_from)
  {
    int ignore1,ignore2;    
    float z_max = z + fromString<float>(g_params.require("OBJ_DEPTH"));
    Mat padded = resize_pad(rawIm,SpearTemplSize);
    if(!check_padded(padded,z_max))
      TIm = Mat();
    else
      TIm = calc_spearman_template(rawIm,z,SpearTemplSize,ignore1,ignore2);
    T = TIm.clone().reshape(0,1);
  }
  
  Mat SpearmanTemplate::getTIm() const
  {
    return TIm;
  }
  
  RotatedRect SpearmanTemplate::getExtractedFrom() const
  {
    return extracted_from;
  }
  
  Mat calc_spearman_template(const Mat&mat,float z_min, const Size&size,int&nBackground,int&nForeground)
  {
    // size up the template
    float z_max = z_min + fromString<float>(g_params.require("OBJ_DEPTH"));
    Mat padded = resize_pad(mat,size);
    assert(padded.isContinuous());
    assert(padded.type() == DataType<float>::type);
    
    // randomize the background
    nBackground = 0;
    nForeground = 0;
    static thread_local std::mt19937 sample_seq(19860);
    uniform_int_distribution<int> id_dist(0,numeric_limits<int>::max());
    std::uniform_real_distribution<float> z_dist(params::MAX_Z(),params::MAX_Z()+100);
    for(int rIter = 0; rIter < padded.rows; rIter++)
      for(int cIter = 0; cIter < padded.cols; cIter++)
      {
	float & z_here = padded.at<float>(rIter,cIter);
	// set the background pixels
	if(z_here >= z_max || z_here  >= params::MAX_Z() || !goodNumber(z_here))
	{
	  z_here = z_dist(sample_seq);
	  nBackground++;
	}
	else if(z_here <= 0 or z_here < z_min)
	  nForeground++;
      }
    
    // compute the Spearman feature
    struct Pixel
    {
      Mat*im;
      int row, col;
      int rnd_id;
      
      Pixel(Mat*im,int row, int col, int rnd_id) : 
	im(im), row(row), col(col), rnd_id(rnd_id)
      {	
      }
      
      bool operator < (const Pixel&rhs) const
      {
	double z1 = im->at<float>(row,col);
	double z2 = rhs.im->at<float>(rhs.row,rhs.col);
	if(z1 == z2)
	{
	  return rnd_id < rhs.rnd_id;
	}
	else
	  return z1 < z2;
      }
    };    
    vector<Pixel> pixels;
    pixels.reserve(padded.size().area());
    for(int rIter = 0; rIter < padded.rows; rIter++)
      for(int cIter = 0; cIter < padded.cols; cIter++)
	pixels.push_back(Pixel(&padded,rIter,cIter,id_dist(sample_seq)));
    std::sort(pixels.begin(),pixels.end());
    int rank = 0;
    for(auto & pixel : pixels)
      padded.at<float>(pixel.row,pixel.col) = rank++;
    
    return padded;
  }  
  
  ///
  /// SECTION: VolumetricTemplate
  ///
  void VolumetricTemplate::validate() const
  {
    require_equal<size_t>(feat_far_envelope.shape()[0] ,   XRES);
    require_equal<size_t>(feat_far_envelope.shape()[1] ,   YRES);
    require_equal<size_t>(feat_near_envelope.shape()[0] ,  XRES);
    require_equal<size_t>(feat_near_envelope.shape()[1] ,  YRES);
  }

  bool VolumetricTemplate::is_valid() const
  {
    return valid and
      feat_far_envelope.shape()[0] == XRES and      
      feat_far_envelope.shape()[1] == YRES and
      feat_near_envelope.shape()[0] == XRES and
      feat_near_envelope.shape()[1] == YRES;
  }

  double VolumetricTemplate::simple_cor(const VolumetricTemplate& other, double admissibility) const
  {
    // check invarients
    other.validate();
    validate();
    for(int dim = 0; dim < 2; ++dim)
    {
      require_equal<size_t>(feat_far_envelope.shape()[dim] ,  other.feat_far_envelope.shape()[dim]);
      require_equal<size_t>(feat_near_envelope.shape()[dim] ,  other.feat_near_envelope.shape()[dim]);
    }

    auto match_one = [this](double here_near,double other_near,double here_far,double other_far)
    {
      double matches = 0;

      //matches += (ZRES - std::abs(
      //this->feat2d[xIter][yIter] - other.feat2d[xIter][yIter]));
      matches += std::min<double>(here_near,other_near);			       
      matches += std::min<double>(ZRES - 1 - here_far,ZRES - 1 - other_far);

      return matches;
    };

    // compute the distance, supressing background
    double matches = 0;
    double matches_average = 0;
    
    for(int yIter = 0; yIter < YRES; ++yIter)
      for(int xIter = 0; xIter < XRES; ++xIter)
      {
	// compute the admissible heuristic
	matches += match_one(this->feat_near_envelope[xIter][yIter],
			     other.feat_near_envelope[xIter][yIter],
			     this->feat_far_envelope[xIter][yIter],
			     other.feat_far_envelope[xIter][yIter]);
      }

    double admissible_score = matches/(XRES*YRES*ZRES);
    if(admissibility >= 1)
      return admissible_score;

    for(int yIter = 0; yIter < YRES; ++yIter)
      for(int xIter = 0; xIter < XRES; ++xIter)
      {
	// compute the inadmissible heuristic
	matches_average += match_one(this->TVis.at<float>(yIter,xIter),
				     other.TVis.at<float>(yIter,xIter),
				     this->TVis.at<float>(yIter,xIter),
				     other.TVis.at<float>(yIter,xIter));
      }
        
    double average_score = matches_average/(XRES*YRES*ZRES);
    return admissibility * admissible_score + (1 - admissibility) * average_score;
  }
  
  double VolumetricTemplate::cor(const VolumetricTemplate& other, double admissibility) const
  {
    return simple_cor(other, admissibility);
  }
  
  shared_ptr< MetaData > VolumetricTemplate::getMetadata() const
  {
    return exemplar;
  }

  VolumetricTemplate::VolumetricTemplate(Vec3i size) : 
    XRES(size[0]),
    YRES(size[1]),
    ZRES(size[2]),
    //feat(boost::extents[XRES][YRES][ZRES]),
    feat_near_envelope(boost::extents[size[0]][size[1]]),
    feat_far_envelope(boost::extents[size[0]][size[1]]),
    valid(false),
    TVis(size[1],size[0],DataType<float>::type,Scalar::all(qnan))
  {
    validate();
  }

  VolumetricTemplate::~VolumetricTemplate()
  {
    validate();
    // todo?    
  }

  Mat VolumetricTemplate::vis_high_res() const
  {
    assert(exemplar != nullptr);
    Rect handBB = getExtractedFrom().boundingRect();//exemplar->get_positives().at("HandBB");
    shared_ptr<ImRGBZ> im = exemplar->load_im();
    Mat vr(handBB.height,handBB.width,DataType<float>::type,Scalar::all(qnan));
    assert(handBB.size().area() > 0);

    for(int yIter = 0; yIter < vr.rows; yIter++)
      for(int xIter = 0; xIter < vr.cols; xIter++)
      {
	int inY = clamp<int>(0,yIter + handBB.y,im->Z.rows-1);
	int inX = clamp<int>(0,xIter + handBB.x,im->Z.cols-1);
	float z_in = im->Z.at<float>(inY,inX);
	if(z_in < params::obj_depth() + z_min)
	  vr.at<float>(yIter,xIter) = std::min(z_in - z_min,params::obj_depth());
      }

    return vr;
  }

  VolumetricTemplate::VolumetricTemplate(const ImRGBZ&im, float z_min, 
					 shared_ptr< MetaData > exemplar, 
					 RotatedRect extractedFrom) :
    XRES(SpearTemplSize.width),
    YRES(SpearTemplSize.height),
    extracted_from(extractedFrom),
    exemplar(exemplar),
    //feat(boost::extents[XRES][YRES][ZRES]),
    feat_near_envelope(boost::extents[XRES][YRES]),
    feat_far_envelope(boost::extents[XRES][YRES]),
    valid(false),
    z_min(z_min)
  {
    // extract the padded template and 
    // check that the window is valid...
    float z_max = z_min + fromString<float>(g_params.require("OBJ_DEPTH"));
    int background_bin = 1*(ZRES-1);//ZRES-1;
    //int foreground_bin = background_bin; // or 0?
    int foreground_bin = 0;
    Mat padded = resize_pad(im.Z,Size(XRES,YRES));    

    // covert to orthographic, if needed
    log_im_decay_freq("VolumetricTemplate",[&]()
		      {
			return horizCat(imageeq("",padded,false,false),imageeq("",im.Z,false,false));
		      });
    // if(exemplar and !check_padded(padded,z_max))
    // {
    //   Mat vis_padded = imageeq("",padded,false,false);
    //   log_im_decay_freq("unreal_padded",vis_padded);
    //   return;
    // }

    // conditionally apply morphological operations
    //if(exemplar)
    //for(int iter = 0; iter < 2; ++iter)
    //padded = imclose(padded);

    // compute the ordinal coordinates
    int nBackground = 0;
    int nForeground = 0;
    Mat spearman = calc_spearman_template(padded,z_min, SpearTemplSize,nBackground,nForeground);
    int nArea = SpearTemplSize.area();
    int nValid = nArea - nBackground;
    assert(spearman.type() == DataType<float>::type);

    // init the members
    //foreground = Mat(YRES,XRES,DataType<uint8_t>::type,Scalar::all(0));
    //color = resize_pad(im.RGB,Size(XRES,YRES)).clone();
    TVis = Mat(YRES,XRES,DataType<float>::type,Scalar::all(qnan));

    // compute the feature representation
    for(int yIter = 0; yIter < YRES; ++yIter)
      for(int xIter = 0; xIter < XRES; ++xIter)
      {
	float depth = padded.at<float>(yIter,xIter);
	//int inter_val = interpolate_linear(depth,z_min,z_max,1,ZRES-1);
	int inter_val = interpolate_linear(spearman.at<float>(yIter,xIter),0,nValid-1,1,ZRES-1);
	int zbin = 
	  (depth < z_min)?foreground_bin:
	  (depth > z_max)?background_bin:
	  inter_val;
	//TVis.at<float>(yIter,xIter) = zbin;	
	feat_near_envelope[xIter][yIter] = zbin;
	feat_far_envelope[xIter][yIter] = zbin;
	TVis.at<float>(yIter,xIter) = .5*(feat_near_envelope[xIter][yIter] + feat_far_envelope[xIter][yIter]);
	
	//if(zbin == background_bin)
	//foreground.at<uint8_t>(yIter,xIter) = 255;
	
	//for(int zIter = 0; zIter < ZRES; ++zIter)
	  //if(zIter < zbin)
	    //feat[xIter][yIter][zIter] = false;
	  //else
	    //feat[xIter][yIter][zIter] = true;
      }

    validate();
    valid = true;
  }

  Mat VolumetricTemplate::getTIm() const
  {
    return TVis.clone();
  }

  Mat VolumetricTemplate::getTNear() const
  {
    return asMat(feat_near_envelope);
  }

  Mat VolumetricTemplate::getTFar() const
  {
    return asMat(feat_far_envelope);
  }

  const Mat& VolumetricTemplate::getTRef() const
  {
    return TVis;
  }
  
  RotatedRect VolumetricTemplate::getExtractedFrom() const
  {
    return extracted_from;
  }
  
  // return a volumetric template with half the resolution
  VolumetricTemplate VolumetricTemplate::pyramid_down() const
  {
    double ps = params::pyramid_sharpness();
    VolumetricTemplate down(Vec3i(ps*XRES,ps*YRES,ps*ZRES));
    down.cluster_size = this->cluster_size;        

    // initialize the envelopes
    for(int yIter = 0; yIter < down.XRES; yIter++)
      for(int xIter = 0; xIter < down.YRES; xIter++)
      {
	down.feat_near_envelope[xIter][yIter] = 0;
	down.feat_far_envelope[xIter][yIter] = down.ZRES - 1;
	down.TVis.at<float>(yIter,xIter) = 0;
      }

    // spatial pool
    for(int yIter = 0; yIter < YRES; ++yIter)
      for(int xIter = 0; xIter < XRES; ++xIter)
      {
	// corresponds to an OR over the sub-voxels
	int downX = clamp<int>(0,ps*xIter,down.XRES-1);
	int downY = clamp<int>(0,ps*yIter,down.YRES-1);
	down.feat_near_envelope[downX][downY] = 
	  std::max<uint8_t>(
	    down.feat_near_envelope[downX][downY],
	    ps*this->feat_near_envelope[xIter][yIter]); 
	// corresponds to an AND over the sub-voxels
	down.feat_far_envelope[downX][downY] = 
	  std::min<uint8_t>(
	    down.feat_far_envelope[downX][downY],
	    ps*this->feat_far_envelope[xIter][yIter]);
	
	// corresponds to an average over the sub-voxels
	if(not this->TVis.empty())
	{
	  float & m_val = down.TVis.at<float>(downY,downX);
	  float t_val   = this->TVis.at<float>(yIter,xIter);
	  m_val += std::pow(ps,3)*t_val; // 4 subwindows, halving the ZBINs
	}
      }

    if(this->TVis.empty())
      down.TVis = this->TVis;

    return down;
  }

  VolumetricTemplate VolumetricTemplate::merge(const VolumetricTemplate& other) const
  {
    VolumetricTemplate merged;
    merged.TVis = TVis.clone();
    assert(this->cluster_size > 0);
    assert(other.cluster_size > 0);
    merged.cluster_size = this->cluster_size + other.cluster_size;
    
    for(int yIter = 0; yIter < YRES; ++yIter)
      for(int xIter = 0; xIter < XRES; ++xIter)
      {
	merged.feat_near_envelope[xIter][yIter] = 
	  std::max(
	    feat_near_envelope[xIter][yIter],
	    other.feat_near_envelope[xIter][yIter]);
	merged.feat_far_envelope[xIter][yIter] = 
	  std::min(
	    feat_far_envelope[xIter][yIter],
	    other.feat_far_envelope[xIter][yIter]);
	//merged.TVis.at<float>(yIter,xIter) = std::min(
	//TVis.at<float>(yIter,xIter),other.TVis.at<float>(yIter,xIter));
	// the merged is the average
	//merged.TVis.at<float>(yIter,xIter) = 
	//.5*(merged.feat_near_envelope[xIter][yIter] + merged.feat_far_envelope[xIter][yIter]);
	// the weighted average
	float & m_val = merged.TVis.at<float>(yIter,xIter);
	float t_val   = this->TVis.at<float>(yIter,xIter);
	float o_val   = other.TVis.at<float>(yIter,xIter);
	m_val = (this->cluster_size/merged.cluster_size) * t_val + (other.cluster_size/merged.cluster_size) * o_val;
      }
    
    return merged;
  }

  double VolumetricTemplate::merge_cost(const VolumetricTemplate& other)
  {
    double cost = 0;
    VolumetricTemplate merged = merge(other);
    
    for(int yIter = 0; yIter < YRES; ++yIter)
      for(int xIter = 0; xIter < XRES; ++xIter)
      {
	cost += std::max(
	  std::abs(merged.feat_near_envelope[xIter][yIter] - feat_near_envelope[xIter][yIter]),
	  std::abs(merged.feat_near_envelope[xIter][yIter] - other.feat_near_envelope[xIter][yIter])
	);

	cost += std::max(
	  std::abs(merged.feat_far_envelope[xIter][yIter] - feat_far_envelope[xIter][yIter]),
	  std::abs(merged.feat_far_envelope[xIter][yIter] - other.feat_far_envelope[xIter][yIter])
	);
	
// 	cost += 
// 	  std::abs(
// 	    feat_near_envelope[xIter][yIter] -
// 	    other.feat_near_envelope[xIter][yIter]);
// 	cost += 
// 	  std::abs(
// 	    feat_far_envelope[xIter][yIter] -
// 	    other.feat_far_envelope[xIter][yIter]);
      }    
    
    return cost;
  }

  float VolumetricTemplate::get_ZMin() const
  {
    return z_min;
  }

  void VolumetricTemplate::incClusterSize(int inc_by)
  {
    cluster_size += inc_by;
  }

  int VolumetricTemplate::getClusterSize()
  {
    return cluster_size;
  }

  Vec3i VolumetricTemplate::resolution() const
  {
    return Vec3d(XRES,YRES,ZRES);
  }

  bool VolumetricTemplate::operator==(const VolumetricTemplate& other) const
  {
    return 
      feat_near_envelope == other.feat_near_envelope && 
      feat_far_envelope == other.feat_far_envelope;
  }

  void VolumetricTemplate::setFromMat(const Mat mat,Array2D&array2d)
  {
    assert(mat.type() == DataType<uint8_t>::type);
    array2d.resize(boost::extents[mat.cols][mat.rows]);

    for(int rIter = 0; rIter < mat.rows; rIter++)
      for(int cIter = 0; cIter < mat.cols; cIter++)
	array2d[cIter][rIter] = mat.at<uint8_t>(rIter,cIter);    
  }

  Mat VolumetricTemplate::asMat(const Array2D&array2d) const
  {
    int cols = feat_far_envelope.shape()[0];
    int rows = feat_far_envelope.shape()[1];
    Mat mat(rows,cols,DataType<uint8_t>::type,Scalar::all(0));
    for(int rIter = 0; rIter < rows; rIter++)
      for(int cIter = 0; cIter < cols; cIter++)
	mat.at<uint8_t>(rIter,cIter) = array2d[cIter][rIter];
    return mat;
  }

  void read(const cv::FileNode&node, deformable_depth::VolumetricTemplate&templ, deformable_depth::VolumetricTemplate)
  {
    node["XRES"] >> templ.XRES;
    node["YRES"] >> templ.YRES;
    node["ZRES"] >> templ.ZRES;
    node["valid"] >> templ.valid;
    node["cluster_size"] >> templ.cluster_size;
    deformable_depth::read(node["extracted_from"],templ.extracted_from);
    string exemplar_file;
    node["exemplar_file"] >> exemplar_file;
    templ.exemplar = metadata_build(exemplar_file,true,false);
    assert(templ.exemplar);
    log_once(safe_printf("read: %s",templ.exemplar->get_filename()));
    // read the big stuff
    templ.setFromMat(read_linked(node["feat_near_envelope"]),templ.feat_near_envelope);
    templ.setFromMat(read_linked(node["feat_far_envelope"]),templ.feat_far_envelope);
    templ.TVis = read_linked(node["TVis"]);
    templ.TVis.convertTo(templ.TVis,DataType<float>::type);
  }

  void write(cv::FileStorage&fs, std::string&, const deformable_depth::VolumetricTemplate&templ)
  {
    string s;

    fs << "{";
    fs << "XRES" << templ.XRES;
    fs << "YRES" << templ.YRES;
    fs << "ZRES" << templ.ZRES;
    fs << "valid" << templ.valid;
    fs << "cluster_size" << templ.cluster_size;
    fs << "extracted_from"; deformable_depth::write(fs,s,templ.extracted_from);    
    fs << "exemplar_file" << (templ.exemplar?templ.exemplar->get_filename():"NONE");
    // write the big stuff (matrices)
    fs << "feat_near_envelope" << write_linked(templ.asMat(templ.feat_near_envelope));
    fs << "feat_far_envelope" << write_linked(templ.asMat(templ.feat_far_envelope));
    fs << "TVis" << write_linked(templ.TVis);
    fs << "}";
  }

  VolumetricTemplate::operator  size_t() const
  {
    std::size_t hash_val = 1986;

    for(int yIter = 0; yIter < this->YRES; ++yIter)
      for(int xIter = 0; xIter < this->XRES; ++xIter)
      {
	uint8_t ne = this->feat_near_envelope[xIter][yIter];
	uint8_t fe = this->feat_far_envelope[xIter][yIter];
	hash_val ^= std::hash<uint8_t>()(ne);
	hash_val ^= std::hash<uint8_t>()(fe);	
	hash_val = deformable_depth::rol(hash_val);
      }

    return hash_val;
  }
}

namespace std
{
  std::size_t hash<deformable_depth::VolumetricTemplate>::operator()(const deformable_depth::VolumetricTemplate k) const
  {
    return k;
  }
}
 
namespace deformable_depth
{ 
  ///
  /// SECTION: CombinedTemplate
  ///
  CombinedTemplate::CombinedTemplate()
  {
  }

  CombinedTemplate::CombinedTemplate(
    const ImRGBZ&rawIm, float z, shared_ptr< MetaData > exemplar, RotatedRect extractedFrom) : 
    rgb_templ(rawIm.skin(),z,exemplar,extractedFrom),
    depth_templ(rawIm,z,exemplar,extractedFrom)
  {
  }
  
  double CombinedTemplate::cor(const CombinedTemplate& other) const
  {
    double pcolor = rgb_templ.cor(other.rgb_templ);
    double pdepth = depth_templ.cor(other.depth_templ); 
    return weighted_geometric_mean(vector<double>{pcolor,pdepth},vector<double>{.95,.05});
  }
  
  RotatedRect CombinedTemplate::getExtractedFrom() const
  {
    return rgb_templ.getExtractedFrom();
  }
  
  shared_ptr< MetaData > CombinedTemplate::getMetadata() const
  {
    return rgb_templ.getMetadata();
  }
  
  Mat CombinedTemplate::getTIm() const
  {
    if(rgb_templ.getTIm().empty() || depth_templ.getTIm().empty())
      return Mat();
    return horizCat(rgb_templ.getTIm(),depth_templ.getTIm());
  }
}

namespace deformable_depth
{
  ///
  /// SECTION: AutoAigned Template
  ///
  AutoAlignedTemplate::AutoAlignedTemplate(const ImRGBZ&im)
  {
    TVis = paint_orthographic(im);
    feat_near_envelope = TVis.clone();
    feat_far_envelope = TVis.clone();      
  }

  double AutoAlignedTemplate::cor(const VolumetricTemplate&other,Point2i p0, float&aligned_depth)
  {
    // figure out what the ROI is.
    Vec3i tRes = other.resolution();
    Point2i p1(p0.x + tRes[0],p0.y + tRes[1]);
    Rect roi(p0,p1);

    // compute the optimal alignment
    //Scalar Tmu = 
    return qnan;
  }

  Vec3i AutoAlignedTemplate::resolution() const
  {
    return Vec3i(TVis.cols,TVis.rows,-1);
  }

  ///
  /// SECTION: AutoAigned Template
  ///
  DiscreteVolumetricTemplate::DiscreteVolumetricTemplate(const ImRGBZ&im)
  {
    TVis = paint_orthographic(im);
    feat_near_envelope = TVis.clone();
    feat_far_envelope = TVis.clone();
  }

  Vec3i DiscreteVolumetricTemplate::resolution() const
  {
    return Vec3i(TVis.cols,TVis.rows,-1);
  }
}
