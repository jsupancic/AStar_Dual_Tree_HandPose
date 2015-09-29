/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "OneFeatureModel.hpp"
#include "util_rect.hpp"
#include "util.hpp"
#include <sstream>
#include <boost/graph/graph_concepts.hpp>
#include "vec.hpp"
#include "Log.hpp"
#include "RespSpace.hpp"
#include "DiskArray.hpp"
#include "MetaData.hpp"
#include "DepthFeatures.hpp"
#include "OcclusionReasoning.hpp"
#include "GlobalFeat.hpp"
#include "SRF_Model.hpp"

namespace deformable_depth
{
  /// DenseOneFeatureModel, SparseOneFeatureModel, OccAwareLinearModel
  typedef OccAwareLinearModel DefaultOneFeatureModelType;
  
  bool is_valid_pos_bb(Rect BB, const ImRGBZ&im)
  {
    try
    {
      DefaultOneFeatureModelType(BB.size(),im.RGB.size());
    }
    catch(BadBoundingBoxException ex)
    {
      return false;
    }
    return true;
  }
      
  OneFeatureModel::OneFeatureModel(shared_ptr<IHOGComputer_Factory> comp_factory) : 
    comp_factory(comp_factory)
    //comp_factory(shared_ptr<IHOGComputer_Factory>(new AdapterFactory<FeatureCacher>(comp_factory)))
  {
  }
  
  OneFeatureModel::OneFeatureModel(
    Size gtSize, Size ISize, 
    double C, double area,
    shared_ptr<IHOGComputer_Factory> comp_factory,
    shared_ptr<LDA> learner) : 
    Model_RigidTemplate(gtSize, ISize, C, area, learner), 
    comp_factory(comp_factory),
    world_area_variance(DEFAULT_world_area_variance) // 1.5 seems reasonable for anything?
    //comp_factory(shared_ptr<IHOGComputer_Factory>(new AdapterFactory<FeatureCacher>(comp_factory)))
  {
    hog = comp_factory->build(ISize,s_cell);
    hog1 = comp_factory->build(TSize,s_cell);
        
    // log
    log_file << "Feature Area = " << area << endl;
  }
    
  OneFeatureModel::~OneFeatureModel()
  {
    delete hog; 
    delete hog1;
  }
  
  void OneFeatureModel::setWorldAreaVariance(double world_area_variance)
  {
    this->world_area_variance = world_area_variance;
  }
  
  template<typename IM_TYPE>
  vector<float> extractBlocks
    (IM_TYPE feat_im,Size featIm_sz,
     Point2i block0, Point2i block1, int nbins, bool all_valid = false) 
  {
    // verbose error checking
    bool good_coords = 
      block1.x < featIm_sz.width &&
      block1.y < featIm_sz.height &&
      block0.x >= 0 &&
      block0.y >= 0;
    if(!good_coords)
    {
      printf("block0 = (%d, %d)\n",block0.y,block0.x);
      printf("block1 = (%d, %d)\n",block1.y,block1.x);
      printf("subimage size in blocks = (%d, %d)\n",
	    block1.y - block0.y + 1,
	    block1.x - block0.x + 1);
      printf("featIm Size: (%d, %d)\n",(int)featIm_sz.height,(int)featIm_sz.width);
    }
    assert(good_coords);
    
    // map subim coords to full im coords
    int blocks_y = featIm_sz.height;
    int blocks_x = featIm_sz.width;
    assert(feat_im.size() == nbins*blocks_x*blocks_y);
    auto getIdx = [blocks_y,nbins]
    (int blockX, int blockY, int bin) -> int
    {
      return 
	blockX*(blocks_y*nbins) + 
	blockY*(nbins) + 
	bin;
    };
    
    // we need to go x first to construct the HOG feature like OpenCV does.
    vector<float> subim;
    int pushes = 0;
    for(int xIter = block0.x; xIter <= block1.x; xIter++)
      for(int yIter = block0.y; yIter <= block1.y; yIter++)
      {
	//printf("Adding block (%d, %d) to subim\n",yIter,xIter);
	for(int bIter = 0; bIter < nbins; bIter++)
	{
	  int idx = getIdx(xIter,yIter,bIter);
	  if(idx < feat_im.size() && idx >= 0)
	  {
	    subim.push_back(feat_im[idx]);
	    pushes++;
	  }
	  else
	  {
	    assert(!all_valid);
	    // bb not fully contained in image.
	    subim.push_back(0);
	    pushes++;
	  }
	}
      }
    
    assert(subim.size() == pushes);
    assert(pushes == nbins*(block1.x-block0.x+1)*(block1.y-block0.y+1));
    assert(subim.size() == nbins*(block1.x-block0.x+1)*(block1.y-block0.y+1));
    return subim;
  }  
  
  DepthFeatComputer& OneFeatureModel::getHOG1()
  {
    return *hog1;
  }
    
  SparseVector OneFeatureModel::extractPos_w_border(
    const ImRGBZ&im, AnnotationBoundingBox bb, 
    IHOGComputer_Factory*scale_fact,function<void (DepthFeatComputer* comp)> config) const
  { 
    // validate args
    if(bb.area() <= 0)
      return vector<float>();
    if(bb.area() <= min_area())
      return vector<float>();
    
    // extract the proper patch from the image
    double xScale = static_cast<double>(TSize.width+2*hog1->getCellSize().width)/TSize.width;
    double yScale = static_cast<double>(TSize.height+2*hog1->getCellSize().height)/TSize.height;
    Rect bb_prime = rectResize(bb,xScale,yScale);
    if(bb.area() <= 0)
      cout << printfpp("bb = %d %d %d %d",bb.x,bb.y,bb.br().x,bb.br().y) << endl;
    assert(bb.area() > 0);
    assert(bb_prime.area() > 0);
    Size SExtract(TSize.width+2*hog1->getCellSize().width,
		  TSize.height+2*hog1->getCellSize().height);    
    assert(im.rows() > 0 && im.cols() > 0);
    const ImRGBZ t = (im)(bb_prime).resize(SExtract);    
    
    // allcoate the extractor
    assert(hog1->getCellSize().width == hog1->getCellSize().height);
    if(scale_fact == nullptr) 
      scale_fact = &*comp_factory;
    unique_ptr<DepthFeatComputer> hog_extract(
      scale_fact->build(SExtract,(int)hog1->getCellSize().width));
    unique_ptr<DepthFeatComputer> scale_1(
      scale_fact->build(hog1->getWinSize(),(int)hog1->getCellSize().width));
    config(&*hog_extract);
    config(&*scale_1);
    
    // get the feature from the patch
    static mutex log_mutex;
    if(log_mutex.try_lock())
    {
      image_safe("log_pos_rgb",t.RGB);
      imageeq("pos_depth",t.Z);
      log_mutex.unlock();
    }
    assert(t.rows() == SExtract.height);
    assert(t.cols() == SExtract.width);
    vector<float> xpos;
    hog_extract->compute(t,xpos);
    if(xpos.size() == 0)
      return xpos;
    for(float&feat : xpos)
      assert(goodNumber(feat));
    int x2 = 1 + scale_1->blocks_x() - 1;
    int y2 = 1 + scale_1->blocks_y() - 1;
    Size extracted_block_size(
      hog_extract->blocks_x(),
      hog_extract->blocks_y());
    xpos = extractBlocks(xpos,extracted_block_size,
			 Point(1,1),Point(x2,y2),scale_1->getNBins(),true);
    for(float&feat : xpos)
      assert(goodNumber(feat));
    return xpos;    
  }
  
  // extract a plane from the image
  static Mat plane(vector<float> vec,int start, int stride, Size size)
  {
    // first extract the plane from vec
    vector<float> linear_templ;
    for(int iter = start; iter < vec.size(); iter += stride)
    {
      linear_templ.push_back(vec[iter]);
    }
    
    // second, reshape to aspect
    if(linear_templ.size() != size.area())
    {
      printf("linear_templ.size() = %d\n",(int)linear_templ.size());
      printf("size.area() = %d\n",(int)size.area());
    }
    assert(linear_templ.size() == size.area());
    
    // now, for whatever reason, OpenCV HOG features
    // are stored column by column not row by row like its matrices.
    Mat templ(size.width,size.height,DataType<float>::type,&linear_templ[0]);
    templ = templ.t();
    return templ.clone();
  }  
  
  Size OneFeatureModel::resp_impl(Mat& resp, 
				  std::vector< float >& im_feats, 
				  const ImRGBZ& im) const
  {    
    // crop image to multiple of cell size.
    auto im_crop = hog->cropToCells(im);
    //imageeq("Detecting on Resized:",im.Z,true,false);
    //printf("detect_at_scale, image resized to (%d, %d)\n",im.rows,im.cols);
    // HOG window should match image size?
    
    // create a separate hog instance for this scale
    DepthFeatComputer* hog_for_scale = comp_factory->build(im_crop->RGB.size(),s_cell);
    
    // get HOG from image
    hog_for_scale->compute(*im_crop,im_feats);
    //hog_for_scale->show("HOG@Scale",vec_f2d(im_feats));
    
    // run convolutions
    assert(hog_for_scale->cellsPerBlock() == 1);
    int stride = hog_for_scale->getNBins();
    resp = Mat(hog_for_scale->blocks_y(),hog_for_scale->blocks_x(),
	       CV_32F,Scalar::all(learner->getB()));
    for(int binIter = 0; binIter < hog_for_scale->getNBins(); binIter++)
    {
      int start = binIter;
      Mat plane_im = plane(im_feats,start,stride,
			   Size(hog_for_scale->blocks_x(),hog_for_scale->blocks_y()));
      Mat plane_w  = plane(learner->getWf(),start,stride,Size(nx,ny));
      Mat plane_resp;
      filter2D(plane_im,plane_resp,-1/*same*/,plane_w,Point(0,0)/*kern center*/);
      assert(resp.type() == plane_resp.type());
      if(resp.size() != plane_resp.size())
      {
	printf("resp.size = %d %d\n",resp.size().height,resp.size().width);
	printf("plane.size = %d %d\n",plane_resp.size().height,plane_resp.size().width);
	assert(resp.size() == plane_resp.size());
      }
      //imageeq("plane_im",imVGA(plane_im));
      //imageeq("plane_resp",imVGA(plane_resp)); 
      //waitKey_safe(0);
      resp += plane_resp;
    }    
    //log_once(printfpp("Dot Product count = %d",
		      //hog_for_scale->blocks_x()*hog_for_scale->blocks_y()));
  
    //imageeq("RESP",resp);
    Size feat_sz(hog_for_scale->blocks_x(),hog_for_scale->blocks_y());
    delete hog_for_scale;
    return feat_sz;
  }
  
  Mat OneFeatureModel::resps(const ImRGBZ& im) const
  {
    vector<float> im_feats;
    Mat resp;
    resp_impl(resp,im_feats,im);
    return resp;
  }
  
  cv::Rect_< double > OneFeatureModel::externBB(int row, int col) const
  {
    float bx_stride = hog->getBlockStride().width;
    float by_stride = hog->getBlockStride().height;    
    Point ul(.5*bx_stride + bx_stride*col,.5*by_stride + by_stride*row);
    Size sz(s_cell*nx,s_cell*ny);
    return Rect_<double>(ul,sz);
  }
  
  DetectionSet OneFeatureModel::externDets(
    vector< Intern_Det >& idets, string filename,
    shared_ptr<FeatIm > im_feats, Size feat_sz, DetectionFilter filter, double scale) const
  { 
    int N = std::min<int>(idets.size(),filter.nmax);
    DetectionSet dets;
    for(int iter = 0; iter < N; iter++)
    {
      const Intern_Det&intern = idets[iter];
      shared_ptr<Detection> det(new Detection());
      assert(det != nullptr);
      
      // set resp (easy part).
      det->resp = intern.resp;
      det->depth = intern.depth;
      
      // this is the tricky part, find the BB and feature
      det->BB = externBB(intern.row,intern.col);
      
      Point2i p0(intern.col,intern.row);
      // subtract 1 because blocks 1 less than cells
      // subtract 1 because <=
      Point2i p1(intern.col + nx - 1,intern.row + ny - 1);
      // skip detections not contained in window
      if(p0.x < 0 || p0.y < 0 || p1.x >= feat_sz.width || p1.y >= feat_sz.height)
	continue;
      
      // record the filename it came from
      det->src_filename = filename;
      
      // and feats
      if(!filter.supress_feature)
      {
	int nbins = hog->getNBins();
	det->feature = [im_feats,feat_sz,p0,p1,nbins](){
	  //feat_array->map();
	  vector<float> feat = extractBlocks(*im_feats,feat_sz,p0,p1,nbins);
	  //feat_array->unmap();
	  return SparseVector(feat);};
	bool DEBUG_EXTERN = false;
	if(DEBUG_EXTERN && det->feature().size() != learner->getW().size())
	{
	  printf("p0 = (%d, %d)\n",p0.y,p0.x);
	  printf("p1 = (%d, %d)\n",p1.y,p1.x);	  
	  printf("hog1.blocks = (%d, %d)\n",hog1->blocks_x(),hog1->blocks_y());
	  printf("ny,nx = (%d, %d)\n",ny,nx);
	  printf("det.feature.size() = %d\n",(int)det->feature().size());
	  printf("learner->getW().size() = %d\n",(int)learner->getW().size());
	  assert(det->feature().size() == learner->getW().size());
	}
      }
      
      // record the scale
      det->scale_factor = scale;
      
      dets.push_back(det);
    }
    //log_file << printfpp("OneFeatureModel externed %d of %d dets",
	//		 (int)dets.size(),(int)idets.size()) << endl;
    
    return dets;
  }  
  
  Mat OneFeatureModel::show(const std::string& title)
  {
    return hog1->show(title,learner->getW()); waitKey_safe(1);
  }
  
  OneFeatureModelBuilder::OneFeatureModelBuilder(
    double C, double area,
    shared_ptr<IHOGComputer_Factory> fc_factory, double minArea, double world_area_variance) :
  C(C), AREA(area), fc_factory(fc_factory), minArea(minArea), 
  world_area_variance(world_area_variance)
  {
    lda_source = [this](){return shared_ptr<LDA>(new QP(this->C));};
  }
    
  Model* OneFeatureModelBuilder::build(cv::Size gtSize, cv::Size imSize) const
  {
    DefaultOneFeatureModelType* result = 
      new DefaultOneFeatureModelType(
	gtSize,imSize,C,AREA,fc_factory,lda_source());
    result->setMinArea(minArea);
    result->setWorldAreaVariance(world_area_variance);
    return result;
  }

  string OneFeatureModelBuilder::name() const
  {
    std::ostringstream oss;
    oss << "RigidModel: ";
    oss << "Feat=" << "??";
    oss << "C=" << C;
    oss << "AREA=" << AREA;
    return oss.str();
  }  
  
  void OneFeatureModel::init_post_construct()
  {
    this->hog = this->comp_factory->build(this->ISize,this->s_cell);
    this->hog1 = this->comp_factory->build(this->TSize,this->s_cell);   
  }
  
  /// SECTION: Serialization
  void write(FileStorage& fs, string& , const shared_ptr< OneFeatureModel >& model)
  {
    fs << "{";
    fs << "Model_RigidTemplate" << *(Model_RigidTemplate*)&*model;
    fs << "area" << model->area;
    fs << "comp_factory" << "DEFAULT";
    fs << "world_area_variance" << model->world_area_variance;
    
    // write the type
    fs << "feat_type" << model->hog->toString();
    
    fs << "}";
  }
  
  void read(const FileNode&node, 
	    shared_ptr< OneFeatureModel >& model, 
	    shared_ptr< OneFeatureModel > )
  {
    if(model == nullptr)
    {
      string type; node["feat_type"] >> type;
      if(type == "ComboComputer_Depth")
      {
	model.reset(new DefaultOneFeatureModelType(
	  shared_ptr<IHOGComputer_Factory>(new COMBO_FACT_DEPTH())));       
      }
      else if(type == "ComboComputer_RGBPlusDepth")
      {
	model.reset(new DefaultOneFeatureModelType(
	  shared_ptr<IHOGComputer_Factory>(new COMBO_FACT_RGB_DEPTH()))); 
      }
      else if(type == "NullFeatComp")
	model.reset(new DefaultOneFeatureModelType(
	  shared_ptr<IHOGComputer_Factory>(new NullFeat_FACT()))); 
      else
	throw std::runtime_error("unknow feature type!");
    }
    
    if(!node["Model_RigidTemplate"].empty())
      read(node["Model_RigidTemplate"],*(Model_RigidTemplate*)&*model);
    else
      read(node,*(Model_RigidTemplate*)&*model);
    node["area"] >> model->area;
    node["world_area_variance"] >> model->world_area_variance;   
    
    model->init_post_construct();
  }
  
  void read(const FileNode& fn, 
	    shared_ptr< FeatureExtractionModel >& model, 
	    shared_ptr< FeatureExtractionModel > )
  {
    model.reset(new FeatureExtractionModel(
	shared_ptr<IHOGComputer_Factory>(new DistInvarDepth_FACT())));       
    
    shared_ptr<OneFeatureModel> of_ptr = std::dynamic_pointer_cast<OneFeatureModel>(model);
    read(fn,of_ptr,shared_ptr< OneFeatureModel >());
    
    model->prime_learner();
  }
  
  /// SECTION: Dense Detection operations
  static void handle_idet(int rIter, int cIter, float resp, float depth,
			  DetectionFilter&filter,vector<Intern_Det>&idets)
  {
    Intern_Det newDet;
    newDet.row = rIter;
    newDet.col = cIter;
    newDet.resp = resp;
    newDet.depth = depth;
    if(newDet.resp > filter.thresh && idets.size() < filter.nmax)
      idets.push_back(newDet);    
  }
  
  DetectionSet DenseOneFeatureModel::detect_at_scale(const ImRGBZ& im, DetectionFilter filter, FeatPyr_Key  scale) const
  {
    // debug, show response image for this level
    //imagesc("Resp Image",resp); cvWaitKey(0);
    int blocks_x, blocks_y;
    shared_ptr<FeatIm > im_feats(new FeatIm());
    Mat resp;
    Size feat_sz = resp_impl(resp,*im_feats,im);
    
    // sort the detections by score.
    vector<Intern_Det> idets;
    for(int rIter = 0; rIter < resp.rows; rIter++)
      for(int cIter = 0; cIter < resp.cols; cIter++)
      {
	handle_idet(rIter,cIter,resp.at<float>(rIter,cIter),qnan,filter,idets);
      }
    // sort from high => low
    std::sort(idets.begin(),idets.end());
    std::reverse(idets.begin(),idets.end());
    
    // build the detections result.
    return externDets(idets,im.filename,im_feats, feat_sz, filter, scale.scale);
  }

  SparseVector DenseOneFeatureModel::extractPos(MetaData& metadata, AnnotationBoundingBox bb) const
  {
    shared_ptr<const ImRGBZ> im = metadata.load_im();
    return OneFeatureModel::extractPos_w_border(*im,bb);    
  }
  
  void DenseOneFeatureModel::prime_learner()
  {
    // DEBUG
    //Size hog1Win = hog1->getWinSize();
    //printf("OneFeatureModel::OneFeatureModel sCell = %d\n",s_cell);
    //printf("OneFeatureModel::OneFeatureModel.hog1.winSize = (%d,%d)\n",
	   //hog1Win.height,hog1Win.width);
    learner->prime(hog1->getDescriptorSize() + GlobalFeature::length);
  }  
  
  DenseOneFeatureModel::DenseOneFeatureModel(Size gtSize, Size ISize, double C, double area, shared_ptr< IHOGComputer_Factory > comp_factory, shared_ptr< LDA > learner): 
    OneFeatureModel(gtSize, ISize, C, area, comp_factory, learner)
  {
    prime_learner();
  }

  DenseOneFeatureModel::DenseOneFeatureModel(shared_ptr< IHOGComputer_Factory > comp_factory): 
    OneFeatureModel(comp_factory)
  {
    prime_learner();
  }
  
  /// SECTION: Sparse Detection operations
  /// 1  : occluder 
  /// 0  : object
  /// -1 : background
  SparseOneFeatureModel::SparseOneFeatureModel(Size gtSize, Size ISize, double C, double area, shared_ptr<IHOGComputer_Factory> comp_factory, shared_ptr<LDA> learner): 
    OneFeatureModel(gtSize, ISize, C, area, 
		    comp_factory,
		    learner)
  {
  }

  SparseOneFeatureModel::SparseOneFeatureModel(shared_ptr<IHOGComputer_Factory> comp_factory): 
    OneFeatureModel(
      comp_factory)
  {
  }
  
  void SparseOneFeatureModel::check_feats_vs_resps(vector<Intern_Det>&idets,DetectionSet&dets) const
  {
    vector<float> wf = learner->getWf();
    float B = learner->getB();
    
    for(int iter = 0; iter < dets.size(); iter++)
    {
      Detection&det = *dets[iter];
      Intern_Det&idet = idets[iter];
      SparseVector x = det.feature();
      float feat_resp = x * vec_f2d(wf) + B;
      if(::abs(det.resp-feat_resp)>1e-4)
      {
	log_file << printfpp("SparseOneFeatureModel: resp comp error! %f %f",
			     det.resp,feat_resp) << endl;
			     
	for(int xIter = 0; xIter < hog1->blocks_x(); xIter++)
	  for(int yIter = 0; yIter < hog1->blocks_y(); yIter++)
	    for(int cell = 0; cell < hog1->getNBins(); cell++)
	    {
	      int idx = hog1->getIndex(xIter, yIter, 0 ,cell);
	      float wValue = wf[idx];
	      float featValue = x[idx];
	      string message = printfpp("wValue = %f  \tfeat_value = %f",wValue,featValue);
	      log_file << message << endl;
	      cout << message << endl;
	    }
	assert(false);
      }
    }    
  }
  
  float SparseOneFeatureModel::doDot(
    int y0, int x0, float z_manifold, int blocks_x, int blocks_y, int nbins, 
    const ImRGBZ& im, vector< float >& wf, shared_ptr< FeatIm > im_feats, 
    DepthFeatComputer& hog_for_scale) const
  {
    float resp = 0;
    int prod_length = 0; 
    int wIdx = 0; // order of iterations important for this to work
    for(int xIter = 0; xIter < blocks_x; xIter++)
      for(int yIter = 0; yIter < blocks_y; yIter++)
	for(int bin = 0; bin < nbins; bin++)
	{
	  // get the w component
	  float wValue = wf[wIdx]; wIdx++;
	  
	  // get the x component
	  float xValue = (*im_feats)[hog_for_scale.getIndex(x0+xIter,y0+yIter,0,bin)];
	  
	  // dot product
	  resp += wValue*xValue;
	  prod_length++;
	}
    assert(learner->getFeatureLength() == wf.size());
    if(learner->getFeatureLength() != prod_length)
    {
      cout << printfpp("descSize(%d) != prodLen(%d)",learner->getFeatureLength(),prod_length) << endl;
      cout << printfpp("hog1: %d %d %d",
	hog1->blocks_x(),hog1->blocks_y(),hog1->getNBins()) << endl;
      assert(false);
    }    
    
    return resp;
  }
    
  DetectionSet SparseOneFeatureModel::detect_at_scale(const ImRGBZ& im, DetectionFilter filter, FeatPyr_Key  scale) const
  {
    auto dotFn = [&](
      int y0, int x0,double manifold_z,int blocks_x, int blocks_y, int nbins,const ImRGBZ&im,
      vector<float>&wf,shared_ptr<FeatIm> im_feats,DepthFeatComputer&hog_for_scale)
      {
	return doDot(y0, x0,manifold_z,blocks_x, blocks_y, nbins,im,wf,im_feats,hog_for_scale);
      };
      
    auto externFn = [&](vector< Intern_Det >& interns, string filename, 
			shared_ptr< FeatIm > im_feats, Size im_blocks, DetectionFilter filter)
    {
      return externDets(interns,im.filename,im_feats, im_blocks, filter,scale.scale);
    };
              
    auto manifoldFn_default_lambda = [&](int x0, int y0)
    {
      // get the BB
      Rect_<double> bb = externBB(y0,x0);
      
      return manifoldFn_default(im,bb);
    };
    
    return detect_at_scale(im,filter,scale,dotFn,externFn,manifoldFn_default_lambda);
  }
  
  void SparseOneFeatureModel::check_window(
    int y0, int x0,double z_manifold,FeatPyr_Key scale,
    const ImRGBZ&im,DetectionFilter&filter,
    int blocks_x, int blocks_y, int nbins,
    vector<float>&wf,float B,float min_world_area,float max_world_area,
    vector<Intern_Det>&idets,shared_ptr<FeatIm> im_feats,
    DepthFeatComputer&hog_for_scale,DotFn dot) const
  {
    // prune out of bounds BBs
    Rect_<double> bb = externBB(y0, x0);
    Rect_<double> global_bb = bb; unPyrBB(global_bb,scale);
    const ImRGBZ & src_image = filter.feat_pyr->getSrcImage();
    if(global_bb.tl().x < 0 || global_bb.tl().y < 0 || 
       global_bb.br().x >= src_image.cols() || global_bb.br().y >= src_image.cols())
      return;
    
    // first compute the BB and check the scale to avoid extra work
    float bb_world_area = im.camera.worldAreaForImageArea(z_manifold,bb);
    if(
       z_manifold <= params::MIN_Z() || z_manifold >= params::MAX_Z() || 
       bb_world_area < min_world_area || bb_world_area > max_world_area)
      return;
    
    // now check if the center pixel is background.
    Point2d center_at_scale = rectCenter(bb);
    float center_z = im.Z.at<float>(center_at_scale.y,center_at_scale.x);
    if(false && filter.is_root_template && 
       (center_z >= z_manifold + getObjDepth() || 
        center_z <= params::MIN_Z() || center_z >= params::MAX_Z())) 
      return;
    
    // second, reject if the skin likelihood is to low
    if(params::use_skin_filter())
    {
      float skin_ratio = filter.feat_pyr->getSrcImage().skin_ratio(global_bb);
      if(filter.testing_mode && skin_ratio < .05)
      {
	//cout << "SparseOneFeatureModel::check_window: reject root for skin " << skin_ratio << endl;
	return;
      }
    }
    
    // prune with subtrees
    //if(filter.is_root_template)
    //{
      //double bc_overlap = subtree_box_consistancy(src_image,global_bb);
      //if(bc_overlap < .25)
	//return;
    //}
    
    // compute the dot product
    float resp = dot(y0, x0, z_manifold,blocks_x, blocks_y, nbins,im,wf,im_feats,hog_for_scale) + B;
    if(resp == -inf)
      return;
    
    // create the detection object
    handle_idet(y0,x0,resp,z_manifold,filter,idets);
  }
  
  Mat SparseOneFeatureModel::resps(const ImRGBZ& im) const
  {
      return Mat();
  }
    
  shared_ptr<FeatIm> SparseOneFeatureModel::getFeat
    (DetectionFilter filter,FeatPyr_Key scale) const
  {
    return filter.feat_pyr->getFeat(scale);
  }
    
  DetectionSet SparseOneFeatureModel::detect_at_scale(
    const ImRGBZ& im, DetectionFilter filter,FeatPyr_Key scale,DotFn dotFn,ExternFn externFn,ManifoldFn manifoldFn) const
  {
    // get valid depth ranges for BB of this size
    float min_world_area, max_world_area;
    minMaxWorldAreas(im,externBB(0,0),min_world_area, max_world_area,filter);
    
    // compute feature
    DepthFeatComputer*hog_for_scale = 
      &filter.feat_pyr->computeFeat(im,
				    [&](Size sz,int sbins)
				    {
				      auto fact_ptr = comp_factory->build(sz,sbins);
				      return fact_ptr;
				    },
				    *hog,scale);
    shared_ptr<FeatIm> im_feats = getFeat(filter,scale);
    Size feat_sz(hog_for_scale->blocks_x(),hog_for_scale->blocks_y());
    
    // get the template
    vector<float> wf = learner->getWf();
    float B = learner->getB();
        
    // run the detector
    int blocks_x1 = hog1->blocks_x(), blocks_y1 = hog1->blocks_y(), nbins1 = hog1->getNBins();
    int blocks_x_scale = hog_for_scale->blocks_x(), blocks_y_scale = hog_for_scale->blocks_y();
    vector<Intern_Det> idets;
    for(int x0 = 0; x0 < blocks_x_scale - blocks_x1 + 1; x0++)
      for(int y0 = 0; y0 < blocks_y_scale - blocks_y1 + 1; y0++)
      {
	vector<float> test_depths = manifoldFn(x0,y0);
	for(float z : test_depths)
	  check_window(y0, x0,z,scale,im,filter,blocks_x1, blocks_y1, nbins1,
	      wf,B,min_world_area,max_world_area,idets,im_feats,*hog_for_scale,dotFn);
      }
      
    DetectionSet dets = externFn(idets,im.filename,im_feats, feat_sz, filter);
    for(auto && det : dets)
      assert(det != nullptr);
    //check_feats_vs_resps(idets,dets);
    return dets;
  }

  SparseVector SparseOneFeatureModel::extractPos(MetaData& metadata, AnnotationBoundingBox bb) const
  {
    // ideally, we'll pad occluded cells with a "1"
    shared_ptr<const ImRGBZ> im = metadata.load_im();
    SparseVector pos = OneFeatureModel::extractPos_w_border(*im,bb,nullptr,
			[&](DepthFeatComputer* comp)
			{
			    ;
			});
    
    return pos;
  }
  
  Mat SparseOneFeatureModel::show(const string& title)
  {    
    return hog1->show(title,learner->getW()); waitKey_safe(1);
  }
  
  void SparseOneFeatureModel::prime_learner()
  {
    learner->prime(hog1->getDescriptorSize());
  }
  
  void SparseOneFeatureModel::minMaxWorldAreas(
    const ImRGBZ&im,Rect_<double> bb0,float& min_world_area, float& max_world_area,
    DetectionFilter filter) const
  {
    areaModel.validRange(world_area_variance,filter.testing_mode,min_world_area,max_world_area);
    
    // debug
    //static mutex m; unique_lock<mutex> l(m);
    //cout << "min_area: " << min_area << " max_area: " << max_area << endl;
    //cout << "f: " << f << endl;
    //cout << "min_depth: " << min_depth << " max_depth: " << max_depth << endl;
  }

  float SparseOneFeatureModel::getObjDepth() const
  {
    return fromString<double>(g_params.require("OBJ_DEPTH"));
  }
  
  ///
  /// Section: Feature extraction by detection
  ///  
  float FeatureExtractionModel::doDot(int y0, int x0, float z_manifold, int blocks_x, int blocks_y, int nbins, const ImRGBZ& im, vector< float >& wf, shared_ptr< FeatIm > im_feats, DepthFeatComputer& hog_for_scale) const
  {
      return inf;
  }

  FeatureExtractionModel::FeatureExtractionModel(shared_ptr< IHOGComputer_Factory > comp_factory): 
    SparseOneFeatureModel(comp_factory)
  {
  }

  FeatureExtractionModel::FeatureExtractionModel(Size gtSize, 
						 Size ISize, 
						 double C, 
						 double area, 
						 shared_ptr< IHOGComputer_Factory > comp_factory, 
						 shared_ptr< LDA > learner): 
    SparseOneFeatureModel(gtSize, ISize, C, area, comp_factory, learner)
  {
    prime_learner();
  }
  
  shared_ptr< FeatIm > FeatureExtractionModel::getFeat(DetectionFilter filter, FeatPyr_Key scale) const
  {
      return nullptr;
  }
}
