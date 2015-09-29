/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "RandomHoughFeature.hpp"
#include "Detector.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include "Detection.hpp"
#include <boost/math/special_functions/round.hpp>
#include <boost/graph/graph_concepts.hpp>
#include "Skin.hpp"
#include "Orthography.hpp"
#include "Quaternion.hpp"
#include "ScanningWindow.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;
 
  using params::RESP_ORTHO_X_RES;
  using params::RESP_ORTHO_Y_RES;  
  using params::RESP_ORTHO_Z_RES;
  static Size ortho_res(RESP_ORTHO_X_RES,RESP_ORTHO_Y_RES);  
  
  ///
  /// SECTION StructuredWindow
  ///  
  
  Point2d StructuredWindow::queryToImCoords(const Point2d& query) const
  {
    int x = boost::math::round(query.x * detection->BB.width) + detection->BB.tl().x;
    int y = boost::math::round(query.y * detection->BB.height) + detection->BB.tl().y;
    return Point2d(x,y);
  }
  
  double StructuredWindow::metric_feature(const Point2d& query) const
  {
    assert(im != nullptr);
    Point2d xy = queryToImCoords(query);
    double ignore, depth = im->Z.at<float>(xy.y,xy.x);
    
    CustomCamera box_camera = im->camera.crop(detection->BB);
    box_camera.setRes(im->cols(),im->rows());
    double did = HOGComputer_Area::DistInvarDepth(depth,box_camera,ignore);;
    //cout << printfpp("z(%f) => did(%f)",depth,did) << endl;
    return did;
    //return im->distInvarientDepths().at<float>(y,x);
  }  
  
  double StructuredWindow::skin_likelihood(const Point2d& query) const
  {
    // get the query position in our local image
    Point2d xy = queryToImCoords(query);
    
    // get the color at that position
    assert(im->RGB.type() == DataType<Vec3b>::type);
    Vec3b color = im->RGB.at<Vec3b>(xy.y,xy.x);
    return deformable_depth::skin_likelihood(color);
  }
  
  Vec2d StructuredWindow::rgChroma(const Point2d& query) const
  {
    Vec3b RGB = im->RGB.at<Vec3b>(query.y,query.x);
    double R = RGB[0];
    double G = RGB[1];
    double B = RGB[2];
    double r = R / (R+G+B);
    double g = G / (R+G+B);
    double b = B / (R+G+B);
    return Vec2d(r,g);
  }
  
  static Rect queryRect(const Point2d&tl,const Point2d&br)
  {
    Rect q(tl,br);
    q.width = std::max(q.width,1);
    q.height= std::max(q.height,1);
    return q;
  }
  
  double StructuredWindow::mean_area(const Rect_< double >& query) const
  {
    // get the query position in our local image
    Point2d tl = queryToImCoords(query.tl());
    Point2d br = queryToImCoords(query.br());
    
    return im->mean_depth(queryRect(tl,br));
  }
  
  double StructuredWindow::relative_depth(const Rect_< double >& query) const
  {
    Point2d tl = queryToImCoords(query.tl());
    Point2d br = queryToImCoords(query.br());
    
    return im->mean_depth(queryRect(tl,br)) - detection->getDepth(*im);
  }
  
  double StructuredWindow::mean_intensity(const Rect_< double >& query) const
  {
    Point2d tl = queryToImCoords(query.tl());
    Point2d br = queryToImCoords(query.br());

    return im->mean_intensity(queryRect(tl,br));
  }
  
  Point3d StructuredWindow::ortho_detBBCenter() const
  {
    assert(im != nullptr);

    Rect detectBB = detection->BB;
    float det_z = im->Z.at<float>(rectCenter(detectBB));
    assert(goodNumber(det_z));
    Rect_<double> detBB_ortho = map2ortho(im->camera,ortho_res,detectBB,det_z);
    Point2d detBBCenter = rectCenter(detBB_ortho);
    assert(0 <= detBBCenter.x && 0 <= detBBCenter.y && 
	    detBBCenter.x < ortho_res.width && detBBCenter.y < ortho_res.height);
    return Point3d(detBBCenter.x,detBBCenter.y,det_z);
  }  
  
  Rect_< double > StructuredWindow::ortho_detBB() const
  {
    return map2ortho(im->camera,ortho_res,detection->BB, detection->getDepth(*im));
  }
  
  ///
  /// SECTION StructuredExample 
  ///

  Point3d StructuredExample::ortho_handBBCenter() const
  {
    // load im
    const MetaData& c_metadata = *metadata;
    shared_ptr<const ImRGBZ> im = c_metadata.load_im();
    
    // get the center for the GT
    Rect handBB = metadata->get_positives_c().at("HandBB");
    handBB = clamp(im->Z,handBB);
    float hand_z = im->Z.at<float>(rectCenter(handBB));
    Rect_<double> handBB_ortho = map2ortho(im->camera,ortho_res,handBB, hand_z);
    Point2d handBBCenter = rectCenter(handBB_ortho);    
    handBBCenter.x = clamp<int>(0,handBBCenter.x,ortho_res.width-1);
    handBBCenter.y = clamp<int>(0,handBBCenter.y,ortho_res.height-1);
    assert(0 <= handBBCenter.x && 0 <= handBBCenter.y && 
	    handBBCenter.x < ortho_res.width && handBBCenter.y < ortho_res.height);
    return Point3d(handBBCenter.x,handBBCenter.y,hand_z);
  }
  
  double compCorrectness(const map<string,AnnotationBoundingBox>&selected,const Rect_<double>&bb)
  {
    double correctness = 0;
    for(auto & gt_bb : selected)
      correctness = std::max<double>(correctness,rectIntersect(gt_bb.second,bb));
    return correctness; //;    
  }
  
  double StructuredExample::correctness() const
  {
    auto&selected = metadata->default_training_positives();
    return compCorrectness(selected,detection->BB);
  }
  
  bool StructuredExample::is_correct() const
  {
    return correctness() > .5;
  }
  
  bool StructuredExample::is_black() const
  {
    auto selected = metadata->get_positives_c();
    return compCorrectness(selected,detection->BB);
  }

  bool StructuredExample::is_white() const
  {
    return correctness() > .75;
  }
  
  ///
  /// SECTION RandomFeature
  /// 
  RandomFeature::RandomFeature()
  {
  }
  
  RandomFeature::RandomFeature(const vector<StructuredExample>&examples) : 
  uuid(deformable_depth::uuid())
  {    
    // randomly select a feature type
    vector<double> feat_type_dist{0,0,.1,.3,.2,.2,.1,.1};
    switch(rnd_multinom(feat_type_dist))
    {
      case 0:
	feature_type = DistanceInvarientDepth;
	break;
      case 1:
	feature_type = SkinLikelihood;
	break;
      case 2:
	feature_type = MeanDepth;
	break;
      case 3:
	feature_type = MeanIntensity;
	break;
      case 4:
	feature_type = R_Chroma;
	break;
      case 5:
	feature_type = G_Chroma;
	break;
      case 6:
	feature_type = Resolution;
	break;
      case 7:
	feature_type = RelativeDepth;
	break;
      default:
	assert(false);
    }
    
    p.x = sample_in_range(0,1);
    p.y = sample_in_range(0,1);
    roi1 = Rect_<double>(Point2d(sample_in_range(0,1),sample_in_range(0,1)),
			 Point2d(sample_in_range(0,1),sample_in_range(0,1)));
    roi2 = Rect_<double>(Point2d(sample_in_range(0,1),sample_in_range(0,1)),
			 Point2d(sample_in_range(0,1),sample_in_range(0,1)));
    const StructuredExample&random_example = examples[rand()%examples.size()];
    this->threshold = feature(random_example);
    
    log_file << printfpp("RandomFeature::RandomFeature %f %f %f %d",
			 (double)p.x,(double)p.y,
			 (double)this->threshold,(int)feature_type) << endl;    
  }
  
  double RandomFeature::feature(const StructuredWindow& ex) const
  {
    switch(feature_type)
    {
      case DistanceInvarientDepth: // size
	return ex.metric_feature(p);
      case SkinLikelihood: // color
	return ex.skin_likelihood(p);
      case MeanDepth: // shape
	return ex.mean_area(roi2) - ex.mean_area(roi1);
      case MeanIntensity:
	return ex.mean_intensity(roi2) - ex.mean_intensity(roi1);
      case R_Chroma:
	return ex.rgChroma(p)[0];
      case G_Chroma:
	return ex.rgChroma(p)[1];
      case Resolution:
	return ex.detection->BB.area();
      case RelativeDepth:
	return ex.relative_depth(roi1);
      default:
	throw std::logic_error("Bad Feature_type");
    }
  }
  
  bool RandomFeature::predict(const StructuredWindow& ex) const
  {
    return feature(ex) < this->threshold;
  }    
  
  double RandomFeature::structural_gain
  (
    double correct_true_ratio, double correct_false_ratio,
    vector<StructuredExample>&correct_exs_true, vector<StructuredExample>&correct_exs_false,
    const PCAPose&pcaPose, ostringstream&oss,
    Mat&pose_mu_true, Mat&pose_cov_true,
    Mat&pose_mu_false, Mat&pose_cov_false
  )
  {
    double latent_structural_gain = 
	- correct_true_ratio * entropy_gaussian(
	      [&](int i)
	      {
		Mat pcFeat = pcaPose.project_q(correct_exs_true[i].metadata);
		oss << "p" << pcFeat;
		return pcFeat;
	      },
	      correct_exs_true.size(),pose_mu_true,pose_cov_true) 
	- correct_false_ratio * entropy_gaussian(
	      [&](int i)
	      {
		Mat pcFeat = pcaPose.project_q(correct_exs_false[i].metadata);
		oss << "f" << pcFeat;
		return pcFeat;
	      },
	      correct_exs_false.size(),pose_mu_false,pose_cov_false);    

    return latent_structural_gain;
  }  
  
  void write(cv::FileStorage&fs, std::string&, const deformable_depth::RandomFeature&rf)
  {
    fs << "{";
    fs << "feature_type" << rf.feature_type;
    fs << "p" << rf.p;
    fs << "roi1" << rf.roi1;
    fs << "roi2" << rf.roi2;
    fs << "threshold" << rf.threshold;
    fs << "uuid" << rf.uuid;
    fs << "}";
  }
  
  ///
  /// SECTION: RandomHoughFeature
  /// 
  RandomHoughFeature::RandomHoughFeature(vector<StructuredExample>&examples) :
    RandomFeature(examples)
  {
    times_voted_true = 0;
    times_voted_false = 0;    
  }
  
  string RandomHoughFeature::get_uuid() const
  {
    return uuid;
  }
  
  double RandomHoughFeature::get_N_false_neg() const
  {
    return (1-correct_false)*get_N_false();
  }

  double RandomHoughFeature::get_N_false_pos() const
  {
    return correct_false*get_N_false();
  }

  double RandomHoughFeature::get_N_true_neg() const
  {
    return (1-correct_true)*get_N_true();
  }

  double RandomHoughFeature::get_N_true_pos() const
  {
    return correct_true*get_N_true();
  }
  
  int RandomHoughFeature::get_N_false() const
  {
    return N_false;
  }

  int RandomHoughFeature::get_N_true() const
  {
    return N_true;
  }
    
  void RandomHoughFeature::print_voting_history() const
  {
    log_file << printfpp("RandomFeature: votes %d (true) %d (false)",
			 (int)times_voted_true,(int)times_voted_false) << endl;
  }
    
  VoteResult RandomHoughFeature::vote(
    HoughOutputSpace& output,
    const StructuredWindow&swin,
    const PCAPose&pose) const
  {
    Point3d center = swin.ortho_detBBCenter();
    Point3d kernel_center = center;
    
    double correct_ratio;
    Mat voting_kernel;
    bool prediction = predict(swin);
    if(prediction)
    {
      // vote for true
      times_voted_true++;
      correct_ratio = correct_true;
      voting_kernel = vote_location.voting_kernel_true;
    }
    else
    {
      // vote for false
      times_voted_false++;
      correct_ratio = correct_false;
      voting_kernel = vote_location.voting_kernel_false;
    }
    kernel_center.x += vote_location.mu(prediction).at<double>(0);
    kernel_center.y += vote_location.mu(prediction).at<double>(1);    
    
    // Call Spaces's vote function
    double conf = output.vote(prediction,correct_ratio,
		       kernel_center,center,voting_kernel,pose,vote_pose);
    
    VoteResult result;
    result.leaf = this;
    result.conf = conf;
    return result;
  }
  
  Mat offset_from_example(const vector<StructuredExample>&exs,int iter)
  {
    // grab current example
    const StructuredExample & ex = exs[iter];
    
    // get the offsets
    Point3d handBBCenter = ex.ortho_handBBCenter();
    Point3d detBBCenter = ex.ortho_detBBCenter();
    int off_x = handBBCenter.x - detBBCenter.x;
    int off_y = handBBCenter.y - detBBCenter.y;      
    //int off_z = (handBBCenter.z - detBBCenter.z)/10;
    
    // write the factored parameters
    Mat sample(1,2,DataType<double>::type,Scalar::all(0));
    sample.at<double>(0,0) = off_x;
    sample.at<double>(0,1) = off_y;  
    require_equal<int>(sample.rows,1);
    require_equal<int>(sample.cols,2);
    return sample;
  }
      
  void RandomFeature::split_props(
    const vector< StructuredExample >& examples, 
    double&true_pos, double&false_pos, double&true_neg, double&false_neg)
  {
    vector<StructuredExample> exs_true,exs_false,correct_exs_true,correct_exs_false;
    split_examples(examples, exs_true, exs_false, 
		   true_pos, false_pos, true_neg, false_neg,
		   correct_exs_true, correct_exs_false);
  }
  
  void RandomFeature::split_examples(
    const vector< StructuredExample >& examples, 
    vector<StructuredExample> &exs_true, 
    vector<StructuredExample> &exs_false,
    double&true_pos, 
    double&false_pos, 
    double&true_neg, 
    double&false_neg,
    vector<StructuredExample> &correct_exs_true, 
    vector<StructuredExample> &correct_exs_false
  )
  {
    // clear
    exs_true.clear();
    exs_false.clear();
    correct_exs_true.clear();
    correct_exs_false.clear();
    
    // compute
    atomic<long> count(0);
    for(const StructuredExample&ex : examples)
    {
      if(predict(ex))
      {
	exs_true.push_back(ex);
	if(ex.is_white() > .75)
	{
	  correct_exs_true.push_back(ex);
	  true_pos++;
	}
	else if(ex.is_black() <= 0)
	  true_neg++;
      }
      else
      {
	exs_false.push_back(ex);
	if(ex.is_white() > .75)
	{
	  correct_exs_false.push_back(ex);
	  false_pos++;
	}
	else if(ex.is_black() <= 0)
	  false_neg++;
      }
      
      long cur_count = count++;
      int stride = (examples.size()/10);
      if(stride > 0 && cur_count % stride == 0)
	log_file << printfpp("info_gain partition %d of %d",
			      (int)cur_count,(int)examples.size()) << endl;
    }    
  }
  
  InformationGain RandomHoughFeature::info_gain(
    vector< StructuredExample >& examples, 
    vector<StructuredExample> &exs_true, 
    vector<StructuredExample> &exs_false,
    const PCAPose&pcaPose)
  {
    h0 = qnan; // not needed in practice entropy_gaussian(examples,totalMu,totalSigma,p_correct);

    double true_pos = 0, false_pos = 0, true_neg = 0, false_neg = 0; 
    vector<StructuredExample> correct_exs_true;
    vector<StructuredExample> correct_exs_false;
    split_examples(examples, 
		   exs_true, exs_false,
		   true_pos, false_pos, 
		   true_neg, false_neg,
		   correct_exs_true,correct_exs_false);
    log_file << printfpp("Split %d/%d",(int)exs_true.size(),(int)exs_false.size()) << endl;
    if(exs_true.size() == 0 || exs_false.size() == 0)
    {
      InfoGain.shannon_info_gain = -inf;
      InfoGain.differential_info_gain = -inf;
      InfoGain.latent_structural_gain = -inf;
      return InfoGain;
    }
    
    // compute the branch probabilities
    double true_ratio = exs_true.size() / (double)examples.size();
    double false_ratio = exs_false.size() / (double)examples.size();
    double n_correct = correct_exs_true.size() + correct_exs_false.size();
    double correct_true_ratio = correct_exs_true.size() / n_correct;
    double correct_false_ratio = correct_exs_false.size() / n_correct;
    N_true = exs_true.size();
    N_false = exs_false.size();
    correct_true = true_pos/N_true;
    correct_false = false_pos/N_false;
    // compute the location differential_info_gain
    InfoGain.differential_info_gain = 
	- true_ratio * entropy_gaussian(
	      [&](int i){
		return offset_from_example(exs_true,i);},
	      exs_true.size(),vote_location.mu_true,vote_location.cov_true) 
	- false_ratio* entropy_gaussian([&](int i){
		return offset_from_example(exs_false,i);},
	      exs_false.size(),vote_location.mu_false,vote_location.cov_false);
    // compute the shannon_info_gain
    InfoGain.shannon_info_gain = 	
	- true_ratio * shannon_entropy({true_pos/exs_true.size(),true_neg/exs_true.size()}) 
	- false_ratio* shannon_entropy({false_pos/exs_false.size(),false_neg/exs_false.size()});
    // compute the info_gain in our latent space
    ostringstream oss;
    InfoGain.latent_structural_gain = structural_gain(
      correct_true_ratio, correct_false_ratio,
      correct_exs_true, correct_exs_false,pcaPose, oss,
      vote_pose.mu_true,vote_pose.cov_true,
      vote_pose.mu_false,vote_pose.cov_false);
	
    // compute the voting kernels
    vote_location.update_votes(2.5);
    vote_pose.update_votes(5);
    vote_pose.message = oss.str();
    
    log_file << *this << endl;

    return InfoGain;
  }
  
  void RandomHoughFeature::log_kernels() const
  {
    //log_im("voting_kernel_true",imageeq("",voting_kernel_true,false,false));
    //log_im("voting_kernel_false",imageeq("",voting_kernel_false,false,false));
    
    // generate the test text
    Mat test_text = image_text(
      printfpp("feat = %d thresh = %f",(int)feature_type,(double)this->threshold));
    
    // generate # the text
    Mat text_true = image_text(
      printfpp("%f %f",(double)correct_true,(double)N_true));
    Mat text_false = image_text(
      printfpp("%f %f",(double)correct_false,(double)N_false)); 
    Mat text = vertCat(text_true,text_false);    
    
    Mat vis_sidebyside = vertCat(vote_location.log_kernels(),text);
    Mat vis_pose = vote_pose.log_kernels(); 
    vis_pose = rotate_colors(vis_pose,Quaternion(1,0,1,1).rotation_matrix());
    
    log_im("voting_kernels",vertCat(test_text,horizCat(vis_sidebyside,vis_pose)));
  }
  
  ostream& operator<< (ostream& os, const HoughVote&vote)
  {
    os << "mu_true: " << vote.mu_true << endl;
    os << "mu_false: " << vote.mu_false << endl;
    os << "cov_true: " << vote.cov_true << endl;
    os << "cov_false: " << vote.cov_false << endl;
    return os;
  }
  
  ostream& operator << (ostream& os, const RandomHoughFeature&rf)
  {
    static mutex m; lock_guard<mutex> l(m);
    os << printfpp("RandomFeature::info_gain H(examples) = %f",rf.h0) << endl;
    os << printfpp("RandomFeature::info_gain (%f,%f)",
		   rf.InfoGain.shannon_info_gain,
		   rf.InfoGain.differential_info_gain) << endl;
    os << "n_true: " << rf.N_true << endl;
    os << "n_false: " << rf.N_false << endl;
    os << "c0: " << rf.p_correct << endl;
    os << "correct_true: " << rf.correct_true << endl;
    os << "correct_false: " << rf.correct_false << endl;
    os << "vote_location: " << rf.vote_location << endl;
    os << "test pt: " << printfpp("(%f, %f)",(double)rf.p.x,(double)rf.p.y) << endl;
    os << "roi1: " << rf.roi1 << endl;
    os << "roi2: " << rf.roi2 << endl;
    os << "threshold: " << rf.threshold << endl;
    os << "feature_type: " << rf.feature_type << endl;
    
    return os;
  }
  
  RandomHoughFeature::RandomHoughFeature()
  {
  }
    
  InformationGain::InformationGain() : 
    differential_info_gain(qnan),
    shannon_info_gain(qnan),
    latent_structural_gain(qnan)
  {
  }
    
  void write(FileStorage& fs, string& , const InformationGain& ig)
  {
    fs << "{";
    fs << "differential_info_gain" << ig.differential_info_gain;
    fs << "shannon_info_gain" << ig.shannon_info_gain;
    fs << "latent_structural_gain" << ig.latent_structural_gain;
    fs << "}";
  }
  
  void read(const FileNode& fn, InformationGain& ig, InformationGain )
  {
    fn["differential_info_gain"] >> ig.differential_info_gain;
    fn["shannon_info_gain"] >> ig.shannon_info_gain;
    fn["latent_structural_gain"] >> ig.latent_structural_gain;
  }
    
  void write(cv::FileStorage&fs, std::string&, const deformable_depth::HoughVote&vote)
  {
    fs << "{";
    fs << "mu_true" << vote.mu_true;
    fs << "mu_false" << vote.mu_false;
    fs << "cov_true" << vote.cov_true;
    fs << "cov_false" << vote.cov_false;
    fs << "sigmaInv_false" << vote.sigmaInv_false;
    fs << "sigmaInv_true" << vote.sigmaInv_true;
    fs << "voting_kernel_true" << vote.voting_kernel_true;
    fs << "voting_kernel_false" << vote.voting_kernel_false;
    fs << "}";
  }
    
  void write(FileStorage& fs, string& , const unique_ptr< RandomHoughFeature >& rf)
  { 
    fs << "{";
    fs << "feature_type" << rf->feature_type;
    fs << "p" << rf->p;
    fs << "roi1" << rf->roi1;
    fs << "roi2" << rf->roi2;
    fs << "threshold" << rf->threshold;
    fs << "h0" << rf->h0;
    fs << "InfoGain" << rf->InfoGain;
    fs << "p_correct" << rf->p_correct;
    fs << "correct_true" << rf->correct_true;
    fs << "correct_false" << rf->correct_false;
    fs << "N_true" << rf->N_true;
    fs << "N_false" << rf->N_false;
    fs << "vote_location" << rf->vote_location;
    fs << "vote_pose" << rf->vote_pose;
    fs << "uuid" << rf->uuid;
    fs << "}";
  }
  
  void read(const cv::FileNode&fn, deformable_depth::HoughVote&vote, deformable_depth::HoughVote)
  {
    fn["mu_true"] >> vote.mu_true;
    fn["mu_false"] >> vote.mu_false;
    fn["cov_true"] >> vote.cov_true;
    fn["cov_false"] >> vote.cov_false;
    fn["sigmaInv_false"] >> vote.sigmaInv_false;
    fn["sigmaInv_true"] >> vote.sigmaInv_true;
    fn["voting_kernel_true"] >> vote.voting_kernel_true;
    fn["voting_kernel_false"] >> vote.voting_kernel_false;    
  }
  
  void read(const FileNode& fn, unique_ptr< RandomHoughFeature >& rf, unique_ptr< RandomHoughFeature > )
  {
    rf.reset(new RandomHoughFeature());
    int ft; fn["feature_type"] >> ft; rf->feature_type = RandomHoughFeature::FeatureTypes(ft);
    read(fn["p"],rf->p,Point2d());
    deformable_depth::read<double>(fn["roi1"],rf->roi1,Rect_<double>());
    deformable_depth::read<double>(fn["roi2"],rf->roi2,Rect_<double>());
    fn["threshold"] >> rf->threshold;
    fn["h0"] >> rf->h0;
    fn["InfoGain"] >> rf->InfoGain;
    fn["p_correct"] >> rf->p_correct;
    fn["correct_true"] >> rf->correct_true;
    fn["correct_false"] >> rf->correct_false;
    fn["N_true"] >> rf->N_true;
    fn["N_false"] >> rf->N_false;
    fn["vote_location"] >> rf->vote_location;
    fn["vote_pose"] >> rf->vote_pose;
    if(!fn["uuid"].empty())
      fn["uuid"] >> rf->uuid;
    else
      rf->uuid = deformable_depth::uuid();
  }
    
  Mat RandomHoughFeature::get_cov_false() const
  {
    return vote_location.cov_false;
  }

  Mat RandomHoughFeature::get_cov_true() const
  {
    return vote_location.cov_true;
  }

  Mat RandomHoughFeature::get_mu_false() const
  {
    return vote_location.mu_false;
  }

  Mat RandomHoughFeature::get_mu_true() const
  {
    return vote_location.mu_true;
  }    
  
  ///
  /// SECTION: Standalone functions
  ///
  vector<StructuredExample> extract_features(FeatureExtractionModel&feature_extractor,
					     vector< shared_ptr< MetaData > >& training_set)
  {
    return extract_features(training_set);
  }

  vector<StructuredExample> 
    extract_features(      
      vector< shared_ptr< MetaData > >& training_set)
  {
    log_file << "++extract_features" << endl;
    vector<StructuredExample> all_feats;
    TaskBlock extract_features("SRF_Model::Train extract_features");
    progressBars->set_progress("srf_extract_features",0,training_set.size());
    for(int iter = 0; iter < training_set.size(); ++iter)
    {      
      extract_features.add_callee([&,iter]()
      {
	auto && ex = training_set.at(iter);
	progressBars->set_progress("srf_extract_features",iter,training_set.size());
	shared_ptr<const ImRGBZ> im = ex->load_im();
	DetectionFilter filt(-inf);
	filt.supress_feature = true;
	//filt.feat_pyr = shared_ptr<IFeatPyr>(new FauxFeatPyr(*im));
	//DetectionSet feats = feature_extractor.detect(*im,filt);
	filt.manifoldFn = manifoldFn_apxMin;
	DetectionSet feats = enumerate_windows(*im,filt);
	feats = removeTruncations(*im,feats);
	log_file << printfpp("SRFModel::train extracted %d examples from %s",
	  (int)feats.size(),ex->get_filename().c_str()) << endl;
	  
	// append
	auto selected_gt = params::defaultSelectFn()(ex->get_positives());
	vector<StructuredExample> here_feats;
	for(auto && feat : feats)
	{
	  // allocated a structured example for each feature
	  StructuredExample s_ex;
	  s_ex.metadata = ex;
	  s_ex.detection = feat;
	  const MetaData&c_metadata = *ex;
	  s_ex.im = c_metadata.load_im();
	  {
	    here_feats.push_back(s_ex);	
	  }
	}
	
	// make sure we get all positives through the filter
	logMissingPositives(*ex,feats);
	      	
	// accumulate the windows...
	{
	  static mutex m; unique_lock<mutex> l(m);
	  all_feats.insert(all_feats.end(),here_feats.begin(),here_feats.end());
	}
	// track memory consumption
	if(iter % 25 == 0)
	  dump_heap_profile();
      });
    }
    extract_features.execute();
    log_file << "--extract_features" << endl;
    progressBars->set_progress("srf_extract_features",0,0);
    return all_feats;
  }
  
  vector<StructuredExample> extract_gt_features(
    FeatureExtractionModel&feature_extractor,vector<StructuredExample>&all_feats)
  {
    map<MetaData*,StructuredExample> closest_matches;
    
    cout << "SRFModel::extract_gt_features begin" << endl;
    for(auto && feat : all_feats)
    {
      Rect_<double> bb_gt   = feat.metadata->get_positives().at("HandBB");
      Rect_<double> bb_cur; 
      if(closest_matches[&*feat.metadata].detection != nullptr)
	bb_cur = closest_matches[&*feat.metadata].detection->BB;
      Rect_<double> bb_cand = feat.detection->BB;
      double old_ol = rectIntersect(bb_cur,bb_gt);
      double new_ol = rectIntersect(bb_cand,bb_gt);
      if(new_ol >= old_ol)
      {
	cout << printfpp("ol: %f => %f",old_ol,new_ol) << endl;
	closest_matches[&*feat.metadata] = feat;
      }
    }
    cout << "SRFModel::extract_gt_features selection complete, converting" << endl;
    
    // vectorize
    vector<StructuredExample> gt_feats;
    for(auto & pair : closest_matches)
    {
      Rect_<double>&bb = pair.second.detection->BB;
      Rect_<double> bb_gt   = pair.second.metadata->get_positives().at("HandBB");
      if(rectIntersect(bb,bb_gt) < .5)
	continue;
      bb = rectFromCenter(rectCenter(bb_gt),bb.size());
      
      //Mat im = pair.second.im->RGB.clone();
      //rectangle(im,bb.tl(),bb.br(),Scalar(255,0,0));
      //log_im("gt_bb",im);
      gt_feats.push_back(pair.second);
    }
    double accept_ratio = gt_feats.size()/static_cast<double>(closest_matches.size());
    log_file << printfpp("extract_gt_features, accepted %f percent",accept_ratio) << endl;
    return gt_feats;
  }  
  
  vector< StructuredWindow > extract_windows(FeatureExtractionModel& feature_extractor, 
					      shared_ptr< ImRGBZ >& sample_im)
  {
    // what were extracting from this frame
    vector<StructuredWindow> frame_wins;
    DetectionFilter filt(-inf);
    filt.supress_feature = true;
    DetectionSet windows = feature_extractor.detect(*sample_im,filt);
    for(auto && win : windows)
    {
      StructuredWindow swin;
      swin.detection = win;
      swin.im = sample_im;
    }
    
    return frame_wins;
  }
}
