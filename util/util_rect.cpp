/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "util_rect.hpp"
#include "util_real.hpp"
#include "util.hpp"
#include "Kahan.hpp"

#ifndef WIN32
#include "Detector.hpp"
#include "Log.hpp"
#include "HandSynth.hpp"
#include <boost/concept_check.hpp>
#endif

namespace deformable_depth
{
  bool rectContains(const Mat& m, Rect roi)
  {
    return 0 <= roi.x && 0 <= roi.width 
      && roi.x + roi.width <= m.cols 
      && 0 <= roi.y && 0 <= roi.height 
      && roi.y + roi.height <= m.rows;
  }
  
  Mat affine_transform(Rect_< double >& src, Rect_< double >& dst)
  {
    Point2f src_coords[] = 
      {src.tl(),src.br(),Point2f(src.tl().x,src.br().y)};
    Point2f dst_coords[] = 
      {dst.tl(),dst.br(),Point2f(dst.tl().x,dst.br().y)};
    Mat transform = getAffineTransform(src_coords,dst_coords);
    transform.convertTo(transform,DataType<float>::type);
    if(transform.type() != DataType<float>::type)
    {
      cout << transform << endl;
      assert(false);
    }
    
    return transform;
  }
  
  Mat affine_transform(Rect src, Rect dst)
  {
    Rect_<double> src_d = src;
    Rect_<double> dst_d = dst;
    return affine_transform(src_d,dst_d);
  }

  Mat affine_transform_rr(RotatedRect src, RotatedRect dst)
  {
    // get points from the src
    Point2f src_coords[4];
    src.points(src_coords);
    // get points from the src
    Point2f dst_coords[4];
    dst.points(dst_coords);
    // 
    Mat transform = getAffineTransform(src_coords,dst_coords);
    transform.convertTo(transform,DataType<float>::type);
    return transform;
  }
  
  float rectIntersect(const Rect&r1, const Rect&r2)
  {
    if(r1.area() == 0 || r2.area() == 0)
      return 0;
    
    // we want: overlap = intersection ./ (area1+area2-intersection);
    
    float a1 = r1.area();
    float a2 = r2.area();
    float in = std::abs((r1&r2).area());
    float un = a1+a2-in;
    if(un <= 0)
      return 0;
    float ol = in/un;
    require_in_range<double>(0,ol,1);
    assert(goodNumber(ol));
    return ol;
  }
  
  float rect_max_intersect(vector< Rect > r1s, vector< Rect > r2s)
  {
    float max = -inf;
    
    for(int iter = 0; iter < r1s.size(); iter++)
      for(int jter = 0; jter < r2s.size(); jter++)
      {
	float intersect = rectIntersect(r1s[iter],r2s[jter]);
	if(intersect > max)
	  max = intersect;
      }
	
    return max;
  }
  
  cv::Size_< double > sizeFromAreaAndAspect(double area, double aspect)
  {
    // area = width * height
    // width = area / height;
    // width = area / (sqrt(area/aspect))
    // width = area / (sqrt(area/aspect))
    // width = sqrt(area * aspect)
    // 
    // aspect = width/height
    // height * aspect = width
    // height = width/aspect
    // height = (area/height)/aspect
    // height = area/(height * aspect)
    // height^2 = area/aspect
    // height =  sqrt(area/aspect)
    
    // area = width * height = sqrt(area*aspect)*sqrt(area/aspect) = area
    
    double width = std::sqrt(area*aspect);
    double height = std::sqrt(area/aspect);
    return Size_<double>(width,height);
  }
  
  Point2d rectCenter(const Rect_<double>& r)
  {
    int cen_x = (r.tl().x + r.br().x)/2;
    int cen_y = (r.tl().y + r.br().y)/2;
    return Point(cen_x,cen_y);
  }
  
  Rect rectUnion(vector<Rect> rects)
  {
    if(rects.size() == 0)
    {
      printf("warning: rectUnion called with empty list\n"); 
      fflush(stdout);
      return cv::Rect();
    }
    
    Rect result = rects[0];
    
#ifdef DD_CXX11
    for(Rect r : rects)
      result = result | r;
#else
    for(int rectIter = 0; rectIter < rects.size(); rectIter++)
      result = result | rects[rectIter];
#endif
    
    return result;
  }
  
  Rect_<double> rectScale(Rect_<double> r, float factor)
  {
    return Rect_<double>(
      Point2d(factor*r.x,factor*r.y), Size_<double>(factor*r.width,factor*r.height)); 
  }
  
  Rect rectResize(Rect r, float xScale, float yScale)
  {
    Point center(r.x+r.width/2,r.y+r.height/2);
    Size size(r.width*xScale,r.height*yScale);
    return rectFromCenter(center,size);
  }
  
  Rect clamp(Rect container, Rect containee)
  {
    return container & containee;
  }

  Rect clamp(Mat container, Rect containee)
  {
    Size2i sz(container.cols,container.rows);
    return clamp(Rect(Point2i(0,0),sz),containee);
  }
  
  void write_corners(vector< Point2d >& out, const Rect_< double >& rect)
  {
    out.push_back(rect.tl());
    out.push_back(rect.br());
    out.push_back(Point2d(rect.br().x,rect.tl().y)); // tr
    out.push_back(Point2d(rect.tl().x,rect.br().y)); // bl
  }
  
  Rect_<double> rectFromCenter(Point2d center, Size_<double> size)
  {
    Point2d tl(center.x-size.width/2.0,
	       center.y-size.height/2.0);
    return Rect_<double>(tl,size);
  }
  
  bool rectCorrect(cv::Rect gt, cv::Rect det)
  {
    Scores score;
    double resp;
    return rectScore(gt,det,resp,score);
  }
  
  void Scores::score(Scores::Type type, double resp)
  {
    if(type == Type::FN)
    {
      // false negative
      add_detection(FrameDet(true,-inf,qnan));
    }
    else if(type == Type::FP)
    {
      // false positive
      add_detection(FrameDet(false,resp,qnan));
      add_detection(FrameDet(true,-inf,qnan));
    }
    else if(type == Type::TP)
    {
      // correct
      add_detection(FrameDet(true,resp,qnan));
    }
    else
      assert(false);
  }
  
  bool rectScore(cv::Rect gt, cv::Rect det, double resp, Scores&score)
  {
    //log_file << "rectScore, resp = " << resp << endl;
    double thresh = .25;
    
    assert(gt != Rect());
    if(det == Rect())
    {
      // false negative
      score.score(Scores::Type::FN,resp);
      return false;
    }
    else if(rectIntersect(gt,det) <= thresh)
    {
      // false positive
      score.score(Scores::Type::FP,resp);
      return false;
    }
    else
    {
      // correct
      score.score(Scores::Type::TP,resp);
      return true;
    }
  }
  
  Point3d point_affine(Point3d point, const Mat& affine_transform)
  {
    Point2d pt2 = point_affine(Point2d(point.x,point.y),affine_transform);
    return Point3d(pt2.x,pt2.y,point.z);
  }

  Point_<double> point_affine(Point_<double> point, const Mat& affine_transform)
  {
    Mat at; affine_transform.convertTo(at,DataType<float>::type);
    assert(at.type() == DataType<float>::type);
    double m11 = at.at<float>(0,0);
    double m12 = at.at<float>(0,1);
    double m13 = at.at<float>(0,2);
    double m21 = at.at<float>(1,0);
    double m22 = at.at<float>(1,1);
    double m23 = at.at<float>(1,2);
    return Point_<double>(
      m11*point.x+m12*point.y+m13,
      m21*point.x+m22*point.y+m23);
  }

  Vec3d vec_affine(Vec3d v, const Mat&affine)
  {
    Point2d pt = point_affine(Point2d(v[0],v[1]),affine);
    return Vec3d(pt.x,pt.y,v[2]);
  }
  
#ifndef WIN32  
  Detection&detection_affine(Detection& detection, Mat& affine_transform)
  {
    detection.applyAffineTransform(affine_transform);
    return detection;
  }
#endif
  
  // return the bounding rectangle...
  cv::Rect_< double > rect_Affine(cv::Rect_< double > rect, const Mat& affine_transform)
  {
    // decompse the transform
    assert(affine_transform.type() == DataType<float>::type);
    float A11 = affine_transform.at<float>(0,0);
    float A12 = affine_transform.at<float>(0,1);
    float A21 = affine_transform.at<float>(1,0);
    float A22 = affine_transform.at<float>(1,1);
    float S = std::sqrt(A11*A11 + A21*A21); // scale
    float theta = rad2deg(std::atan(A21/A11)); // rotate
    float M = (A22 - A11)/A21;// sheere

    // find the new center
    Point2f tl = point_affine(rect.tl(),affine_transform);
    Point2f br = point_affine(rect.br(),affine_transform);
    Point2f cn = rectCenter(Rect(tl,br));

    // find the scale factors
    Mat rot_transform = affine_transform.clone();
    rot_transform.at<float>(0,2) = 0;
    rot_transform.at<float>(1,2) = 0;
    Vec3d xTr = vec_affine(Vec3d(1,0,0),rot_transform);
    Vec3d yTr = vec_affine(Vec3d(0,1,0),rot_transform);
    double xScale = std::sqrt(xTr.ddot(xTr));
    double yScale = std::sqrt(yTr.ddot(yTr));
   
    // define the rotated rectangle.
    RotatedRect rr(cn,Size(rect.width*xScale,rect.height*yScale),theta);
    return rr.boundingRect();
  }
  
  ///
  /// SECTION: Scores
  ///
  double Scores::tp(double threshold) const
  {
    double tp = 0;
    
    // true: correct
    // positive: resp > t
	for(int iter = 0; iter < detections.size(); ++iter)
    {
		const FrameDet & det = detections[iter];
      if(det.correct() && det.score() > threshold)
	tp++;
    }
    
    return tp;
  }
  
  double Scores::fn(double threshold) const
  {
    double fn = 0;
    
    // missing result
    // correct but, 
    // negative: resp < t
    for(int iter = 0; iter < detections.size(); ++iter)
    {
		const FrameDet & det = detections[iter];
      if(det.correct() && det.score() <= threshold)
	fn++;
    }
    
    return fn;
  }
  
  double Scores::fp(double threshold) const
  {
    double fp = 0;
    
    // false: not correct
    // positive: resp >= t
    for(int iter = 0; iter < detections.size(); ++iter)
    {
		const FrameDet & det = detections[iter];
      if(!det.correct() && det.score() > threshold)
	fp++;
    }
    
    return fp;
  }
  
  double Scores::v(double threshold) const
  {
    return qnan;
  }
  
  double Scores::p(double t) const
  {
    return tp(t)/(tp(t)+fp(t));
  }

  double Scores::r(double t) const
  {
    return tp(t)/(tp(t)+fn(t));
  }
  
  void Scores::add_detection(FrameDet det)
  {
    unique_lock<mutex> l(monitor);
	  
    detections.push_back(det);
  }
  
  Scores::Scores(const Scores& copy)
  {
    unique_lock<mutex> l1(monitor);
    unique_lock<mutex> l2(copy.monitor);    
    pose_correct = (copy.pose_correct);
    pose_incorrect = (copy.pose_incorrect);
    detections = (copy.detections);
  }
  
  Scores& Scores::operator=(const Scores& copy)
  {
    unique_lock<mutex> l1(monitor);
    unique_lock<mutex> l2(copy.monitor);
    pose_correct = (copy.pose_correct);
    pose_incorrect = (copy.pose_incorrect);
    detections = (copy.detections);
    
    return *this;
  }
  
  vector< FrameDet > Scores::getDetections() const
  {
    return detections;
  }
  
  string Scores::toString(double t) const
  {
    return printfpp("scores: p = %f r = %f f1 = %f (tp = %f fp = %f fn = %f)",
      p(t),r(t),f1(t), tp(t), fp(t), fn(t)
    );
  }
  
  Scores::Scores(vector< Scores > combine)
  {
    for(int iter = 0; iter < combine.size(); iter++)
    {
      Scores&other = combine[iter];	
      pose_correct += other.pose_correct;
      pose_incorrect += other.pose_incorrect;
      detections.insert(detections.end(),other.detections.begin(),other.detections.end());
    }
  }
  
  bool pr_comp(double v1_score, double v1_subscore, double v2_score, double v2_subscore)
  {
#ifdef DD_CXX11
    if(std::isnan(v1_subscore) || std::isnan(v2_subscore))
#else
	if(_isnan(v1_subscore) || _isnan(v2_subscore))
#endif
      return v1_score > v2_score;
    else
    {
      // finger
      //return sigmoid(v1.score())*sigmoid(v1.subscore()) > 
	      //sigmoid(v2.score())*sigmoid(v2.subscore());
      //return v1 > v2;
      //return v1.score() > v2.score();
      return std::min(v1_score,v1_subscore) >
	      std::min(v2_score,v2_subscore);
      //return std::sqrt(v1.score() * v1.subscore()) < 
	      //std::sqrt(v2.score() * v2.subscore());
      //return v1.subscore() > v2.subscore();
      
      // use our regression ! 
      //double a = 1;//0.00187905;
      //double b = 0;//1.48429e-05;
      //double s1 = a * v1.score() + b * v1.subscore();
      //double s2 = a * v2.score() + b * v2.subscore();
      //return s1 > s2;
    }        
  }
  
#ifdef DD_CXX11
  auto PR_Det_Sort_Fn = [](const FrameDet&v1, const FrameDet&v2)
  {
    return pr_comp(v1.score(),v1.subscore(),v2.score(),v2.subscore());
  };  
#endif
  
  void Scores::compute_pr(vector< double >& P, vector< double >& R, vector< double >& V) const
  {
#ifdef DD_CXX11
    double nSamples = tp(-inf) + fn(-inf);
    
    // compute TP (True positive indicator array
    vector<FrameDet> finger_dets = getDetections();
    std::sort(finger_dets.begin(),finger_dets.end(),
      [&](const FrameDet&v1, const FrameDet&v2)
      {
	return PR_Det_Sort_Fn(v1,v2);
      }
    );
    //std::reverse(finger_dets.begin(),finger_dets.end());
    vector<double> TP, FP;
	for(int iter = 0; iter < finger_dets.size(); ++iter)
	{
		FrameDet & det = finger_dets[iter];

      if(det.is_detection())
      {
	TP.push_back(det.correct());
	FP.push_back(!det.correct());
      }
	}
    vector<double> cumTP(TP.size(),0), cumFP(FP.size(),0);
    std::partial_sum(TP.begin(),TP.end(),cumTP.begin());
    std::partial_sum(FP.begin(),FP.end(),cumFP.begin());
    
    // compute the P and R rates
    P = cumTP / (cumTP + cumFP);
    R = cumTP / (nSamples);
    V = vector<double>(P.size(),0);
#else
	  // windows is a problem... lacks modern C++ compiler
	  P = vector<double>(1,qnan);
	  R = vector<double>(1,qnan);
	  V = vector<double>(1,qnan);
#endif
  }
  
  double Scores::pose_accuracy() const
  {
    return pose_correct/(pose_correct+pose_incorrect);
  }
  
  double Scores::f1(double t) const
  {
    return 2*p(t)*r(t)/(p(t)+r(t));
  }
  
  void write(FileStorage& fs, const string& , const Scores& score)
  {
    //write out tp, fp, fn 
    fs 
    << "{" 
    << "detections" << score.detections
    << "pose_correct" << score.pose_correct
    << "pose_incorrect" << score.pose_incorrect
    << "}";
  }
  
  void read(const FileNode& node, Scores& score, const Scores& default_value)
  {
    if(node.empty())
      score = default_value;
    else
    {
      read(node["detections"],score.detections);
      score.pose_correct = node["pose_correct"];
      score.pose_incorrect = node["pose_incorrect"];
    }
  }  
  
  Rect rectOfBlob(cv::Mat& blobIm, int targetVal)
  {
    // find the bouds
    int x_right = 0, x_left = blobIm.cols-1, y_bottom = 0, y_top = blobIm.rows -1;
    for(int yIter = 0; yIter < blobIm.rows; yIter++)
        for(int xIter = 0; xIter < blobIm.cols; xIter++)
            if(blobIm.at<unsigned char>(yIter,xIter) == targetVal)
            {
                x_right = std::max(x_right,xIter);
                x_left = std::min(x_left,xIter);
                y_bottom = std::max(y_bottom,yIter);
                y_top = std::min(y_top,yIter);
            }

    // construct the result
    if(x_right > x_left && y_bottom > y_top)
    {
        printf("found blobal hand = (%d, %d) to (%d, %d)\n",y_top,x_left,y_bottom,x_right);
        return Rect(Point(x_left,y_top),Point(x_right,y_bottom));
    }
    else
        return Rect();
  }
  
  double sign(double value)
  {
    if(value > 0)
      return +1;
    if(value < 0)
      return -1;
    else 
      return 0;
  }
  
  Rect operator*(double scalar, const Rect& r)
  {
    return Rect(scalar* r.tl(), scalar* r.br());
  }

  Rect operator+(const Rect& r1, const Rect& r2)
  {
    return Rect(r1.tl() + r2.tl(), r1.br() + r2.br());
  }

  string to_string(const Rect & r)
  {
    ostringstream oss;

    oss << "[" << r.tl() << " to " << r.br() << "]";

    return oss.str();
  }
  
  std::ostream& operator<<(std::ostream& out, Rect& r)
  {
    out << r.x << " " << r.y << " " << r.width << " " << r.height;
    
    return out;
  }

  std::istream& operator>>(std::istream& in, Rect& r)
  {
    in >> r.x;
    in >> r.y;
    in >> r.width;
    in >> r.height;
    
    return in;
  }
  
  // SECTION: FrameDet
  void write(FileStorage& fs, string , const FrameDet& frameDet)
  {
    fs << "{"
      << "correct" << (double)frameDet.correct()
      << "score" << frameDet.score() <<
      "}";
  }

  void read(FileNode fn, FrameDet& frameDet, FrameDet )
  {
    double dCorrect;
    dCorrect = fn["correct"];
    frameDet = FrameDet((bool)dCorrect,fn["score"],0);
  }
  
  bool FrameDet::is_detection() const
  {
    return m_score > -inf;
  }
  
  bool FrameDet::operator<(const FrameDet& other) const
  {
    if(m_score == other.m_score)
      return m_subscore < other.m_subscore;
    return m_score < other.m_score;
  }
  
  bool FrameDet::operator>(const FrameDet& other) const
  {
    if(m_score == other.m_score)
      return m_subscore > other.m_subscore;
    return m_score > other.m_score;
  }
  
  bool FrameDet::correct() const
  {
    return m_correct;
  }

  FrameDet::FrameDet(bool correct, double score, double subscore) : 
    m_correct(correct), m_score(score), m_subscore(subscore)
  {
  }

  double FrameDet::subscore() const
  {
    return m_subscore;
  }

  double FrameDet::score() const
  {
    return m_score;
  }
  
  ///
  /// SECTION: DetectorScores
  /// 
#ifdef DD_CXX11
  bool DetectorScores::ScoredBB::operator<(const DetectorScores::ScoredBB& other) const
  {
    return pr_comp(resp,qnan,other.resp,qnan);
  }
  
  void DetectorScores::compute_pr(
    vector< double >& P, 
    vector< double >& R, 
    vector< double >& V) const
  { 
    set<string> frames;
    for(auto gt : ground_truth)
      frames.insert(gt.first);
    for(auto det : detections_per_frame)
      frames.insert(det.first);
    vector<string> ordered_frames(frames.begin(),frames.end());
    
    // two statistics, R and R^2 per precision
    map<double/*r*/,KahanSummation> N;
    map<double/*r*/,KahanSummation> P1;
    map<double/*r*/,KahanSummation> P2;
    
    for(int bootStrapIter = 0; bootStrapIter < 1000; ++bootStrapIter)
    {
      //
      DetectorScores bootstrap_scores(matchFn);    
      
      // (1) generate a bootstrap sample
      for(int sampleIter = 0; sampleIter < frames.size(); sampleIter++)
      {
	// choose a frame
	string selected_frame = ordered_frames[rand()%ordered_frames.size()];
	string new_id = uuid();
	
	// copy the ground truth
	auto ground_truth_range = ground_truth.equal_range(selected_frame);
	for(auto iter = ground_truth_range.first; iter != ground_truth_range.second; ++iter)
	{
	  assert(iter->first == selected_frame);
	  bootstrap_scores.put_ground_truth(new_id,iter->second);
	}
	
	// copy the detections
	auto detection_range = detections_per_frame.equal_range(selected_frame);
	for(auto iter = detection_range.first; iter != detection_range.second; ++iter)
	{
	  assert(iter->first == selected_frame);
	  bootstrap_scores.put_detection(new_id,iter->second.BB,iter->second.resp);
	}
      }
      
      // 
      vector<double> sampleP, sampleR, sampleV;
      bootstrap_scores.compute_pr_simple(sampleP,sampleR,sampleV);
      // convert to a map
      map<double,double> RP;
      for(int iter = 0; iter < sampleP.size(); ++iter)
      {
	RP[sampleR[iter]] = sampleP[iter];
      }
      // this is how we interpolate the discrete/arbitrary 
      // PR values into a smoothly enumerated curve.
      //P = sampleP; R = sampleR; V = sampleV; return;
      for(double r = 0; r <= 1; r += .01) // for each recall rate
      {
	auto lb_iter = RP.lower_bound(r);
	if(RP.size() > 0 && lb_iter != RP.end())
	{	  
	  double p = lb_iter->second;
	  N[r] += 1.0;
	  P1[r] += p;
	  P2[r] += p*p;
	}
	else
	{
	  double p = 0;
	  N[r] += 1.0;
	  P1[r] += p;
	  P2[r] += p*p;
	}
      }      
    }
    
    // output the PR curve
    P = vector<double>(N.size(),0);
    R = vector<double>(N.size(),0);
    V = vector<double>(N.size(),0);
    auto rIter = N.begin();
    for(int iter = 0; iter < N.size() && rIter != N.end(); ++iter, ++rIter)
    {
      double r = rIter->first;
      double n = N[r].current_total();
      double p1 = P1[r].current_total()/n;
      double p2 = P2[r].current_total()/n;
      P[iter] = p1;
      R[iter] = r;
      //  ep = icdf('norm',(1-.95)/2,0,1) to compute the endpoint for a 95% CI
      double ep = 2.5758; // .999 CI
      // 1.96 for 95 % confidence interval
      V[iter] = ep*std::sqrt(p2 - p1*p1)/std::sqrt(detections.size());
    }    
    //std::reverse(P.begin(),P.end());
  }
  
  double DetectorScores::compute_pr_simple(
    vector< double >& P, 
    vector< double >& R, 
    vector<double>&V,
    SoftErrorFn softErrorFn) const
  {
    // (1) The multimap will keep the detections sorted by score.
    double soft_error = 0;
    
    // compute TP (True positive indicator array)
    // by assigning detections to ground truths
    vector<double> TP, FP;
    std::multimap<string,RectAtDepth> unassigned_gts = ground_truth;
    for(auto & scored_bb : detections)
    {
      // for all unassigned detections in the current frame
      auto eq_range = unassigned_gts.equal_range(scored_bb.filename);
      for(auto iter = eq_range.first; iter != eq_range.second; ++iter)
      {
	bool accept = matchFn(iter->second,scored_bb.BB,scored_bb.filename);
	if(accept)
	{
	  if(softErrorFn == nullptr)
	    soft_error = qnan;
	  else
	    soft_error += softErrorFn(iter->second,scored_bb.BB);
	  unassigned_gts.erase(iter);
	  TP.push_back(1);
	  FP.push_back(0);
	  goto NEXT_DETECTION;
	}
      }
      
      // failed to assign detection to a GT
      soft_error += inf;
      TP.push_back(0);
      FP.push_back(1);
      
      NEXT_DETECTION:
      ;
    }
      
    // accompulate to compute PR vectors
    vector<double> cumTP(TP.size(),0), cumFP(FP.size(),0);
    std::partial_sum(TP.begin(),TP.end(),cumTP.begin());
    std::partial_sum(FP.begin(),FP.end(),cumFP.begin());
    
    // compute the P and R rates
    P = cumTP / (cumTP + cumFP);
    R = cumTP / ((double)ground_truth.size());
    V = vector<double>(P.size(),0);
    
    return soft_error/detections.size();
  }
  
  DetectorScores::DetectorScores(DetectorScores::MatchFn matchFn) : 
    matchFn(matchFn)
  {
  }
  
  bool DetectorScores::put_detection(string filename, RectAtDepth detection, double resp)
  {
    if(resp == -inf || ((Rect_<double>)detection) == Rect_<double>())
      return false;
    
    detections.insert(ScoredBB{detection,resp,filename});
    detections_per_frame.insert(pair<string,ScoredBB>(filename,
							 ScoredBB{detection,resp,filename}));
    
    return true;
  }

  void DetectorScores::put_ground_truth(string filename, RectAtDepth gt)
  {
    if(((Rect_<double>)gt) == Rect_<double>())
      return;    
    
    ground_truth.insert(pair<string,RectAtDepth>(filename,gt));
  }
  
  void DetectorScores::merge(DetectorScores& other)
  {
    detections.insert(other.detections.begin(),other.detections.end());
    ground_truth.insert(other.ground_truth.begin(),other.ground_truth.end());
    detections_per_frame.insert(other.detections_per_frame.begin(),other.detections_per_frame.end());
  }
  
  double DetectorScores::p(double t) const
  {
    assert(t == -inf);
    vector< double > P, R, V;
    compute_pr(P,R,V);
    
    if(P.size() == 0)
      return qnan;
    
    return P.back();
  }

  double DetectorScores::r(double t) const
  {
    assert(t == -inf);
    vector< double > P, R, V;
    compute_pr(P,R,V);
    
    if(R.size() == 0)
      return qnan;
    
    return R.back();
  }
  
  double DetectorScores::v(double thresh) const
  {
    assert(thresh == -inf);
    vector<double> P,R,V;
    compute_pr(P,R,V);
    
    if(V.size() == 0)
      return qnan;
    
    return V.back();
  }    
#endif
  
  ///
  /// SECTION: RectAtDepth
  ///
  double RectAtDepth::dist(const RectAtDepth& other) const
  {
    Point2d us = rectCenter(*this);
    Point2d ot = rectCenter(other);
    
    Vec3d disp_cm(std::abs(us.x-ot.x),
		  std::abs(us.y-ot.y),
		  std::abs(z1 - other.z1));
    double dist = std::sqrt(disp_cm.dot(disp_cm));
    return dist;
  }
  
  RectAtDepth::RectAtDepth() : z1(qnan), z2(qnan)
  {
  }

  RectAtDepth::RectAtDepth(const Rect_< double >& copy) : 
    Rect_<double>(copy), z2(qnan), z1(qnan)
  {
  }

  RectAtDepth::RectAtDepth(Rect_<double> r, double z1, double z2) : 
    Rect_<double>(r), z1(z1), z2(z2)
  {
  }

  double RectAtDepth::depth() const
  {
    return z1;
  }

  double&RectAtDepth::depth()
  {
    return z1;
  }

  double RectAtDepth::volume() const
  {
    return area()*std::abs<double>(z2 - z1);
  }

  RectAtDepth RectAtDepth::intersection(const RectAtDepth&other) const
  {
    Rect rIntersection = static_cast<Rect_<double>>(*this) & static_cast<Rect_<double>>(other);
    double z1 = std::max<double>(this->z1,other.z1);
    double z2 = std::min<double>(this->z2,other.z2);
    return RectAtDepth(rIntersection,z1,z2);
  }
}
