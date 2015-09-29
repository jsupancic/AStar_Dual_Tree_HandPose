/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Detection.hpp"
#include "util_rect.hpp"
#include "OcclusionReasoning.hpp"
#include <boost/graph/graph_concepts.hpp>
#include <boost/multi_array.hpp>
 
namespace deformable_depth
{
  /// SECTION: DetectionSEt
  DetectionSet::DetectionSet(const vector<DetectorResult>&cpy) : 
    vector<DetectorResult>(cpy)
  {
  }

  /// SECTION: Detection Filter
  DetectionFilter::DetectionFilter(float thresh, int nmax, std::string pose_hint) :
    thresh(thresh), nmax(nmax), pose_hint(pose_hint), supress_feature(false), 
    sort(true), verbose_log(false), require_dot_resp(false), allow_occlusion(true),
    testing_mode(false), is_root_template(true), slide_window_only(false),
    manifoldFn(manifoldFn_default), use_skin_filter(true)
  {
  }
  
  void DetectionFilter::apply(DetectionSet& detections)
  {
    bool sorted = true;
    if(this->sort)
    {
      std::sort(detections.begin(),detections.end(),
		[](const DetectorResult & lhs, const DetectorResult & rhs)
		{
		  return lhs->resp > rhs->resp;
		});
      //reverse(detections.begin(),detections.end());
      sorted = true;
    }
    else
    {
      // check that the detections are sorted
      for(int iter = 0; iter < static_cast<int>(detections.size()) - 1; ++iter)
	if(detections[iter]->resp < detections[iter+1]->resp);
	  sorted = false;
    }
    
    // apply max
    if(sorted)
    {
      int N = std::min<int>(this->nmax,detections.size());
      if(detections.size() > N)
	detections.erase(detections.begin()+(N-1),detections.end());
    }
    else if(nmax < numeric_limits<decltype(this->nmax)>::max())
      log_once("warning, could not nmax supress because input wasn't sorted");
  }
  
  ///
  /// SECTION: NMS
  ///
  DetectionSet sort(DetectionSet src)
  {
    // conversion to linked list is stupid but I'm lazy...
    list<DetectorResult> working_list(src.begin(),src.end());
    working_list.sort(
      [](const DetectorResult & lhs, const DetectorResult & rhs)
      {
	// biger BBs have more evidnece? 
	if(lhs->resp == rhs->resp)
	  return lhs->BB.area() > rhs->BB.area();
	else
	  return lhs->resp > rhs->resp;
      });
    //reverse(working_list.begin(),working_list.end());
    
    DetectionSet answer;
    answer.insert(answer.end(),working_list.begin(),working_list.end());
    return answer;     
  }
  
  DetectionSet nms_w_list(DetectionSet src, double overlap)
  {
    DetectionSet sorted_src = sort(src);
    list<DetectorResult> working_list(sorted_src.begin(),sorted_src.end());
    
    // Detection are sorted from high resp to low resp
    for(auto iter = working_list.begin(); iter != working_list.end(); iter++)
    {
      // take the next best, and supress the rest
      auto jter = iter; jter++;
      while(jter != working_list.end())
	if(rectIntersect((*iter)->BB,(*jter)->BB) > overlap)
	{
	  assert(jter != working_list.end());
	  jter = working_list.erase(jter);
	}
	else
	  jter++;
    }
    
    DetectionSet answer;
    answer.insert(answer.end(),working_list.begin(),working_list.end());
    return answer; 
  }
  
  DetectionSet nms_apx_w_array(const DetectionSet& src, double nmax, Size imSize)
  {
    DetectionSet sorted_src = sort(src);
    reverse(sorted_src.begin(),sorted_src.end());
    
    // 2 w * 2 h = nmax
    // w / h = W/H => w = W/H * h
    // => 2 W/H * h * 2 h = nmax
    // h^2 = nmax / (4 W/H )
    // h = sqrt(nmax / (4 W/H)
    // h = nmax / (4*w)
    int h = std::ceil(std::sqrt(nmax / (4*imSize.width/imSize.height)));
    int w = std::ceil(nmax/(4*h));
    assert((int)w > 0);
    assert((int)h > 0);
    boost::multi_array<DetPtr,4 > space(boost::extents[w][h][w][h]);
    for(DetPtr dp : sorted_src)
    {
      int x1 = clamp<int>(0,interpolate_linear(dp->BB.tl().x,0,imSize.width-1,0,w-1),w-1);
      int y1 = clamp<int>(0,interpolate_linear(dp->BB.tl().y,0,imSize.height-1,0,h-1),h-1);
      int x2 = clamp<int>(0,interpolate_linear(dp->BB.br().x,0,imSize.width-1,0,w-1),w-1);
      int y2 = clamp<int>(0,interpolate_linear(dp->BB.br().y,0,imSize.height-1,0,h-1),h-1);
      space[x1][y1][x2][y2] = dp;
    }
    
    // read from the supressed space
    DetectionSet supressed;
    for(int x1 = 0; x1 < w; x1++)
      for(int y1 = 0; y1 < h; y1++)
	for(int x2 = 0; x2 < w; x2++)
	  for(int y2 = 0; y2 < h; y2++)
	    if(space[x1][y1][x2][y2])
	      supressed.push_back(space[x1][y1][x2][y2]);
    return supressed;
  }
  
  ////
  /// SECTION: EPM stats
  ///
  EPM_Statistics::EPM_Statistics(string title) : title(title)
  {
  }
  
  EPM_Statistics::~EPM_Statistics()
  {
  }
  
  long EPM_Statistics::getCounter(string name) const
  {
    if(counters.find(name) != counters.end())
      return counters.at(name);
    else
      return 0;
  }
  
  vector< string > EPM_Statistics::toLines()
  {
    vector<string> lines;
    for(auto && counter : counters)
    {
      ostringstream oss;
      oss << counter.first << ": " << counter.second;
      lines.push_back(oss.str());
    }
    return lines;
  }
  
  void EPM_Statistics::print()
  {
    static mutex m; lock_guard<mutex> l(m);
    cout << "===================" << title << "======================" << endl;
    for(auto && counter : counters)
      cout << "EPM_Statistics: " << counter.first << ": " << counter.second << endl;
    cout << "=========================================" << endl;
  }
  
  void EPM_Statistics::count(string id, int increment)
  {
    unique_lock<mutex> l(monitor);
    counters[id] += increment;
  }
  
  void EPM_Statistics::add(const EPM_Statistics&other)
  {
    unique_lock<mutex> l(monitor);
    for(auto && counter : other.counters)
    {
      counters[counter.first] += counter.second;
    }
  }  
  
  ///
  /// SECTION: Detection
  ///
  
  double Detection::getDepth(const ImRGBZ&im) const
  {
    return (goodNumber(depth))?depth:medianApx(im.Z,BB,0);
  }
  
  bool Detection::operator<(const Detection&other) const
  {
    if(resp == other.resp)
      return rand()%2;
    else
      return resp < other.resp;
  }
  string Detection::toString()
  {
    // print pose
//     return printfpp("%s,%s,%f,%f,%f,%f",
// 		    src_filename.c_str(),pose.c_str(),
// 		    BB.x,BB.y,BB.width,BB.height);
    // don't print pose
    return printfpp("%s,%f,%f,%f,%f,%f",
		    src_filename.c_str(),
		    BB.x,BB.y,BB.width,BB.height,depth);
  }
  
  bool Detection::is_black_against(
    map<string,AnnotationBoundingBox>&positives,double ol_thresh) const
  {
    static boost::regex unlabeledRE(".*DontCare.*");
    bool black = true;
    // base case: check the root
    for(auto && pos : positives)
    {
      if(boost::regex_match(pos.first,unlabeledRE))
      {
	double ol = (BB & pos.second).area() / static_cast<double>(BB.area());
	if(ol >= ol_thresh)
	  black &= false;
      }
      else
	if(rectIntersect(BB,pos.second) >= ol_thresh)
	  black &= false;
    }
    
    // recursively: check the parts
    for(string part_name : part_names())
	black &= getPart(part_name).is_black_against(positives);
    return black;
  }
  
  bool Detection::is_occluded() const
  {
    return !is_visible(occlusion);
  }
  
  Detection::Detection(const Detection& other)
  {
    *this = other;
  }
  
  Detection& Detection::operator=(const Detection& copyFrom)
  {
    assert(this != nullptr);
    feature = [](){return SparseVector(0);};
    
    // these should be copied by value
    rawBB = copyFrom.rawBB;
    in_plane_rotation = copyFrom.in_plane_rotation;
    feature = copyFrom.feature;
    BB = copyFrom.BB;
    depth = copyFrom.depth;
    resp = copyFrom.resp;
    z_size = copyFrom.z_size;
    blob = copyFrom.blob;
    pose = copyFrom.pose;
    src_filename = copyFrom.src_filename;
    lr_flips = copyFrom.lr_flips;
    occlusion = copyFrom.occlusion;
    scale_factor = copyFrom.scale_factor;
    exemplar = copyFrom.exemplar;
    real = copyFrom.real;
    supressed = copyFrom.supressed;
    pyramid_parent_windows = copyFrom.pyramid_parent_windows;
    down_pyramid_count = copyFrom.down_pyramid_count;
    if(copyFrom.parts != nullptr)
    {
      parts.reset(new std::map<string,vector<Detection> >); 
      *parts = *copyFrom.parts;
    }
    if(copyFrom.m_part_names != nullptr)
    {
      m_part_names.reset(new set<string>);
      *m_part_names = *copyFrom.m_part_names;
    }
    pw_debug_stats = copyFrom.pw_debug_stats;
    
    return *this;
  }
  
  vector< Point2d > Detection::keypoints(bool toplevel) const
  {
    vector<Point2d> keypoints;
    
    if(toplevel)
      write_corners(keypoints,BB);
    else
      keypoints.push_back(rectCenter(BB));
    
    for(string part_name : part_names())
    {
      vector<Point2d> new_keypoints = getPart(part_name).keypoints(false);
      keypoints.insert(keypoints.end(),new_keypoints.begin(),new_keypoints.end());
    }
    
    return keypoints;
  }

  set< string > Detection::parts_flat() const
  {
    set<string> flat_parts;
    
    for(string part_name : part_names())
    {
      set<string> new_parts = getPart(part_name).parts_flat();
      flat_parts.insert(new_parts.begin(),new_parts.end());
    }
    
    return flat_parts;
  }
  
  string Detection::print_resp_computation() const
  {
    float resp_here = resp;
    string part_resp_comps;
    
    for(string part_name : part_names())
    {
      resp_here -= getPart(part_name).resp;
      part_resp_comps += string(" ") + getPart(part_name).print_resp_computation();
    }
    
    string resp_comp; 
    string occlusion_mark = is_occluded()?"O":"V";
    if(parts != nullptr && parts->size() > 0)
    {
      resp_comp = printfpp("%+f = %+f%s ",resp,resp_here,occlusion_mark.c_str());
      resp_comp += "(" + part_resp_comps + ")";
    }
    else
      resp_comp = printfpp("%+f%s",resp,occlusion_mark.c_str());
    return resp_comp;
  }
   
  void Detection::tighten_bb()
  {
    BB = Rect();

    if(parts != nullptr)
    {      
      assert(part_names().size() == parts->size());
      for(string part_name : part_names())
      {
	for(int iter = 0; iter < (*parts).at(part_name).size(); ++iter)
	{
	  Detection&subdet = (*parts).at(part_name)[iter];
	  if(BB == Rect_<double>())
	    BB = subdet.BB;
	  else
	    BB |= subdet.BB;    	  
	}
      }
    }
  }

  void Detection::applyAffineTransform(Mat& affine)
  {
    BB = rect_Affine(BB,affine);
    lr_flips++;    
    
    if(parts != nullptr)
    {      
      assert(part_names().size() == parts->size());
      for(string part_name : part_names())
      {
	for(int iter = 0; iter < (*parts).at(part_name).size(); ++iter)
	{
	  Detection&subdet = (*parts).at(part_name)[iter];
	  subdet.applyAffineTransform(affine);
	}
      }
    }
  }
  
  void Detection::emplace_part(string part_name, const Detection&part_detection, bool keep_old)
  {
    if(parts == nullptr)
      parts.reset(new map<string,vector<Detection> >());
    if(m_part_names == nullptr)
      m_part_names.reset(new set<string>);
    
    m_part_names->insert(part_name);
    if(keep_old)
    {
      log_file << "warning: keep_old blows up memory consumption!!!" << endl;
      (*parts)[part_name].push_back(part_detection);
    }
    else if((*parts)[part_name].size() > 0)
    {
      (*parts)[part_name].front() = part_detection;
      assert((*parts)[part_name].size() == 1);
    }
    else
    {
      (*parts)[part_name].push_back(part_detection);
      assert((*parts)[part_name].size() == 1);
    }
  }
  
  void Detection::set_parts(map< string, AnnotationBoundingBox > parts)
  {
    // clear
    if(this->parts)
      this->parts->clear();
    for(auto & pair : parts)
    {
      Detection part_det;
      part_det.BB = pair.second;
      emplace_part(pair.first, part_det, false);
    }
  }

  const Detection& Detection::getPart(string part_name) const
  {
    return (*parts).at(part_name).back();
  }

  const Detection& Detection::getPartCloseset(string part_name, Rect_< double > bb) const
  {
    const Detection* closeset_detection;
    double max_overlap = -inf;
    
    for(const Detection& part_candidate : (*parts).at(part_name))
    {
      double overlap = rectIntersect(bb,part_candidate.BB);
      if(overlap > max_overlap)
      {
	max_overlap = overlap;
	closeset_detection = &part_candidate;
      }
    }
    
    assert(closeset_detection != nullptr);
    return *closeset_detection;
  }
  
  Detection& Detection::getPart(string part_name)
  {
    const Detection*cthis = this;
    return const_cast<Detection&>(cthis->getPart(part_name));
  }

  Detection& Detection::getPartCloseset(string part_name, Rect_< double > bb)
  {
    const Detection*cthis = this;
    return const_cast<Detection&>(cthis->getPartCloseset(part_name,bb));
  }

  const set< string >& Detection::part_names() const
  {
    if(m_part_names == nullptr)
    {
      static const set<string> empty_set;
      return empty_set;
    }
    else
      return *m_part_names;
  }
   
  void write(FileStorage& fs, string , const Detection&detection)
  {
    fs << "{";
    if(detection.feature)
      fs << "feature" << static_cast<vector<float>>(detection.feature());
    else 
      fs << "feature" << vector<float>();
    fs << "BB" << detection.BB;
    fs << "depth" << detection.depth;
    fs << "resp" << detection.resp;
    fs << "blob" << detection.blob;
    fs << "pose" << detection.pose;
    fs << "source_filename" << detection.src_filename;
    fs << "supressed" << detection.supressed;
    fs << "z_size" << detection.z_size;
    // write the parts
    if(detection.parts != nullptr && detection.parts->size() > 0)
    {
      map<string,Detection> best_parts;
      for(string part_name : detection.part_names())
	best_parts[part_name] = detection.getPart(part_name);      
      fs << "parts"; write(fs,best_parts);
    }
    fs << "lr_flips" << detection.lr_flips;
    fs << "occlusion" << detection.occlusion;
    fs << "real" << detection.real;
    fs << "}";
  }   
   
  void read(const cv::FileNode& fn, Detection& detection, Detection )
  {
    // load the feature
    vector<float> feature; 
    fn["feature"] >> feature;
    detection.feature = [feature](){return feature;};
    
    Rect bb;
    read(fn["BB"],bb);
    detection.BB = bb;
    fn["depth"] >> detection.depth;
    fn["resp"] >> detection.resp;
    fn["blob"] >> detection.blob;
    fn["pose"] >> detection.pose;
    fn["source_filename"] >> detection.src_filename;
    fn["real"] >> detection.real;
    fn["z_size"] >> detection.z_size;
    
    // load the parts
    if(!fn["parts"].empty())
    {
      map<string,Detection> best_parts;
      read(fn["parts"],best_parts);
      for(auto best_part : best_parts)
	detection.emplace_part(best_part.first,best_part.second);
    }
    
    fn["lr_flips"] >> detection.lr_flips;
    fn["occlusion"] >> detection.occlusion;
  }
  
  void translate(DetectionSet& detections, Vec2d offset)
  {
    for(DetectorResult & det_result : detections)
    {
      det_result->BB.x += offset[0];
      det_result->BB.y += offset[1];
    }
  }  
  
  DetectorResult nearest_neighbour(const DetectionSet& dets, Rect_< double > bb)
  {
    DetectorResult nn;
    
    for(const DetectorResult&det : dets)
      if(!nn || rectIntersect(nn->BB,bb) < rectIntersect(det->BB,bb))
	nn = det;
    
    return nn;
  }

  DetectionSet removeTruncations(const ImRGBZ&im,DetectionSet src)
  {
    DetectionSet filtered;

    for(auto && det : src)
    {
      if(rectContains(im.Z,det->BB))
	filtered.push_back(det);
    }

    return filtered;
  }

  DetectionSet fromMetadata(MetaData&datum)
  {
    // get the parts
    auto all_poss = datum.get_positives();
    decltype(all_poss) poss;
    for(auto && pos : all_poss)
    {
      if(pos.second.tl() != Point2d())
      {	
	log_once(safe_printf("info(fromMetadata%) poss = % TAKE = %",
			     datum.get_filename(),pos.first,pos.second));
	poss.insert(pos);
      }
      else
	log_once(safe_printf("info(fromMetadata%) poss = % SKIP = %",
			     datum.get_filename(),pos.first,pos.second));
    }

    // convert the datum to a detection
    DetectorResult det = make_shared<Detection>();    
    det->BB = poss["HandBB"];
    log_once(safe_printf("info(fromMetadata%) # poss = %",datum.get_filename(),poss.size()));
    det->set_parts(poss);

    DetectionSet results;
    results.push_back(det);
    return results;
  }
}
