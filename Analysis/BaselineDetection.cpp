/**
 * Copyright 2012: James Steven Supancic III
 **/

#include <Video.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>

#include "BaselineDetection.hpp"
#include "InverseKinematics.hpp"
#include "RegEx.hpp"
#include "util.hpp"
#include "util_file.hpp"

namespace deformable_depth
{
  using namespace cv;
  using namespace std;  
  
	void write(cv::FileStorage &fs,const std::string &,const BaselineDetection&writeme)
	{
		fs << "{";
		//fs << "blob" << writeme.blob;
		fs << "bb" << writeme.bb;
		fs << "parts"; write(fs,writeme.parts);
		fs << "filename" << writeme.filename;
		fs << "resp" << writeme.resp;
		fs << "notes" << writeme.notes;
		fs << "}";
	}

  void write(cv::FileStorage &fs,const std::string &s,const shared_ptr<BaselineDetection>&writeme)
  {
    write(fs,s,*writeme);
  }
  
  void read(const FileNode&fn, shared_ptr<BaselineDetection>&readme, shared_ptr<BaselineDetection> def)
  {
    if(not readme)
      readme.reset(new BaselineDetection());
    read(fn,*readme,BaselineDetection());
  }

#ifdef DD_CXX11
  void read(const FileNode& node, BaselineDetection& pxcDet, BaselineDetection)
  {
    node["blob"] >> pxcDet.blob;
    read(node["bb"],pxcDet.bb);
    if(goodNumber(pxcDet.bb.x)) pxcDet.bb.x = std::round(pxcDet.bb.x);
    if(goodNumber(pxcDet.bb.y)) pxcDet.bb.y = std::round(pxcDet.bb.y);
    if(goodNumber(pxcDet.bb.width)) pxcDet.bb.width = std::abs(std::round(pxcDet.bb.width));
    if(goodNumber(pxcDet.bb.height)) pxcDet.bb.height = std::abs(std::round(pxcDet.bb.height));
    read<BaselineDetection>(node["parts"],pxcDet.parts);
    if(node["resp"].empty())
      pxcDet.resp = 0;
    else
    {
      node["resp"] >> pxcDet.resp;
      if(pxcDet.resp == -inf && pxcDet.bb.size().area() > 0)
	pxcDet.resp = -numeric_limits<float>::max();
    }
    node["filename"] >> pxcDet.filename;
    node["notes"] >> pxcDet.notes;
    
    // prune the part set
	typedef std::map<string,BaselineDetection> PartMap;
    PartMap parts;
    for(PartMap::iterator iter = pxcDet.parts.begin(); iter != pxcDet.parts.end(); ++iter)
      if(pxcDet.include_part(iter->first))
	parts[iter->first] = iter->second;
    pxcDet.parts = parts;
  }
  
  string BaselineDetection::respString() const
  {
    if(resp > -numeric_limits<float>::max())
       return printfpp("%f",resp);
    else
      return "?";
  }
#endif

  ///
  /// Section: Baseline Detection
  ///
  BaselineDetection::BaselineDetection() : resp(-inf), remapped(false)
  {
  }
  
  void BaselineDetection::draw(Mat&im,Scalar color)
  {
	  rectangle(im,bb.tl(),bb.br(),color);
	  for(map<string,BaselineDetection>::iterator iter = parts.begin();
		  iter != parts.end(); ++iter)
	  {
		  iter->second.draw(im,color);
	  }
  }
  
#if defined(DD_CXX11)
  BaselineDetection loadBaseline_text_one(ifstream&input, int length, int&frame)
  {
    BaselineDetection result;
    
    // read the frame
    string header; getline(input,header);boost::algorithm::trim(header);
    if(header == "")
      return BaselineDetection();
    log_file << printfpp("loadBaseline_txt record = %s",header.c_str()) << endl;
    vector<string> header_numbers = regex_match(header,boost::regex("\\d+"));
    if(header_numbers.size() == 0)
      return BaselineDetection();
    frame = fromString<int>(header_numbers[0]);
    log_file << printfpp("loadBaseline_text_one: parsed greg's frame %d",frame) << endl;
    
    // read the root BB
    double x, y, w, h;
    //x /= 2;
    //y /= 2;
    input >> x >> y >> w >> h;
    //cout << printfpp("read hand BB: %f %f %f %f",x,y,w,h) << endl;
    result.bb = rectFromCenter(Point(x,y),Size(w,h));
    log_file << "loadBaseline_text_one: parsed hand BB: " << result.bb << endl;
      
    // reasponce
    input >> result.resp;
    log_file << "loadBaseline_text_one: parsed resp : " << result.resp << endl;
    
    // generate the detection
    //assert(frame < length);
    
    for(int iter = 1; iter <= 5; ++iter)
    {
      string finger_line; 
      do {
	getline(input,finger_line);
	boost::algorithm::trim(finger_line);
      } while(finger_line=="" and input);
      vector<string> finger_numbers = regex_match(finger_line,boost::regex("\\d+"));
      BaselineDetection finger;
      finger.resp = result.resp;
      if(finger_numbers.size() == 2)
      {
	int x, y;
	istringstream iss(finger_line);
	iss >> x >> y;
	//x /= 2;
	//y /= 2;
	double hand_side = std::sqrt(result.bb.area());
	Size sz(hand_side/10,hand_side/10);
	finger.bb = rectFromCenter(Point2i(x,y),sz);
      }
      else if(finger_numbers.size() == 4)
      {
	Point2d center(
	  fromString<double>(finger_numbers[0]),
	  fromString<double>(finger_numbers[1])
	);
	Size2f size(
	  fromString<double>(finger_numbers[2]),
	  fromString<double>(finger_numbers[3])
	);
	finger.bb = Rect(center,size);
      }
      else
      {
	log_file << "warning: finger_numbers.size() = " << finger_numbers.size() << endl;
	log_file << "warning: finger_line = " << finger_line << endl;
	//assert(false);
      }    
      
      // write the new part
      string part_name = printfpp("dist_phalan_%d",iter);
      result.parts[part_name] = finger;
      log_file << "loadBaseline_text_one: parsed part: " << part_name << " : " << finger.bb << endl;      
    }   

    return result; 
  }

  vector< BaselineDetection > loadBaseline_text(string filename, int length)
  {
    log_file << printfpp("Greg: loading %s ",filename.c_str()) << endl;
    // Greg's files
    vector<BaselineDetection> track;
    if(length > 0)
      track.resize(length);
    ifstream input(filename,std::ifstream::in);
    if(!input.is_open())
    {
      cout << "Failed to open: " << filename << endl;
      assert(input.is_open());
    }
    
    while(input)
    {
      //cout << "==================" << endl;
      assert(input.good());
      
      int frame;
      BaselineDetection loaded_det = loadBaseline_text_one(input,length,frame);
      frame = std::max<int>(0,frame--);
      if(loaded_det.bb == Rect())
	continue;
      
      // resize if the track if required
      if(frame >= track.size())
	track.resize(frame+1);
      
      if(track.at(frame).bb == Rect())
	track.at(frame) = loaded_det;
      //else if(track[frame].resp < loaded_det.resp)
      else if(track.at(frame).bb.area() < loaded_det.bb.area())
      {
	// take the larger because it is nearer?
	track.at(frame) = loaded_det;
      }
    }
    
    // interpolate RC
    interpolate(track);
    post_interpolate_parts(track);

    return track;    
  }
  
  multimap< int, BaselineDetection > loadBaseline_yml(string filename, int length)
  {
    // James' files
    log_file << printfpp("James: loading %s ",filename.c_str()) << endl;
    FileStorage store(filename,FileStorage::READ);
    if(!store.isOpened())
      cout << "Filed to open : " << filename << endl;
    assert(store.isOpened());
    multimap<int, BaselineDetection> track;
    if(length > 0)
      for(int iter = 0; iter < length; iter++)
	track.insert(pair<int,BaselineDetection>(iter,BaselineDetection()));
    
    for(FileNodeIterator iter = store.root().begin(); iter != store.root().end(); ++iter)
    {
      // skip all but the first detection
      std::vector< std::string > num_matches = regex_match(
	(*iter).name(), boost::regex("[\\d]+"));
      //cout << "matches = " << num_matches.size() << endl;
      int frame_num = fromString<int>(num_matches[0]);
      int det_num = num_matches.size()>1?fromString<int>(num_matches[1]):0;
      //cout << "det_num = " << num_matches[1] << " as frame " << frame_num << endl;
      log_file << "loading: " << (*iter).name() << " into " << frame_num << endl;
      
      // frame past end of video
      if(length > 0 && frame_num >= length)
      {
	log_file << safe_printf("warning: frame_num(%) >= video length(%)!",frame_num,length) << endl;
	continue;
      }
      
      // load the label for this frame
      BaselineDetection bdet; *iter >> bdet;
      track.insert(pair<int,BaselineDetection>(frame_num,bdet));
    }
    
    store.release();
    return track;    
  }

  vector< BaselineDetection > loadBaseline(string filename, int length)
  {
    if(boost::regex_match(filename,boost::regex(".*txt")))
    {
      return loadBaseline_text(filename,length);
    }
    else
    {
      multimap< int, BaselineDetection > mm_result = loadBaseline_yml(filename,length);
      cout << "loadBaseline_yml returned " << mm_result.size() << endl;
 
      // take the highest scoring detection per frame.
      vector<BaselineDetection> result;
      if(length > 0)
	result.resize(length);
      for(auto r : mm_result)
      {
	// skips
	if(g_params.option_is_set("SCORE_SKIP_LEFT") and 
	   boost::regex_match(r.second.notes,boost::regex(".*left.*",boost::regex::icase)))
	  continue;
	if(g_params.option_is_set("SCORE_SKIP_RIGHT") and 
	   boost::regex_match(r.second.notes,boost::regex(".*right.*",boost::regex::icase)))
	  continue;
	
	//log_file << "candidate resp = " << r.second.resp << endl;
	int frame_num = r.first;
	result.resize(std::max((int)result.size(),frame_num+1));
	if(result[frame_num].bb == Rect() or result[frame_num].resp < r.second.resp)
	{
	  string p_first = std::to_string(frame_num);
	  string p_second = toString(r.second.bb);
	  log_file << safe_printf("writing: filename(%) frameNum(%) into rect(%)",filename,p_first,p_second) << endl;
	  result[frame_num] = r.second;
	}
      }
      return result;
    }
  }
#endif
  
  void BaselineDetection::scale(float factor)
  {
    bb.x /= 2;
    bb.y /= 2;
    bb.width /= 2;
    bb.height /= 2;    
    
	for(map<string,BaselineDetection>::iterator iter = parts.begin();
		iter != parts.end(); ++iter)
      iter->second.scale(factor);
  }
  
#ifdef DD_CXX11  
  BaselineDetection::BaselineDetection(const Detection&copyMe)
  {
    blob = copyMe.blob;
    bb = copyMe.BB;
    filename = copyMe.src_filename;
    resp = copyMe.resp;
    remapped = false;
    notes = copyMe.pose;
    
    for(string part_name : copyMe.part_names())
    {
      if(include_part(part_name))
	parts[part_name] = BaselineDetection(copyMe.getPart(part_name));
    }
  }
  
  bool BaselineDetection::include_part(string part_name) const
  {
    return boost::regex_match(part_name,boost::regex("dist_phalan_.*"));
  }

#endif

  //#ifdef DD_ENABLE_HAND_SYNTH
  BaselineDetection operator*(double weight, const BaselineDetection& mult)
  {
    BaselineDetection result(mult);
    
    result.bb = weight * result.bb;
    for(auto && part : result.parts)
      part.second = weight * part.second;
    
    return result;
  }

  BaselineDetection operator+(const BaselineDetection& lhs, const BaselineDetection& rhs)
  {
    BaselineDetection result(lhs);
    
    result.bb = result.bb + rhs.bb;
    for(auto && part : result.parts)
      part.second = part.second + rhs.parts.at(part.first);
      
    return result;
  }

  void BaselineDetection::interpolate_parts(const BaselineDetection& source)
  {
    assert(source.bb != Rect());
    assert(bb != Rect());
    
    for(auto && part : source.parts)
      if(parts.find(part.first) == parts.end() || parts.at(part.first).bb == Rect())
      {
	// merge the part...
	Mat affine = affine_transform(source.bb,bb);
	Rect part_bb = part.second.bb;
	Rect transformed_part_bb = rect_Affine(part_bb,affine);
	parts[part.first].bb = transformed_part_bb;
      }
      
    assert(parts.size() >= source.parts.size());
  }
  
  void dilate(vector< BaselineDetection >& track, int bandwidth)
  {
    vector< BaselineDetection > old_track = track;
    auto valid = [&](int index){return old_track[index].bb != Rect();};
    
    for(int iter = 0; iter < track.size(); ++iter)
      if(valid(iter))
	for(int jter = iter - bandwidth; jter < iter + bandwidth; ++jter)
	  if(0 <= jter && jter < track.size() && !valid(jter))
	    track[jter] = track[iter];
  }

  void interpolate(vector< BaselineDetection >& track)
  {
    vector<BaselineDetection*> next_valid(track.size(),nullptr);
    vector<BaselineDetection*> last_valid(track.size(),nullptr);
    vector<double> dist_to_next_valid(track.size(),inf);
    vector<double> dist_to_last_valid(track.size(),inf);
    
    auto valid = [&](int index){return track[index].bb != Rect();};
    
    // compute the weights in linear time
    for(int iter = 0; iter < track.size(); ++iter)
    {
      if(valid(iter))
      {
	dist_to_last_valid[iter] = 0;
	last_valid[iter] = &track[iter];
      }
      else if(iter == 0)
      {
	last_valid[iter] = nullptr;
	dist_to_last_valid[iter] = inf;
      }
      else
      {
	last_valid[iter] = last_valid[iter - 1];
	dist_to_last_valid[iter] = dist_to_last_valid[iter - 1] + 1;
      }   
    }
    // backwards pass
    for(int iter = track.size() - 1; iter >= 0; --iter)
    {
      if(valid(iter))
      {
	dist_to_next_valid[iter] = 0;
	next_valid[iter] = &track[iter];
      }
      else if(iter == track.size() - 1)
      {
	dist_to_next_valid[iter] = inf;
	next_valid[iter] = nullptr;
      }
      else
      {
	dist_to_next_valid[iter] = dist_to_next_valid[iter + 1] + 1;
	next_valid[iter] = next_valid[iter+1];
      }
    }
    
    // forwards pass, do the damn iterpolation
    for(int iter = 0; iter < track.size(); ++iter)
      if(!valid(iter))
      {
	if(next_valid[iter] == nullptr)
	{
	  assert(last_valid[iter] != nullptr);
	  track[iter] = *last_valid[iter];
	}
	else if(last_valid[iter] == nullptr)
	{
	  assert(next_valid[iter] != nullptr);
	  track[iter] = *next_valid[iter];
	}
	else
	{
	  double weight_last = dist_to_next_valid[iter]/(dist_to_last_valid[iter] + dist_to_next_valid[iter]);
	  double weight_next = 1 - weight_last;
	  track[iter] = weight_last * (*last_valid[iter]) + weight_next * (*next_valid[iter]);
	}
      }
  }
  
  void post_interpolate_parts(vector< BaselineDetection >& track)
  {
    for(int iter = 0; iter < track.size(); ++iter)
    {
      if(track[iter].bb == Rect())
	cout << "iter = " << iter << endl;
      assert(track[iter].bb != Rect());
    }    
    
    for(int iter = 0; iter < 2; ++ iter)
    {
      for(int iter = 1; iter < track.size(); ++iter)
	track[iter].interpolate_parts(track[iter-1]);
      
      for(int iter = track.size() - 2; iter >= 0; --iter)
	track[iter].interpolate_parts(track[iter+1]);
    }
  }

#ifdef DD_ENABLE_HAND_SYNTH      
  PoseRegressionPoint interpolate_ik_regress_full_hand_pose_vis_ik
  (vector< BaselineDetection >& track, int frameIter, Rect handBB)
  {
    return ik_regress(track[frameIter]);
  }
  
  static void interpolate_ik_regress_full_hand_pose_vis
  (vector< BaselineDetection >& track, int frameIter, 
   Rect handBB,PoseRegressionPoint&ik_match, Mat&at)
  {
    Mat vis(240,320,DataType<Vec3b>::type,Scalar::all(0));
    cv::rectangle(vis,handBB.tl(),handBB.br(),Scalar(255,100,100));
    for(auto & pair : track[frameIter].parts)
    {
      Rect partBB = pair.second.bb;
      cv::rectangle(vis,partBB.tl(),partBB.br(),Scalar(255,100,100));
    }
    for(auto & pair : ik_match.parts)
    {
      Rect partBB = pair.second;
      partBB = rect_Affine(partBB,at);
      log_file << "partBB " << pair.second << " => " << partBB << endl;
      cv::rectangle(vis,partBB.tl(),partBB.br(),Scalar(100,100,255));
    }
    log_im_decay_freq("interpolate_ik_regress_full_hand_pose",vis);    
  }
  
  void interpolate_ik_regress_full_hand_pose(vector< BaselineDetection >& track)
  {
    // debug mutex
    //static mutex m; lock_guard<mutex> l(m);
    cout << "++interpolate_ik_regress_full_hand_pose " << endl;
    
    TaskBlock ik_track("ik_track");
    for(int frameIter = 0; frameIter < track.size(); frameIter++)
    {
      // skip unannotated frames
      if(frameIter % params::video_annotation_stride() != 0)
	continue;
      
      // skip frames with no labeled hand
      Rect handBB = track[frameIter].bb;
      if(handBB == Rect())
      {
	log_file << "continue frame = " << frameIter << endl;
	continue;
      }
      
      ik_track.add_callee([&,handBB,frameIter]()
      {
	// do the regression
	PoseRegressionPoint ik_match = interpolate_ik_regress_full_hand_pose_vis_ik
	  (track, frameIter, handBB);
	shared_ptr<PoseRegressionPoint> ik_match_ptr(new PoseRegressionPoint(ik_match));
	for(int jter = frameIter; jter < 
	  std::min<int>(frameIter + params::video_annotation_stride(),track.size()-1); ++jter)
	  track[jter].pose_reg_point = ik_match_ptr;
	
	// compute the affine transform from the input handBB to the IKHBB
	Mat at = affine_transform(ik_match.parts["HandBB"],handBB);
	
	// visualize the result
	interpolate_ik_regress_full_hand_pose_vis
	  (track, frameIter, handBB,ik_match, at);
	
	// compute the max resp over parts
	double max_part_resp = -inf;
	if(track[frameIter].parts.size() == 0)
	  max_part_resp = track[frameIter].resp;
	else
	  for(auto & part : track[frameIter].parts)
	    max_part_resp = std::max(max_part_resp,part.second.resp);
	  
	// now that we're convinced it works, use it to update the track
	for(auto & reg_part : ik_match.parts)
	{
	  if(track[frameIter].parts[reg_part.first].bb == Rect())
	  {
	    Rect partBB = reg_part.second;
	    partBB = rect_Affine(partBB,at);	  
	    track[frameIter].parts[reg_part.first].bb = partBB;
	    track[frameIter].parts[reg_part.first].resp = max_part_resp;
	    log_file << "added: " << reg_part.first << " resp = " << max_part_resp << endl;
	  }
	}	
      });
    }
    ik_track.execute();
    cout << "--interpolate_ik_regress_full_hand_pose " << endl;
  }
#endif

  ///
  /// SECTION: Showbaseline on video
  /// 
  void show_baseline_on_video()
  {
#if ! defined(WIN32) and DD_ENABLE_HAND_SYNTH
    shared_ptr<Video> video  = load_video(g_params.require("VIDEO"));
    multimap< int, BaselineDetection > mm_result = 
      loadBaseline_yml(g_params.require("BASELINE"),-1);
      
    vector<Scalar> colors{
      Scalar(0,255,0),
      Scalar(255,0,0),
      Scalar(0,0,255)
    };
      
    for(int iter = 1; iter < video->getNumberOfFrames(); ++iter)
    {
      shared_ptr<MetaData> frame = video->getFrame(iter,true);
      shared_ptr<ImRGBZ> im = frame->load_im();
      
      Mat RGB = im->RGB.clone();
      Mat Depth = imageeq("",im->Z.clone(),false,false);
      auto range = mm_result.equal_range(iter);
      int num = 0;
      for(auto iter = range.first; iter != range.second; ++iter)
      {
	if(num >= colors.size())
	  assert(false);
	BaselineDetection & det = iter->second;
	det.draw(RGB,colors[num++]);
	det.draw(Depth,colors[num++]);
      }
      
      log_im(printfpp("%d",iter),RGB);
      log_im(printfpp("%d",iter),Depth);
    }
      
    cout << "DONE" << endl;
#else
	  assert(false);
#endif
  }
}

