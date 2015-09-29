/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "DynVolNN.hpp"
#include "Colors.hpp"

namespace deformable_depth
{
  DynVolNN::DynVolNN() :
    min_z(inf), max_z(-inf)
  {
    // load the training set
    ifstream notes("data/zball_big/notes.txt");
    while(notes)
    {
      string filename; notes >> filename;
      float z_min; notes >> z_min;
      float z_max; notes >> z_max;

      Mat T = read_depth_mm(safe_printf("data/zball_big/%",filename));
      if(T.empty())
      {
	log_file << "warning " << filename << " was empty" << endl;
	continue;
      }
      T = imclamp(T,z_min,z_max);
      log_im_decay_freq("Template",[&](){return eq(T);});
      templates.push_back(DynVolTempl{T,z_min,z_max});
      min_z = std::min(min_z,z_min);
      max_z = std::max(max_z,z_max);
    }

    // 
  }

  static int pyrLevel()
  {
    if(g_params.has_key("PYRDOWN"))
      return fromString<int>(g_params.require("PYRDOWN"));
    else
      return 0;
  }

  // static Mat convolve_sparse(const Mat&Z, const Mat&T,const Mat&valid_mask)
  // {
  //   assert(Z.type() == DataType<float>::type);
  //   assert(T.type() == DataType<float>::type);
  //   assert(valid_mask.type() == DataType<uint8_t>::type);
    
  //   Mat R(Z.rows,Z.cols,DataType<float>::type,Scalar::all(0));

  //   for(int yIter = 0; yIter < Z.rows; ++yIter)
  //     for(int xIter = 0; xIter < Z.cols; ++xIter)
  //     {
  // 	float z = Z.at<float>(yIter,xIter);
  // 	uint8_t valid = valid_mask.at<uint8_t>(yIter,xIter);
  // 	if(valid)
  // 	{
  // 	  for(int ty = 0; ty < T.rows; ++ty)
  // 	    for(int tx = 0; tx < T.cols; ++tx)
  // 	    {
  // 	      float t = T.at<float>(ty,tx);
  // 	      int rY = yIter - ty;
  // 	      int rX = xIter - tx;
  // 	      if(0 <= rY && rY < R.rows && 0 <= rX && rX < R.cols)
  // 		R.at<float>(rY,rX) += t*z;
  // 	    }
  // 	}
  //     }
    
  //   return R;
  // }
  
  // Mat matchTemplateL2Normalized(const Mat&Z, const Mat&match_T,const Mat&mask)
  // {
  //   Mat T; cv::flip(match_T,T,-1);
    
  //   Mat T_ones(T.rows,T.cols,DataType<float>::type,Scalar::all(1));
  //   Mat T2 = T.mul(T);
  //   Mat Z2 = Z.mul(Z);

  //   Mat SSD = T2.dot(T_ones) - 2 * convolve_sparse(Z,T,mask) - convolve_sparse(Z2,T_ones,mask);
  //   Mat NCS = sqrt(T2.dot(T_ones) * convolve_sparse(Z2,T_ones,mask));
    
  //   return SSD/NCS;
  // }
  
  class MatchPacket
  {    
  public:
    Mat r;
    float max_resp;
    Rect bb;
    DynVolTempl t;
    int stride;

    MatchPacket(){};
    
    MatchPacket(const SphericalOccupancyMap&SOM,const DynVolTempl&t,function<bool (const Mat&t_active)> checked)
    {
      //
      float maxRes = fromString<float>(g_params.require("MAX_RES"));
      float zRes   = fromString<float>(g_params.require("Z_RES")); // cm
      float sf = std::sqrt(maxRes/t.t.size().area());
      
      // get inputs
      Mat zSlice = imclamp(SOM.get_OM(),t.z_min,t.z_max);
      Mat valid  = imunclamped(SOM.get_OM(),t.z_min,t.z_max);
      Mat z_active = (zSlice.clone());
      Mat t_active = (t.t.clone());

      // down size if needed
      if(sf < 1)
      {
	cv::resize(t_active,t_active,Size(),sf,sf,params::DEPTH_INTER_STRATEGY);
	cv::resize(z_active,z_active,Size(),sf,sf,params::DEPTH_INTER_STRATEGY);
	stride = std::floor(1/sf);
	assert(stride >= 1);
      }
      else
	stride = 1;
      t_active = imround(t_active,zRes);
      z_active = imround(z_active,zRes);

      // skip duplicates
      if(checked(t_active))
      {
	return;
      }
      
      // match
      //Mat r = matchTemplateL2Normalized(z_active,t_active,valid);
      r; cv::matchTemplate(z_active, t_active, r, CV_TM_SQDIFF_NORMED);

      // upsize if needed
      if(sf < 1)
	cv::resize(r,r,Size(),1/sf,1/sf,params::DEPTH_INTER_STRATEGY);

      // normalize score
      double z_side = (t.z_max - t.z_min);
      r = -r/(z_side*t_active.size().area());

      // extract top detection for visualization and debug
      bb = Rect(extrema(r).maxLoc,t.t.size());
      if(rectContains(zSlice,bb))
      {
	max_resp = extrema(r).max;
      }
      else
	max_resp = -inf;
      this->t = t;
    }

    void log(string prefix,SphericalOccupancyMap&SOM)
    {
      Mat vis_r, vis_t, vis_z, vis_match,message,vis_raw_z;
      Mat zSlice = imclamp(SOM.get_OM(),t.z_min,t.z_max);
      
      // vis
      vis_raw_z = eq(SOM.get_OM());
      vis_r = eq(r);
      vis_t = eq(t.t);
      vis_z = eq(zSlice);
      Mat vis_t_full(vis_z.rows,vis_z.cols,DataType<Vec3b>::type,Scalar::all(255));
      {
	Mat src_roi = imroi(vis_t,Rect(Point(0,0),bb.size()));
	Mat dst_roi = imroi(vis_t_full,bb);
	src_roi.copyTo(dst_roi);
      }
      vis_match = im_merge(
	imroi(vis_r,Rect(Point(0,0),vis_z.size())),
	vis_z,
	vis_t_full);
      cv::rectangle(vis_match,bb.tl(),bb.br(),toScalar(BLUE));
      message = image_text(safe_printf("resp = %",extrema(r).max));
      vector<Mat> vs{vis_r,vis_t,vis_z,vis_match,vis_raw_z,message};
      log_im(prefix,tileCat(vs));
    }
    
    bool operator<(const MatchPacket&other) const
    {
      return max_resp > other.max_resp;
    }      
  };

  void DynVolNN::log_times() const
  {
    lock_guard<mutex> l(monitor);
    for(auto && pair : performance_times)
    {
      int z = pair.first;
      double mean_time = performance_times.at(z) / performance_counts.at(z);
      log_file << safe_printf("z[%,%] = % / % = % ms",
			      50*z+min_z,50*(z+1)+min_z,
			      performance_times.at(z),performance_counts.at(z),
			      mean_time) << endl;
    }
  }
  
  DetectionSet DynVolNN::detect(const ImRGBZ&im,DetectionFilter filter) const 
  {
    SphericalOccupancyMap SOM(im);

    vector<MatchPacket> packets(templates.size());
    TaskBlock proc_templates("proc_templates");
    tbb::concurrent_unordered_set<size_t> checked_templates;
    for(int iter = 0; iter < templates.size(); ++iter)
    {
      proc_templates.add_callee([&,iter]()
				{
				  Timer timer;
				  timer.tic();
				  const DynVolTempl&t = templates.at(iter);
				  packets.at(iter) = MatchPacket(SOM,t,[&](const Mat&t)
								 {
								   size_t hash = hash_code(t);
								   auto r = checked_templates.insert(hash);
								   if(!r.second)
								   {
								     cout << "template duplicate skipped! " <<
								       hash << endl;
								   }
								   return !r.second;
								 });
				  long interval = timer.toc();
				  lock_guard<mutex> l(monitor);
				  performance_times[(t.z_min - min_z)/50] += interval;
				  performance_counts[(t.z_min - min_z)/50] ++;
				});
    }
    proc_templates.execute();
    log_file << safe_printf("info: % checked among % templates",checked_templates.size(),templates.size()) << endl;
    std::sort(packets.begin(),packets.end());
    for(int iter = 0; iter < 1; ++iter, iter *= 2)
    {
      string fn = im.filename;
      for(char&c : fn)
	if(c == '/')
	  c = '_';
      packets.at(iter).log(safe_printf("packet_[%]_[%]_",fn,iter),SOM);
    }

    log_times();
    DetectionSet all_dets;
    TaskBlock take_dets("take_dets");
    int stride = 1;
    for(int yIter = stride/2; yIter < im.rows(); yIter += stride)
      take_dets.add_callee([&,yIter]()
			   {
			     for(int xIter = stride/2; xIter < im.cols(); xIter += stride)
			     {
			       float max_resp = -inf;
			       Rect max_bb;
			       for(auto && packet : packets)
			       {
				 if(packet.bb.size().area() <= 0)
				   continue;
				 
				 if(yIter < packet.r.rows && xIter < packet.r.cols)
				 {
				   float resp = packet.r.at<float>(yIter,xIter);
				   if(resp > max_resp)
				   {
				     max_resp = resp;
				     max_bb   = Rect(Point(xIter,yIter),packet.bb.size());
				   }				   
				 }				 
			       }//end for packet

			       auto det = make_shared<Detection>();
			       det->BB = max_bb;
			       det->resp = max_resp;
			       static mutex m; lock_guard<mutex> l(m);
			       all_dets.push_back(det);	  	
			     }// end for xIter
			   });  	
    take_dets.execute();
    all_dets = sort(all_dets);   
    return (all_dets);
  }

  DynVolNN::~DynVolNN()
  {
    log_times();
  }
  
  void DynVolNN::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
  }
  
  void DynVolNN::train_on_test(vector<shared_ptr<MetaData>>&training_set,
			     TrainParams train_params)
  {
  }
  
  Mat DynVolNN::show(const string&title)
  {
    return image_text("DynVolNN");
  }
}

