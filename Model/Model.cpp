/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Detector.hpp"
#include "util.hpp"
#include "PXCSupport.hpp"
#include "vec.hpp"
#include "RespSpace.hpp"
#include "Log.hpp"
#include "FauxLearner.hpp"
#include "Semaphore.hpp"
#include "ThreadPool.hpp"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <highgui.h>
#include <sstream>
#include <boost/iterator.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/graph/graph_concepts.hpp>

namespace deformable_depth
{     
  ///
  /// SECTION: Model
  ///
  static void draw_one_det(Mat&showMe,DetectorResult&det,double strength)
  {
    // draw the detection
    { 
      double b = 255.0*strength;
      double r = 255 - b;
      auto bb = det->BB; cv::rectangle(showMe,bb.tl(),bb.br(),Scalar(b,0,r)); 
    }
    int index = 0;

    // draw the detection's parts
    for(string part_name : det->part_names())
    {
      Scalar color;
      if(part_name == "dist_phalan_5") // thumb is special case
      {
	color = Scalar(0,255,255);
      }
      else if(boost::regex_match(part_name,boost::regex(".*dist_phalan.*")))
      {
	color = Scalar(0,127,255); // orange
      }
      else
      {
	Vec3b vcolor = getColor(index++);
	Scalar color(vcolor[0],vcolor[1],vcolor[2]);
      }
      rectangle(showMe,det->getPart(part_name).BB.tl(),det->getPart(part_name).BB.br(),color);
    }
  }

  static void draw_parents(Mat&showMe, DetectorResult&det)
  {
    auto bb = det->BB;
    rectangle(showMe,bb.tl(),bb.br(),Scalar(0,255,0));    
    for(auto && parent : det->pyramid_parent_windows)
      draw_parents(showMe,parent);
  }

  Visualization Model::visualize_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const
  {
    // the dets must be sorted!
    dets = sort(dets);
    auto nms_dets = nms_w_list(dets,.25);

    // show all detections which have scores equal to the max scoring detection
    // assume dets is in sorted order
    int max_dets = 0;
    double resp;
    for(int iter = 0; iter < dets.size(); ++iter)
    {
      if(iter == 0)
	resp = dets[iter]->resp;
      else if(dets[iter]->resp < resp)
	break;
      
      max_dets++;
    }
    
    Mat showNMS = background.clone();
    Mat showMe = background.clone();
    Mat parents= background.clone();
    for(int iter = 0; iter < std::min<int>(max_dets,dets.size()); iter++)
    {
      DetectorResult &det = dets[iter];
      draw_one_det(showMe,det,1.0);
      draw_parents(parents,det);
    }
    for(int iter = 0; iter < nms_dets.size(); ++iter)
      draw_one_det(showNMS,nms_dets.at(iter),1-static_cast<float>(iter)/nms_dets.size());

    Visualization vis;
    vis.insert(showMe,"showMe");
    vis.insert(parents,"parents");
    vis.insert(showNMS,"showNMS");

    return vis;
  }

  Mat Model::vis_result(const ImRGBZ&im,Mat&background,DetectionSet&dets) const
  {
    return visualize_result(im,background,dets).image();
  }

  void Model::train_on_test(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    // NOP
  }
}
