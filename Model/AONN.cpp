/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "AONN.hpp"
#include "ScanningWindow.hpp"

namespace deformable_depth
{
  AONN_Model::AONN_Model()
  {
  }

  DetectionSet AONN_Model::detect_linear(const ImRGBZ&im,DetectionFilter filter) const
  {
    // convert the image to a template
    //DiscreteVolumetricTemplate imT(im);    
    AutoAlignedTemplate imT(im);
    Vec3i res = imT.resolution();

    // convolve
    DetectionSet dets;
    for(auto && pair : allTemplates)
    {
      const string&uuid = pair.first;
      auto&T = pair.second;

      Vec3i tRes = T.resolution();

      for(int xIter = 0; xIter < res[0]; ++xIter)
	for(int yIter = 0; yIter < res[0]; ++yIter)
	{
	  Point2i p0(xIter,yIter);
	  Point2i p1(xIter + tRes[0],yIter + tRes[1]);
	  float depth, cor = imT.cor(T,p0,depth);
	  // alloc the detection
	  auto detection = make_shared<Detection>();
	  detection->BB = Rect(p0,p1);
	  detection->resp = -cor;
	  detection->depth = depth;
	  // generate parts
	  auto parts = parts_per_ex.at(uuid);
	  set_det_positions(T,uuid,T.getExtractedFrom(),parts,detection,depth);
	  // add the detection to the set
	  dets.push_back(detection);
	}
    }

    // TODO
    dets = sort(dets);
    return dets;
  }

  DetectionSet AONN_Model::detect_AStar(const ImRGBZ&im,DetectionFilter filter) const
  {
    // convert the image to a template
    AutoAlignedTemplate imT(im);        

    // extern
    DetectionSet dets;
    return dets;
  }

  // virtual method implementations
  DetectionSet AONN_Model::detect(const ImRGBZ&im,DetectionFilter filter) const
  {   
    Timer timer;
    DetectionSet dets = detect_linear(im,filter);
    log_file << safe_printf("AONN_Model::detect % % milliseconds",im.filename,timer.toc()) << endl;

    return dets;
  }

  void AONN_Model::train(vector<shared_ptr<MetaData>>&training_set,TrainParams train_params)
  {
    // setup the training set
    vector<shared_ptr<MetaData> >  positive_set,negtive_set;    
    split_pos_neg(training_set,positive_set,negtive_set);
    int npos = fromString<int>(g_params.require("NPOS"));
    this->training_set = random_sample<shared_ptr<MetaData> >(positive_set,npos,19860);
    for(auto && datum : training_set)
      log_once(safe_printf("selected positive example: %",datum->get_filename()));

    // extract the templates from said set.
    auto extracted = deformable_depth::extract_templates(this->training_set,train_params);
    allTemplates = extracted.allTemplates;
    parts_per_ex = extracted.parts_per_ex;
    sources      = extracted.sources;
  }

  Mat AONN_Model::show(const string&title)
  {
    return Mat(image_text(string("AONN_Model::show") + title));
  }
}
