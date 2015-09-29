/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Faces.hpp"
#include "util.hpp"
#include "util_rect.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: FaceDetection
  /// 
  std::shared_ptr<CascadeClassifier> detector;
  mutex monitor;
  map<string/*filename*/,vector<Rect>/*detections*/> cache;
  
  /// SECTION: SimpleFaceDetection
  void SimpleFaceDetector::build_cache(const ImRGBZ& example)
  { /*NOP*/ }

  vector< Rect > SimpleFaceDetector::detect(const ImRGBZ& im)
  {
    unique_lock<mutex> l(monitor);
    
    // run the detector
    vector<Rect> detections;
    detector->detectMultiScale(im.RGB,detections);
    
    return detections;
  }

  SimpleFaceDetector::SimpleFaceDetector()
  {
    unique_lock<mutex> l(monitor);
    static mutex m; lock_guard<mutex> l2(m);

    if(detector == nullptr)
      detector.reset(new CascadeClassifier("./face_cascade.xml"));
    assert(!detector->empty());       
  }
  
  Mat SimpleFaceDetector::detect_and_show(const ImRGBZ& im)
  {
    vector<Rect> detections = detect(im);
    
    return show(im,detections);
  }
  
  DetectionSet
  SimpleFaceDetector::filter(DetectionSet&  src, const ImRGBZ& im, float thresh)
  {
    log_once(printfpp("SimpleFaceDetector::filter thresh = %f",thresh));    
    
    DetectionSet dst;
    vector<Rect> bbs_faces = detect(im);
    
    // compare each hand candidate
    for(int handIter = 0; handIter < src.size(); ++handIter)
    {
      // against each face detection
      vector<Rect> det_one_vec(1,src[handIter]->BB);
      if(rect_max_intersect(det_one_vec,bbs_faces) <= thresh)
	dst.push_back(src[handIter]);
      
//       bool overlap = false;
//       for(int faceIter = 0; faceIter < bbs_faces.size(); ++faceIter)
//       {
// 	Point2d f = rectCenter(bbs_faces[faceIter]);
// 	Point2d h = rectCenter(src[handIter]->BB);
// 	double pix_dist = 
// 	  std::sqrt<double>((f.x-h.x)*(f.x-h.x) + (f.y-h.y)*(f.y-h.y)).real();
// 	if(im.camera.DistPixtoWorldCM(pix_dist,src[handIter]->depth) < 20)
// 	  overlap = true;
//       }
//       if(!overlap)
// 	dst.push_back(src[handIter]);
    }
    
    if(thresh == 1)
      assert(dst.size() == src.size());
    return dst;
  }
  
  Mat SimpleFaceDetector::show(const ImRGBZ& im, vector< Rect >& detections)
  {
    Mat showme = im.RGB.clone();
    for(int iter = 0; iter < detections.size(); iter++)
      rectangle(showme,detections[iter].tl(),detections[iter].br(),Scalar(255,0,0));
    image_safe("face_detections",showme);
    return showme;
  }
  
  /// SECTION: CachingFaceDetection
  
  vector< Rect > CachingFaceDetector::detect(const ImRGBZ&im)
  {
    // this is a critical section due to the caching singleton
    unique_lock<mutex> l(monitor);
    
    // try to retrive from cache
    assert(im.filename != "");
    if(cache.find(im.filename) == cache.end());
    {
      l.release();
      build_cache(im);
      l.lock();
    }
    
    // (1) Retireve from cache
    vector<Rect> detections = cache[im.filename];
    // (2) transform
    for(Rect& face : detections)
      face = rect_Affine(face,im.affine);
        
    // DEBUG
    //show(im,detections);
    
    return detections;
  }
  
  void CachingFaceDetector::build_cache(const ImRGBZ&im)
  {
    // this is a critical section due to the caching singleton
    unique_lock<mutex> l(monitor);    
    
    if(detector == nullptr)
      detector.reset(new CascadeClassifier("./face_cascade.xml"));
    assert(!detector->empty());    
    
    vector<Rect> detections;
    detector->detectMultiScale(im.RGB,detections);
    cache[im.filename] = detections;
  }
  
  Mat CachingFaceDetector::show(const ImRGBZ& im, vector< Rect >& detections)
  {
    Mat showme = im.RGB.clone();
    for(int iter = 0; iter < detections.size(); iter++)
      rectangle(showme,detections[iter].tl(),detections[iter].br(),Scalar(255,0,0));
    image_safe("face_detections",showme);
    return showme;
  }
  
  Mat CachingFaceDetector::detect_and_show(const ImRGBZ&im)
  {
    vector<Rect> detections = detect(im);
    
    return show(im,detections);
  }

  CachingFaceDetector::CachingFaceDetector()
  {
  }
  
  DetectionSet CachingFaceDetector::filter(DetectionSet& src, 
					   const ImRGBZ&im, float thresh)
  {
    DetectionSet dst;
    vector<Rect> bbs_faces = detect(im);
    
    for(int iter = 0; iter < src.size(); iter++)
    {
      vector<Rect> det_one_vec(1,src[iter]->BB);
      if(rect_max_intersect(det_one_vec,bbs_faces) <= thresh)
	dst.push_back(src[iter]);
    }
    
    if(thresh == 1)
      assert(dst.size() == src.size());
    return dst;
  }
}
