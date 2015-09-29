/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "CyberGlove.hpp"
#include "params.hpp"
#include "util_file.hpp"
#include "Log.hpp"
#include "util.hpp"
#include <iostream>
#include <fstream>
#include <boost/filesystem/path.hpp>
#include "LibHandSynth.hpp"

namespace deformable_depth
{
#ifdef DD_ENABLE_HAND_SYNTH
  
  using namespace std;
  
  static vector<double> read(ifstream&ifs, size_t count)
  {
    vector<double> v;
    
    for(int iter = 0; iter < count; ++iter)
    {
      double d;
      ifs >> d;
      v.push_back(d);
    }
    
    assert(v.size() == count);
    return v;
  }
  
  /**
   * -- in the *.dat file each row is a sample containing:
   * a timestamp starting from 0
   * 6 reals from the FoB tracker
   * 9 zeros (disabled sensors)
   * 22 8 bit integers from the Cybergglove
   * 4 16 bit integers from the pressure sensors (only the first is meaningful)
   **/  
  UnigeParamterSequence::UnigeParamterSequence(string filename)
  {
    boost::filesystem::path vid_path(filename);
    boost::filesystem::path directory = vid_path.parent_path();
    string data_path = directory.string() + "/" + 
      allStems(directory.string(),".dat").front() + ".dat";
    
    log_file << "data_path: " << data_path << endl;
    ifstream ifs(data_path);
    while(!ifs.eof())
    {
      // read timestamp
      double timestamp; ifs >> timestamp;
      cout << "parsed time: " << timestamp << endl;
      
      // read 6 reals from the FoB tracker
      fob[timestamp] = read(ifs,6);
      
      // read 9 zeros (disabled sensors)
      disabled[timestamp] = read(ifs,9);
      
      // read 22 8 bit integers from the Cybergglove
      cyberglove[timestamp] = read(ifs,22);
      
      // 4 16 bit integers from the pressure sensors (only the first is meaningful)
      pressure[timestamp] = read(ifs,4);
    }
    ifs.close();
    
    // read the .stmp file
    string stmp_filename = directory.string() + "/" + vid_path.stem().string() + ".stmp";
    log_file << "stmp_filename: " << stmp_filename << endl;
    ifstream stmp_ifs(stmp_filename);
    while(!stmp_ifs.eof())
    {
      // timestamp2frame
      double timestamp; stmp_ifs >> timestamp;
      double frame; stmp_ifs >> frame;
      timestamp2frame[timestamp] = frame;
      frame2timestamp[frame] = timestamp;
      cout << printfpp("(%f, %f)",timestamp,frame) << endl;
    }
    stmp_ifs.close();
    
    // loadup the video
    video.open(filename);
    assert(video.isOpened());
  }
  
  vector< double > UnigeParamterSequence::getCyberGlove(size_t index)
  {
    auto iter = cyberglove.lower_bound(frame2timestamp[index]);
    if(iter == cyberglove.end())
      --iter;
    return iter->second;
  }

  vector< double > UnigeParamterSequence::getFoB(size_t index)
  {
    auto iter = fob.lower_bound(frame2timestamp[index]);
    if(iter == fob.end())
      --iter;
    return iter->second;
  }

  Mat UnigeParamterSequence::getFrame(size_t index)
  {
    Mat frame;
    video.set(CV_CAP_PROP_POS_FRAMES,index);
    video.read(frame);
    return frame;
  }
  
  size_t UnigeParamterSequence::length()
  {
    return video.get(CV_CAP_PROP_FRAME_COUNT);
  }
  
  void cyberglove_reverse_engineer()
  {
    assert(g_params.has_key("VID"));
    UnigeParamterSequence parameterSeq(g_params.get_value("VID"));
    LibHandSynthesizer synth;
    
    for(int iter = 0; iter < parameterSeq.length(); ++iter)
    {
      Mat frame = parameterSeq.getFrame(iter);
      vector<double> cyberglove = parameterSeq.getCyberGlove(iter);
      vector<double> fob = parameterSeq.getFoB(iter);
      image_safe("Frame",frame,false);
      
      //cout << "FOB: " << fob << endl;
      cout << "CYB: " << cyberglove << endl;
      //std::reverse(cyberglove.begin(),cyberglove.end());
      libhand::FullHandPose hand_pose = synth.get_hand_pose();
      // metacarpal joints
      double a = 1;
      double b = -90;
      double c = -1.0;
      auto g = [](double x){return x;};
      auto f = [](double x){return x;};
      hand_pose.bend(0)  = f(.5*a*deg2rad(c*g(cyberglove[16])-b/2)); // pinky
      hand_pose.bend(3)  = f(a*deg2rad(c*g(cyberglove[12])-b));
      hand_pose.bend(6)  = f(a*deg2rad(c*g(cyberglove[8])-b));
      hand_pose.bend(9)  = f(a*deg2rad(c*g(cyberglove[4])-b));
      hand_pose.bend(13) = f(a*deg2rad(-c*g(cyberglove[0])+2*b)); // thumb
      // proximal joints
      hand_pose.bend(1)  = f(.5*a*deg2rad(c*cyberglove[17]-b/2));
      hand_pose.bend(4)  = f(a*deg2rad(c*cyberglove[13]-b));
      hand_pose.bend(7)  = f(a*deg2rad(c*cyberglove[9]-b));
      hand_pose.bend(10)  = f(a*deg2rad(c*cyberglove[5]-b));
      hand_pose.bend(14) = f(a*deg2rad(-c*cyberglove[1]+b));
      // distal joints
      hand_pose.bend(2)  = f(.5*a*deg2rad(c*cyberglove[18]-b/2));
      hand_pose.bend(5)  = f(a*deg2rad(c*cyberglove[14]-b));
      hand_pose.bend(8)  = f(a*deg2rad(c*cyberglove[10]-b));
      hand_pose.bend(11) = f(a*deg2rad(-c*cyberglove[6]+b));
      //hand_pose.bend(14) = deg2rad(-cyberglove[1]+90); // thumb has none!

      // now handle abduction
      //hand_pose.side(0)  = abduction_a*deg2rad(c*cyberglove[19]-abduction_b); // pinky
      //hand_pose.side(3)  = abduction_a*deg2rad(c*cyberglove[15]-abduction_b);
      //hand_pose.side(6)  = abduction_a*deg2rad(c*cyberglove[11]-abduction_b);
      //hand_pose.side(9)  = abduction_a*deg2rad(c*cyberglove[7]-abduction_b);
      //hand_pose.side(13) = abduction_a*deg2rad(c*cyberglove[3]-abduction_b); // thumb      
      
      // 
      auto cam_spec = synth.get_cam_spec();
      cam_spec.theta += .01;
      cam_spec.phi += .01;
      synth.set_hand_pose(hand_pose);
      synth.set_cam_spec(cam_spec);
      image_safe("Rendered (+1)",synth.render_only());
      cam_spec.theta *= -1;
      synth.set_cam_spec(cam_spec);
      image_safe("Rendered (-1)",synth.render_only());
      cam_spec.theta *= -1;
      synth.set_cam_spec(cam_spec);
    }
  }
#else
  void cyberglove_reverse_engineer()
  {
    throw std::runtime_error("Unsupported");
  }
#endif
}
