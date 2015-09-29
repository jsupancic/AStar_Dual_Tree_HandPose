/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifdef DD_ENABLE_HAND_SYNTH
#include "KTHGrasp.hpp"
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "InverseKinematics.hpp"
#include "util.hpp"
#include "LibHandSynth.hpp"
#include "IKSynther.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  class KTHGraspSynther : public IKGraspSynther
  {
  public:
    KTHGraspSynther(string filename);
  };
  
  KTHGraspSynther::KTHGraspSynther(string filename)
  {
    ifstream ifs(filename);
    int iter = 0;
    while(!ifs.eof())
    {
      cout << printfpp("KTH_grasp_synth iter = %d",iter++) << endl;
      string line; getline(ifs,line);
      istringstream iss(line);
      IK_Grasp_Pose pose;
      int subject, trial, grasp, frame;
      iss >> subject >> trial >> grasp >> frame;
      pose.identifier = printfpp("%d_%d_%d_%d",subject,trial,grasp,frame);
      pose.joint_positions["metacarpals"] = Vec3d(0,0,0);
      for(int kth_finger = 0; kth_finger < 5; kth_finger++)
      {
	double x,y,z, q0, qx, qy, qz;
	iss >> x >> y >> z >> q0 >> qx >> qy >> qz;
	
	string name;
	if(kth_finger == 0) // index
	  name = "finger4joint3tip";
	else if(kth_finger == 1) // thumb
	  name = "finger5joint3tip";
	else if(kth_finger == 2) // middle
	  name = "finger3joint3tip";
	else if(kth_finger == 3) // ring
	  name = "finger2joint3tip";
	else // little
	  name = "finger1joint3tip";
	pose.joint_positions[name] = Vec3d(x,y,z);
      }
      
      grasp_Poses.push_back(pose);
    }    
  };
    
  void kth_grasp_synth()
  {
    // (0) Setup a synther for the retrieved exemplars
    KTHGraspSynther kth_synther("/home/jsupanci/workspace/data/grasp/kth.txt"); 
    
    // (2) Synthesize some training data for Greg.
    for(int iter = 0; iter < 10000; ++iter)
    {
      IKGraspSynther::SynthContext context;
      kth_synther.synth_one_example(context);
    }
  }
}
#endif
