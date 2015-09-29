/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_FORTH3DTRACKER
#define DD_FORTH3DTRACKER

#include <OpenNI.h>
#include <string>
#include <opencv2\opencv.hpp>
#include <HandTrackerLib\HandTracker.hpp>
#include "BaselineDetection.hpp"

using namespace std;
using cv::Mat;
using namespace deformable_depth;

struct ForthFrameInfo
{
	Mat rgb;
	Mat bgr; 
	Mat depth;
	FORTH::DepthPixel* p_depth;
	FORTH::RGBPixel*   p_rgb;
};

class Forth3DTracker
{
public:
	Forth3DTracker(string oni_filename, bool right_hand, int start_frame);

protected:
	// methods
	void setup(string oni_file, bool right_hand);
	BaselineDetection getActiveDetection(ForthFrameInfo&frame);
	void tracker_track(ForthFrameInfo frame);
	void tracker_initialize(ForthFrameInfo frame);
	ForthFrameInfo getFrame(int index,int n_frames);

	// members
	openni::Device device;
	openni::PlaybackControl*control;
	openni::VideoStream color_video, depth_video;
	FORTH::HandTracker *ht;
	int zpd; // zero plane distance
	double zpps; // zero plane pixel size
	// for vis
	Mat their_vis;
	Mat their_vis_depth;

	// friends
	friend static void init_tracker_onMouse(int event, int x, int y, int flags, void*data);
};

#endif
