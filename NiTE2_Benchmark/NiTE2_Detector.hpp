/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef NiTE2_DETECTOR
#define NiTE2_DETECTOR

#include <string>
#include <OpenNI.h>
#include <NiTE.h>
#include <memory>
#include <opencv2\opencv.hpp>
#include <BaselineDetection.hpp>

using namespace nite;
using namespace std;
using namespace openni;
using namespace deformable_depth;

// class which uses DD depth with NiTE2 to detect hands 
// and score the performance of the algorithm.
class NiTE2_Detector
{
public:
	NiTE2_Detector(string ONI_File_location, int start_frame);

protected:
	// types
	struct Frame
	{
		bool valid;
		HandTrackerFrameRef handFrame;
		openni::VideoFrameRef depthFrame;
		cv::Mat cv_depth_float;
		cv::Mat cv_depth_uint16;
	};

	// methods
	void init(string ONI_File_location);
	Frame captureFrame();
	cv::Point init_frame(Frame&frame,cv::Mat&cv_show);
	cv::Point track_frame(Frame&frame,cv::Mat&cv_show);
	BaselineDetection makeDetection(Frame&frame,cv::Point loc);

	// members
	openni::Device device;
	shared_ptr<nite::HandTracker> tracker;
	openni::PlaybackControl*control;
	openni::VideoStream video;
};

#endif
