/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "NiTE2_Detector.hpp"
#include "stdafx.h"
#include <assert.h>
#include <OpenNI.h>
#include <NiTE.h>
#include <string>
#include <opencv2\opencv.hpp>
#include "util.hpp"
#include <iostream>
#include "BaselineDetection.hpp"

using namespace std;
using namespace cv;
using namespace openni;
using namespace deformable_depth;
using namespace nite;

void NiTE2_Detector::init(string ONI_File_location)
{
	// startup NiTE2 and OpenNI
	assert(openni::OpenNI::initialize() == openni::STATUS_OK);
	assert(device.open(ONI_File_location.c_str()) == openni::STATUS_OK);
	tracker.reset(new nite::HandTracker());
	assert(tracker->create(&device) == nite::STATUS_OK);
	control = device.getPlaybackControl();
	video.create(device,SensorType::SENSOR_DEPTH);
	control->setSpeed(-1);
}

NiTE2_Detector::Frame NiTE2_Detector::captureFrame()
{
	// get the current frame
	Frame frame;
	assert(tracker->readFrame(&frame.handFrame) == nite::STATUS_OK);
	if(!frame.handFrame.isValid())
	{
		cout << "hand frame invalid" << endl;
		frame.valid = false;
		return frame;
	}
	cout << "got hand frame" << endl;
	frame.depthFrame = frame.handFrame.getDepthFrame();
	if(!frame.depthFrame.isValid())
	{
		cout << "depth frame invalid" << endl;
		frame.valid = false;
		return frame;
	}
	cout << "got depth frame" << endl;
	cout << "raw res " << printfpp("%d x %d",(int)frame.depthFrame.getHeight(),(int)frame.depthFrame.getWidth()) << endl;
	uint16_t* p_depth_data = (uint16_t*)frame.depthFrame.getData();
	cout << "depthFrame.getDataSize = " << frame.depthFrame.getDataSize() << endl;
	size_t depth_data_size = sizeof(uint16_t)*frame.depthFrame.getHeight()*frame.depthFrame.getWidth();
	cout << "depth_data_size = " << depth_data_size << endl;

	// convert to opencv for visualization
	frame.cv_depth_uint16 = Mat(frame.depthFrame.getHeight(),frame.depthFrame.getWidth(),CV_16UC1,(void*)p_depth_data,sizeof(uint16_t)*frame.depthFrame.getWidth());
	frame.cv_depth_uint16.convertTo(frame.cv_depth_float,CV_32F);

	return frame;
}

Point NiTE2_Detector::init_frame(Frame&frame,cv::Mat&cv_show)
{
	// get the location from the user
	cout << "clock to initialize" << endl;
	image_safe("vis",cv_show,false); waitKey_safe(10);
	cout << "vis res = " << printfpp("%d x %d",(int)cv_show.rows,(int)cv_show.cols) << endl;
	bool visible;
	Point2i hand_pos = getPt("vis",&visible);
	cv::circle(cv_show,Point(hand_pos.x,hand_pos.y),3,Scalar(255,0,0));

	// send the location to NiTE2
	HandId handId;
	uint16_t z = frame.cv_depth_uint16.at<uint16_t>(hand_pos.y,hand_pos.x);
	tracker->startHandTracking(nite::Point3f(hand_pos.x,hand_pos.y,z),&handId);

	return hand_pos;
}

Point NiTE2_Detector::track_frame(Frame&frame,cv::Mat&cv_show)
{
	Point tracked_location(-1,-1);

	// process the gesture detections
	// because no gestures are detected in our sample video, this is 
	// only partially implemented.
	const nite::Array<GestureData>&gestures = frame.handFrame.getGestures();
	cout << printfpp("found %d gestures",(int)gestures.getSize()) << endl;
	for(int gestureIter = 0; gestureIter < gestures.getSize(); gestureIter++)
	{
		const nite::GestureData&cur_gest = gestures[gestureIter];
		const nite::Point3f& gest_loc = cur_gest.getCurrentPosition();

		float d_x, d_y;
		tracker->convertHandCoordinatesToDepth(gest_loc.x,gest_loc.y,gest_loc.z,&d_x,&d_y);

		cout << printfpp("Gesture %d @ %d %d",gestureIter,(int)d_x,(int)d_y) << endl;
		cout << "ERROR: NO SUPPORT FOR GESTURES AS IS USUALLY ISNT NEEDED!!!" << endl;
		exit(0);
		cv::circle(cv_show,Point(d_x,d_y),3,Scalar(0,255,0));
	}

	// process the hands
	const nite::Array<HandData>&hands = frame.handFrame.getHands();
	cout << printfpp("found %d hands",(int)hands.getSize()) << endl;
	for(int handIter = 0; handIter < hands.getSize(); handIter++)
	{
		const nite::HandData&cur_hand = hands[handIter];
		const nite::Point3f& hand_loc = cur_hand.getPosition();

		float d_x, d_y;
		tracker->convertHandCoordinatesToDepth(hand_loc.x,hand_loc.y,hand_loc.z,&d_x,&d_y);
		tracked_location = Point(d_x,d_y);

		cout << printfpp("Hand %d @ %d %d",handIter,(int)d_x,(int)d_y) << endl;
		cv::circle(cv_show,Point(d_x,d_y),3,Scalar(255,0,0));
	}

	return tracked_location;
}

BaselineDetection NiTE2_Detector::makeDetection(NiTE2_Detector::Frame&frame,cv::Point loc)
{
	// occlusion case
	BaselineDetection det;
	if(loc.x == -1 && loc.y == -1)
		return det;
	loc.x = clamp(0,loc.x,frame.cv_depth_float.cols);
	loc.y = clamp(0,loc.y,frame.cv_depth_float.rows);

	// 58h and 45v from the box.
	CustomCamera oni_camera(58, 45, frame.depthFrame.getWidth(), frame.depthFrame.getHeight());
	float z = frame.cv_depth_float.at<float>(loc.y,loc.x)/10; // openni mm to dd cms
	det.bb = oni_camera.bbForDepth(z, loc.y, loc.x , 15,20);

	// generate the parts for the default pose
	for(int iter = 1; iter <= 5; iter++)
	{
		string partname = printfpp("dist_phalan_%d",iter);
		double r = static_cast<double>(iter-1)/4.0;
		double alpha_x = r*.2 + (1-r)*.8;
		double alpha_y = (iter == 1 || iter == 5)?.55:.7;

		int x = alpha_x * det.bb.tl().x + (1 - alpha_x)*det.bb.br().x;
		int y = alpha_y * det.bb.tl().y + (1 - alpha_y)*det.bb.br().y;

		Size sz(det.bb.width/5,det.bb.height/5);

		BaselineDetection part;
		part.bb = rectFromCenter(Point(x,y),sz);
		det.parts[partname] = part;
	}


	return det;
}

// RAII CTOR, inits everything required for 
// detection....
NiTE2_Detector::NiTE2_Detector(string ONI_File_location, int start_frame)
{
	init(ONI_File_location);
	VideoWriter vis;
	vis.open(ONI_File_location + ".Nite2.avi",CV_FOURCC('F','M','P','4'),15,Size(320,240),true);

	// now start feeding it frames...
	int n_frames = control->getNumberOfFrames(video);
	cout << "n_frames: " << n_frames << endl;
	vector<BaselineDetection> track;
	for(int iter = 0; iter < n_frames; ++iter)
	{
		// capture the next frame
		control->seek(video,iter);
		Frame frame = captureFrame();
		cout << "frame " << iter << endl;	
		cout << "cv_cap res = " << printfpp("%d x %d",(int)frame.cv_depth_float.rows,(int)frame.cv_depth_float.cols) << endl;
		Mat cv_show = imagesc("depth data",frame.cv_depth_float,false,false); 

		BaselineDetection cur_det;
		if(iter == start_frame)
		{
			// initialization mode
			Point location = init_frame(frame,cv_show);
			cur_det = makeDetection(frame,location);
		}
		else if(iter > start_frame)
		{
			// tracking mode
			Point location = track_frame(frame,cv_show);
			cur_det = makeDetection(frame,location);
		}

		cur_det.draw(cv_show);
		cur_det.filename = ONI_File_location + printfpp(":%d:NiTE2",iter);
		track.push_back(cur_det);

		// visualize the results
		image_safe("vis",cv_show,false); waitKey_safe(10);
		vis.write(cv_show);
	}

	// write the detections to disk.
	FileStorage store(ONI_File_location + ".Nite2.yml",FileStorage::WRITE);
	for(int iter = 0; iter < track.size(); iter++)
	{
		store << printfpp("frame%d",(int)iter) << track[iter];
	}
	store.release();

	cout << "NiTE Detector Done" << endl;
}
