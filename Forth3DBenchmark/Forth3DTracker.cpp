/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "Forth3DTracker.hpp"
#include <OpenNI.h>
#include <iostream>
#include <PS1080.h>
#include <HandTrackerLib\HandTracker.hpp>
#include <opencv2\opencv.hpp>
#include <util.hpp>
#include <util_rect.hpp>

using namespace std;
using namespace openni;
using namespace FORTH;
using namespace cv;

void ensure(bool condition,string message)
{
	if(!condition)
	{
		cout << "ensurence failed: " << message << endl;
		exit(1);
	}
}

void Forth3DTracker::setup(string oni_filename, bool right_hand)
{
	//const char* device_name = openni::ANY_DEVICE;//
	const char* device_name = oni_filename.c_str();
	//cout << "device name: " << device_name << endl;

	// initialize OpenNI2
	ensure(openni::OpenNI::initialize() == openni::STATUS_OK,"initialize");
	ensure(device.open(device_name) == openni::STATUS_OK,"");
	ensure(depth_video.create(device,SensorType::SENSOR_DEPTH) == openni::STATUS_OK,"depth_video.create");
	ensure(color_video.create(device,SensorType::SENSOR_COLOR) == openni::STATUS_OK,"color_video.create");
	ensure(depth_video.start() == openni::STATUS_OK,"depth_video.start");
	ensure(color_video.start() == openni::STATUS_OK,"color_video.start");
	ensure(depth_video.isValid() && color_video.isValid(),"depth and color valid");
	cout << "configuring the controller" << endl;
	control = device.getPlaybackControl();
	//ensure(control != NULL,"control != NULL");
	//ensure(control->isValid(),"control->isValid()");
	if(control)
	{
		control->setSpeed(-1);
		cout << "controller configured" << endl;
	}

	// configure for Forth3DHandTracker
	device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	depth_video.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE,&zpd);
	depth_video.getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE,&zpps);
	ht = HandTracker::getInstance();
	ht->setSensorParameters(640,480,640,480,zpd,zpps);
	ht->initialize(right_hand?RIGHT_HAND:LEFT_HAND,0.9);
}

void Forth3DTracker::tracker_track(ForthFrameInfo frame)
{
	// track
	cout << "track track" << endl;
	result_vector solution = ht->getLastHandPose();
	float score = ht->track(frame.depth.ptr<FORTH::DepthPixel>(0),frame.rgb.ptr<FORTH::RGBPixel>(0),solution);

	// visualize on RGB
	cout << "track vis" << endl;
	their_vis = frame.bgr.clone();
	ht->visualizeHandPose(solution,their_vis.ptr<FORTH::RGBPixel>(0),640,480,false,score);
	imshow("tracking",their_vis);
	// visualize on depth
	their_vis_depth = imageeq("",frame.depth,false,false);
	ht->visualizeHandPose(solution,their_vis_depth.ptr<FORTH::RGBPixel>(0),640,480,false,score);
	imshow("tracking - depth",their_vis_depth);
}

static const string win_name = "forth_init";

struct InitTrackerState
{
	bool done;
	ForthFrameInfo frame;
	FORTH::HandTracker *ht;
};

static void init_tracker_onMouse_nop(int event_code, int x, int y, int flags, void*data)
{}

static void init_tracker_onMouse(int event_code, int x, int y, int flags, void*data)
{
	InitTrackerState* state = static_cast<InitTrackerState*>(data);

	// get the current solution
	cout << "updating init visualization" << endl;
	result_vector solution = state->ht->getLastHandPose();
	if(solution.size() == 0)
	{
		result_vector init_sol = state->ht->getInitialHandPose();
		if(init_sol.size() == 0)
		{
			cout << "no init sol" << endl;
			return;
		}
		state->ht->setHandPose(init_sol);
		cout << "solution not ready" << endl;
		return;
	}
	solution[0] = 1.5*(x - .5*state->frame.depth.cols);
	solution[1] = 1.5*(y - .5*state->frame.depth.rows);
	for(int iter = 0; iter < solution.size(); iter++)
		cout << solution[iter] << " ";
	cout << endl;

	// update the model position based on input
	switch(event_code)
	{
	case CV_EVENT_MOUSEMOVE: // move
		break;
	case CV_EVENT_LBUTTONDOWN: // bigger
		solution[2] -= 10;
		break;
	case CV_EVENT_RBUTTONDOWN: // smaller
		solution[2] += 10;
		break;
	case CV_EVENT_MBUTTONDOWN: // done
		state->done = true;
		break;
	default: // nothing
		break;
	}

	// compute the score for this solution
	cout << "scoring the current pose" << endl;
	float score = .5;
	printf("state.depth %d %d\n",state->frame.depth.rows,state->frame.depth.cols);
	printf("state.color %d %d\n",state->frame.bgr.rows,state->frame.bgr.cols);
	score = state->ht->evaluateHandPose(state->frame.depth.ptr<FORTH::DepthPixel>(0),state->frame.rgb.ptr<FORTH::RGBPixel>(0),solution);
	cout << "score: " << score << endl;

	// and redraw the visualization
	cout << "drawing the current pose" << endl;
	Mat vis = state->frame.bgr.clone();
	state->ht->visualizeHandPose(solution,vis.ptr<FORTH::RGBPixel>(0),640,480,true,score);
	imshow(win_name,vis);
	cout << "current pose drawn" << endl;

	// save the updated hand pose
	state->ht->setHandPose(solution);
}

void Forth3DTracker::tracker_initialize(ForthFrameInfo frame)
{
	// init the shared state
	InitTrackerState state;
	state.frame = frame;
	state.ht = ht;
	state.done = false;

	// show the background
	imshow(win_name,frame.bgr);
	setMouseCallback(win_name,init_tracker_onMouse,&state);

	// get 
	do
	{
		waitKey(10);
	} while(!state.done);

	// disable the callback
	setMouseCallback(win_name,init_tracker_onMouse_nop,NULL);
	destroyWindow(win_name);
}

ForthFrameInfo Forth3DTracker::getFrame(int iter,int n_frames)
{
	// read the frame
	cout << "+frame " << iter << " of " << n_frames << endl;
	if(control)
	{
		control->seek(depth_video,iter);
		//control->seek(color_video,iter);
	}
	VideoFrameRef colorFrame, depthFrame;
	color_video.readFrame(&colorFrame);
	depth_video.readFrame(&depthFrame);
	ensure(depthFrame.isValid() && colorFrame.isValid(),"both frames valid");

	// convert the frame from OpenNI2 format to OpenCV format.
	FORTH::DepthPixel* p_depth = (FORTH::DepthPixel*)depthFrame.getData();
	FORTH::RGBPixel*   p_rgb   = (FORTH::RGBPixel*  )colorFrame.getData();
	cv::Mat cv_depth(depthFrame.getHeight(),depthFrame.getWidth(),CV_16UC1,(void*)p_depth,sizeof(uint16_t)*depthFrame.getWidth());
	cv::Mat cv_rgb  (colorFrame.getHeight(),colorFrame.getWidth(),CV_8UC3 ,(void*)p_rgb  ,3*sizeof(uint8_t)*colorFrame.getWidth());
	cv::resize(cv_depth,cv_depth,Size(640,480));
	cv::resize(cv_rgb,cv_rgb,Size(640,480));
	cv::Mat cv_bgr; cv::cvtColor(cv_rgb,cv_bgr, CV_RGB2BGR);

	// prepare a frame info object to pass
	ForthFrameInfo frame;
	frame.bgr = cv_bgr;
	frame.rgb = cv_rgb;
	frame.depth = cv_depth;
	frame.p_depth = p_depth;
	frame.p_rgb = p_rgb;

	return frame;
}

Mat FORTH_to_CV(const matrix4x4&forth_mat)
{
	Mat cv_mat(4,4,DataType<float>::type);
	for(int iter = 0; iter < 4; iter++)
		for(int jter = 0; jter < 4; jter++)
			cv_mat.at<float>(iter,jter) = forth_mat.data[iter][jter];

	return cv_mat;
}

cv::Mat getPerspectiveTransform(double fovy, double aspect, double z_near, double z_far)
{
	Mat pt(4,4,DataType<float>::type,Scalar::all(0));

	pt.at<float>(0,0) = fovy/aspect;
	pt.at<float>(1,1) = fovy;
	pt.at<float>(2,2) = (z_far + z_near)/(z_near - z_far);
	pt.at<float>(2,3) = (2*z_far*z_near)/(z_near - z_far);
	pt.at<float>(3,2) = -1;

	return pt;
}

BaselineDetection Forth3DTracker::getActiveDetection(ForthFrameInfo&frame)
{
	// what we are building
	BaselineDetection det;

	// the camera intrinsics: 58h and 45v from the box.
	CustomCamera oni_camera(58, 45, frame.bgr.cols, frame.bgr.rows);
	double part_scale = 5;
	Mat visParts = frame.bgr.clone(); resize(visParts,visParts,Size(),part_scale,part_scale);

	// get the 4D homogenous transforms which represent the pose.
	result_vector solution = ht->getLastHandPose();
	matrix_vector cylinderMatrices, sphereMatrices;
	ht->decodeHandPoseToMatrices(solution, cylinderMatrices, sphereMatrices);

	// first, let's convert these to a vector of OpenCV matrices.
	vector<cv::Mat> cv_transforms;
	for(int iter = 0; iter < cylinderMatrices.size(); iter++)
		cv_transforms.push_back(FORTH_to_CV(cylinderMatrices[iter]));
	for(int iter = 0; iter < sphereMatrices.size(); iter++)
		cv_transforms.push_back(FORTH_to_CV(sphereMatrices[iter]));

	// now, we have it!
	Mat perspective_transform = getPerspectiveTransform(45,640.0/480.0,zpd,10000);
	for(int iter = 0; iter < cv_transforms.size(); iter++)
	{
		// render the center of the object to the screen using linear algebra
		Mat pt_0(4,1,DataType<float>::type,Scalar::all(0));
		pt_0.at<float>(3,0) = 1;
		Mat pt_t = perspective_transform*cv_transforms[iter]*pt_0;
		cout << pt_t << endl;
		float x_raw = pt_t.at<float>(0)/pt_t.at<float>(3);
		float y_raw = pt_t.at<float>(1)/pt_t.at<float>(3);
		cout << "x_raw: " << x_raw << " y_raw: " << y_raw << endl;
		double hyper_param = 16; // was 12.5
		float x = hyper_param*(x_raw) + 640/2;  // 10 16
		float y = hyper_param*(y_raw) + 480/2;
		cout << "x: " << x << " y: " << y << endl;

		// get the depth
		if(x <= 5 || x >= frame.depth.cols || y <= 5 || y >= frame.depth.rows)
			continue;
		float z = frame.depth.at<uint16_t>(y,x)/10.0; // openni mm to dd cms

		// emmit the object as a part
		BaselineDetection part;
		part.bb = oni_camera.bbForDepth(z, y, x , 5,5);
		//part.bb = rectFromCenter(Point(x,y),Size(2,2));
		switch(iter)
		{
		case 20:
			det.parts["dist_phalan_1"] = part;
			break;
		case 24:
			det.parts["dist_phalan_2"] = part;
			break;
		case 28:
			det.parts["dist_phalan_3"] = part;
			break;
		case 32:
			det.parts["dist_phalan_4"] = part;
			break;
		case 36:
			det.parts["dist_phalan_5"] = part;
			break;
		default:
			break;
		}

		// merge the part into the entire object BB
		if(part.bb.tl().x > 5 && part.bb.tl().x > 5)
		{
			if(det.bb == Rect())
				det.bb = part.bb;
			else
				det.bb = det.bb | part.bb;
		}

		// draw the numbers so we can identify which numbers correspond to which parts
		cv::putText(visParts,printfpp("%d",iter) ,Point(part_scale*x,part_scale*y),CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(0,0,255),1,CV_AA);
	}

	// print camera peroprties
	cout << "zpd" << zpd << endl;
	cout << "zpps" << zpps << endl;
	//imshow("parts",visParts);

	return det;
}

Forth3DTracker::Forth3DTracker(string oni_filename, bool right_hand, int start_frame)
{
	// setup the library interfaces
	setup(oni_filename, right_hand);
	vector<BaselineDetection> track;
	VideoWriter vis, their_vid, their_vid_depth;
	vis.open(oni_filename + ".FORTH3D.avi",CV_FOURCC('F','M','P','4'),15,Size(640,480),true);
	their_vid.open(oni_filename + ".FORTH3D_THEIRS.avi",CV_FOURCC('F','M','P','4'),15,Size(640,480),true);
	their_vid_depth.open(oni_filename + ".FORTH3D_THEIRS_DEPTH.avi",CV_FOURCC('F','M','P','4'),15,Size(640,480),true);

	// process the video
	int n_frames = control!=NULL?control->getNumberOfFrames(depth_video):999999;
	cout << "n_frames: " << n_frames << endl;
	for(int iter = 0; iter < n_frames; ++iter)
	{
		ForthFrameInfo frame = getFrame(iter,n_frames);

		// get the label for the frame
		BaselineDetection cur_detection;
		if(iter == start_frame)
		{
			// initialization
			cout << "+tracker_initialize" << endl;
			tracker_initialize(frame);
			cout << "-tracker_initialize" << endl;
			cur_detection = getActiveDetection(frame);
		}
		else if(iter > start_frame)
		{
			// tracking
			cout << "+tracker_track" << endl;
			tracker_track(frame);
			cout << "-tracker_track" << endl;
			cur_detection = getActiveDetection(frame);
		}

		// get a detection
		cur_detection.draw(frame.bgr);
		track.push_back(cur_detection);

		// show and store the result
		cout << "+showing loaded frame" << endl;
		imshow("loaded frame",frame.bgr); waitKey(30);
		cout << "-showing loaded frame" << endl;
		vis.write(frame.bgr);
		their_vid.write(their_vis);
		their_vid_depth.write(their_vis_depth);
	}

	// save the result to a file.
	cout << "Writing Track into YAML" << endl;
	FileStorage store(oni_filename + ".FORTH3D.yml",FileStorage::WRITE);
	for(int iter = 0; iter < track.size(); iter++)
	{
		cout << printfpp("storing: frame%d",(int)iter) << endl;
		store << printfpp("frame%d",(int)iter) << track[iter];
	}
	store.release();
}
