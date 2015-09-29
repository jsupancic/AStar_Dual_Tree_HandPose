/**
 * Copyright 2013: James Steven Supancic III
 **/
#include "using.hpp"

#pragma once

class Evaluator
{
private:
	PXCSmartPtr<PXCSession>&session;
	UtilRender renderer_depth;
	PXCSmartPtr<PXCAccelerator> accel; 
	PXCSmartPtr<UtilRender> renderer_blob;
	PXCSmartPtr<PXCGesture> gestureDetector;
	PXCGesture::ProfileInfo detector_profile;
	UtilCmdLine cmdline;
	UtilCaptureFile capture;
	//FILE* log;
protected:
	PXCDetection detect(PXCSmartPtr<PXCImage> & loaded_PXC, Mat Z, int iterations = 100);
	vector<PXCDetection> detect_all(PXCSmartPtr<PXCImage> & loaded_PXC, Mat Z, int iterations = 100);
	void vis_detection(Mat&RGB,Mat&Z,Mat&UV,Rect&bb_gt,PXCDetection&detection);
public:
	PXCDetection detect(Mat Z, Mat Conf, Mat UV, Mat RGB, int iterations = 100);
	Evaluator(PXCSmartPtr<PXCSession>&session);
	virtual ~Evaluator();
	void eval(string dir = "Z:\\dropbox\\data\\Yi_Test\\");
	void eval_video(string video_file);
	void eval_bin(string directory);
	void logToFiles(string im_file, string label_file, Rect BB);
};
