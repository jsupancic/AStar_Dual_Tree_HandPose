/**
 * Copyright 2013: James Steven Supancic III
 **/

#pragma once

#include "using.hpp"

struct FrameCapture
{
public:
	Mat rgb, z_z, z_conf, z_uv;
};

class PXC_CaptureSystem
{
public:
	PXC_CaptureSystem(void);
	~PXC_CaptureSystem(void);
	FrameCapture next();
public:
	UtilCmdLine*cmdl;
	UtilCaptureFile*capture;
	PXCSmartPtr<PXCSession> session;
	int num_streams;
	std::vector<UtilRender*> renders;    
    std::vector<PXCCapture::VideoStream*> streams; 
	PXCSmartSPArray*sps;
    PXCSmartArray<PXCImage>*image;
};

void imageeq(const string&title,const Mat&im);
void imagesc(const string&title,const Mat&im);
pxcF32 getProp(PXCCapture::Device*devInfo,PXCCapture::Device::Property prop);
void DumpPXC(const string&filename,Mat&RGB, Mat&Z, Mat&conf, Mat&UV, PXCCapture::Device*devInfo);
