/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011-2012 Intel Corporation. All Rights Reserved.

Modified: James Steven Supancic III, Copyright 2012

*******************************************************************************/
#include "using.hpp"
#include "video_viz.h"

/// SECTION: Props
// print out camera properties
void  props()
{
	// setup the PXC environment
	PXCSmartPtr<PXCSession> session;
	assert(PXCSession_Create(&session) == PXC_STATUS_NO_ERROR);
	UtilCmdLine cmdline(session);
	UtilCaptureFile capture(session, cmdline.m_recordedFile, cmdline.m_bRecord);
	
	// setup the capture
	PXCCapture::VideoStream::DataDesc request; 
    memset(&request, 0, sizeof(request));
    request.streams[0].format=PXCImage::COLOR_FORMAT_DEPTH;
    assert(capture.LocateStreams (&request) == PXC_STATUS_NO_ERROR); 

	// get the default device
	PXCCapture::Device*dev = capture.QueryDevice();
	
	// cout
	float lowConf = getProp(dev,dev->PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE);
	cout << "DEPTH_LOW_CONFIDENCE_VALUE" << lowConf << endl;
	cout << "DEPTH_SATURATION_VALUE" << getProp(dev,dev->PROPERTY_DEPTH_SATURATION_VALUE) << endl;
	// fov
	PXCPointF32 fov; dev->QueryPropertyAsPoint(dev->PROPERTY_DEPTH_FIELD_OF_VIEW,&fov);
	PXCPointF32 rgbFov; dev->QueryPropertyAsPoint(dev->PROPERTY_COLOR_FIELD_OF_VIEW,&rgbFov);
	cout << "depth fov: " << fov.x << ", " << fov.y << endl;
	cout << "RGB   fov: " << rgbFov.x << ", " << rgbFov.y << endl;
	// min max range
	::PXCRangeF32 z_range;
	dev->QueryPropertyAsRange(dev->PROPERTY_DEPTH_SENSOR_RANGE,&z_range);
	cout << "z_range: " << z_range.min << " " << z_range.max << endl;
	// focal length
	float depth_focalLength = getProp(dev,dev->PROPERTY_DEPTH_FOCAL_LENGTH);
	cout << "depth_focalLength: " << depth_focalLength << endl;
	float color_focalLength = getProp(dev,dev->PROPERTY_COLOR_FOCAL_LENGTH);
	cout << "color_focalLength: " << color_focalLength << endl;
}

int wmain(int argc, WCHAR* argv[]) 
{
	assert(argc >= 2);
	wstring command(argv[1]);
	wprintf(argv[1]); printf("\n");
	if(command == L"capture")
	{
		int rVal = capture_data();
		return rVal;
	}
	else if(command == L"eval")
	{
		try
		{
			system("dir");
			evaluate_PXC_Hand_detection_old();
		}
		catch(std::exception e)
		{
			system("dir");
			printf("caught exception: %s\n",e.what());
			system("pause");
			return -1;
		}
	}
	else if(command == L"props")
	{
		props();
	}
	else if(command == L"video_viz")
		video_viz();

	printf("DONE\n");
	string v; cin >> v;
	return 0;
}
