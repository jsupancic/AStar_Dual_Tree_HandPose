/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "PXC_CaptureSystem.h"
#include "using.hpp"

FrameCapture PXC_CaptureSystem::next()
{
	PXCSmartArray<PXCImage> images(2);
	PXCSmartSP sp;

	// capture from the camera
	capture->ReadStreamAsync(images,&sp);
	sp->Synchronize();

	// James's code
	PXCCapture::VideoStream::ProfileInfo pcolor;
    capture->QueryVideoStream(0)->QueryProfile(&pcolor);
    PXCCapture::VideoStream::ProfileInfo pdepth;
    capture->QueryVideoStream(1)->QueryProfile(&pdepth);

	printf("Captured %d new images!\n",(int)images.QuerySize());
	printf("D Size = (%d, %d)\n",(int)pdepth.imageInfo.height,(int)pdepth.imageInfo.width);

	// try to convert the captured images into CV Mats and save them.
	PXCImage::ImageData dcolor;
    PXCImage::ImageData ddepth;
	pxcStatus acq_stat1 = images[0]->AcquireAccess(PXCImage::ACCESS_READ,PXCImage::COLOR_FORMAT_RGB24,&dcolor);
	assert(acq_stat1 == PXC_STATUS_NO_ERROR);
	pxcStatus acq_stat2 = images[1]->AcquireAccess(PXCImage::ACCESS_READ,PXCImage::COLOR_FORMAT_DEPTH,&ddepth);
	assert(acq_stat2 == PXC_STATUS_NO_ERROR);
	// RGB Data will always be stored in plane[0]
	// Depth data:
	//		planes[0] stores DEPTHMAP or VERTICES data
	//		planes[1] stores CONFIDENCEMAP data
	//		plnaes[2] stores UVMAP data (used to map depths onto the RGB image ('registration')?
	
	// let's handle the RGB image first.
	Mat Mat_RGB(
		(int)pcolor.imageInfo.height, 
		(int)pcolor.imageInfo.width,
		CV_8UC3,((void*)dcolor.planes[0]) ); //

	// next, grab the Z data...
	Mat Z_z(pdepth.imageInfo.height,pdepth.imageInfo.width,CV_16U,(void*)ddepth.planes[0]);//
	Mat Z_conf(pdepth.imageInfo.height,pdepth.imageInfo.width,CV_16U,(void*)ddepth.planes[1]);//
	Mat Z_uv(pdepth.imageInfo.height,pdepth.imageInfo.width,CV_32FC2,(void*)ddepth.planes[2]);//
	// check memory allocations
	//Mat_RGB = Mat_RGB.clone(); 
	//Z_z = Z_z.clone(); 
	//Z_conf = Z_conf.clone(); 
	//Z_uv = Z_uv.clone();

	FrameCapture result;
	result.rgb = Mat_RGB.clone();
	result.z_conf = Z_conf.clone();
	result.z_z = Z_z.clone();
	result.z_uv = Z_uv.clone();

	images[0]->ReleaseAccess(&dcolor);
	images[1]->ReleaseAccess(&ddepth);

	return result;
}

PXC_CaptureSystem::PXC_CaptureSystem(void)
{
    // Create session
    pxcStatus sts;
    sts=PXCSession_Create(&session);
    if (sts<PXC_STATUS_NO_ERROR) {
        wprintf_s(L"Failed to create a session\n");
        exit(3);
    }

    cmdl = new UtilCmdLine (session);

    capture = new UtilCaptureFile(session, cmdl->m_recordedFile, cmdl->m_bRecord);
    /* set source device search critieria */
    capture->SetFilter(cmdl->m_sdname?cmdl->m_sdname:L"DepthSense Device 325");
    for (std::list<PXCSizeU32>::iterator itrc=cmdl->m_csize.begin();itrc!=cmdl->m_csize.end();itrc++)
        capture->SetFilter(PXCImage::IMAGE_TYPE_COLOR,*itrc);
    for (std::list<PXCSizeU32>::iterator itrd=cmdl->m_dsize.begin();itrd!=cmdl->m_dsize.end();itrd++)
        capture->SetFilter(PXCImage::IMAGE_TYPE_DEPTH,*itrd);
    
    PXCCapture::VideoStream::DataDesc request; 
    memset(&request, 0, sizeof(request));
    request.streams[0].format=PXCImage::COLOR_FORMAT_RGB32; 
    request.streams[1].format=PXCImage::COLOR_FORMAT_DEPTH;
 
    sts = capture->LocateStreams (&request); 
    if (sts<PXC_STATUS_NO_ERROR) {
        // let's search for color only
        request.streams[1].format=0;
        sts = capture->LocateStreams(&request); 
        if (sts<PXC_STATUS_NO_ERROR) {
            wprintf_s(L"Failed to locate video stream(s)\n");
            exit(1);
        }
    }

    for (int idx=0; ;idx++) {
        PXCCapture::VideoStream *stream_v = capture->QueryVideoStream(idx);
        if (stream_v) {
            PXCCapture::Device::StreamInfo sinfo;
            sts = stream_v->QueryStream(&sinfo);
            WCHAR stream_name[256];
            switch (sinfo.imageType) {
            case PXCImage::IMAGE_TYPE_COLOR: 
                swprintf_s<256>(stream_name, L"Stream#%d (Color)", idx);
                renders.push_back(new UtilRender(stream_name));
                streams.push_back (stream_v); 
                break;
            case PXCImage::IMAGE_TYPE_DEPTH: 
                swprintf_s<256>(stream_name, L"Stream#%d (Depth)", idx);
                renders.push_back(new UtilRender(stream_name));
                streams.push_back (stream_v);
                break;
            default:
                break; 
            }
        } else break;
    }

	num_streams = (int)renders.size();
    if (num_streams == 0) {
        wprintf_s(L"Failed to find video stream(s)\n");
        exit(1);
    }

	// setup the GUI
	namedWindow("RGB",CV_WINDOW_NORMAL);
	namedWindow("Z",CV_WINDOW_NORMAL);

	sps = new PXCSmartSPArray(num_streams);
    image = new PXCSmartArray<PXCImage>(num_streams);

    for (int i=0;i<num_streams;i++)  
        streams[i]->ReadStreamAsync (&(*image)[i], &(*sps)[i]); 
}


PXC_CaptureSystem::~PXC_CaptureSystem(void)
{
    // destroy resources
    for (int i=0;i<num_streams;i++) 
	{
        if (renders[i]) delete renders[i];
    }

	delete image;
	delete cmdl;
	delete sps;
	delete capture;
}

void imageeq(const string&title,const Mat&im)
{
	double min, max;
	minMaxLoc(im,&min,&max,NULL,NULL);
	Mat norm = 255*(im-min)/(max-min);
	Mat dpy; norm.convertTo(dpy,CV_8UC1);
	Mat dpyEq; equalizeHist(dpy,dpyEq);
	imshow(title,dpyEq);
	cout << "shown: " << title << endl;
}

void imagesc(const string&title,const Mat&im)
{
	double min, max;
	minMaxLoc(im,&min,&max,NULL,NULL);
	Mat norm = 255*(im-min)/(max-min);
	Mat dpy; norm.convertTo(dpy,CV_8UC1);
	imshow(title,dpy);
}

pxcF32 getProp(PXCCapture::Device*devInfo,PXCCapture::Device::Property prop)
{
	pxcF32 value;
	devInfo->QueryProperty(prop,&value);
	return value;
}

// take note of
// PROPRETY_DEPTH_SATURATED_VALUE and PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE 
// write a captured PXC frame to disk
void DumpPXC(const string&filename,Mat&RGB, Mat&Z, Mat&conf, Mat&UV, PXCCapture::Device*devInfo)
{
	printf("writing %s\n",filename.c_str());
	FileStorage dumpFile(filename,FileStorage::WRITE); 
	try
	{
		printf("test\n");
		static const string STR_RGB("RGB");
		static const string STR_ZMap("ZMap");
		static const string STR_ZConf("ZConf");
		static const string STR_UV("UV");
		cout << "RGB" << "ZMap" << "ZConf" << "UV" << endl;
		//dumpFile.writeObj("RGB",&RGB);
		dumpFile << "RGB" << RGB;
		dumpFile << "ZMap" << Z;
		dumpFile << "ZConf" << conf;
		dumpFile << "UV" << UV;
		dumpFile << "PROPERTYDEPTHLOWCONFIDENCEVALUE" << getProp(devInfo,devInfo->PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE);
		dumpFile << "PROPERTYDEPTHSATURATIONVALUE" << getProp(devInfo,devInfo->PROPERTY_DEPTH_SATURATION_VALUE);
	}
	catch(std::exception e)
	{
		cout << "DumpPXC error: " << e.what(); 
		exit(2);
	}
	dumpFile.release();
}
