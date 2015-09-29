// James Steven Supancic III, Copyright 2012
#include "using.hpp"
#include "PXC_CaptureSystem.h"

bool capture_mainLoop(
	PXC_CaptureSystem&system,
	int&nwindows, int iter)
{
	// capture the frame data.
	FrameCapture frame = system.next();

	// show results?
	imshow("RGB",frame.rgb); 
	imagesc("Z",frame.z_z); 
	int code = cvWaitKey(1);
	ostringstream savename; savename << "cap" << iter << ".yml.gz";
	if(code == 's')
	{
		string filename = savename.str();
		DumpPXC(filename,frame.rgb, frame.z_z, frame.z_conf, frame.z_uv, system.capture->QueryDevice());
	}

	// store the caputure
	//FileStorage store("filename",FileStorage::WRITE);
	//store.release();

	return 1; /* continue */
}

int capture_data()
{
	PXC_CaptureSystem system;

	// main loop where we show frames
    int nwindows = system.renders.size();
    for (int k = 0; k < (int)system.cmdl->m_nframes && nwindows>0; k++) 
	{
		try
		{
			if(!capture_mainLoop(system,nwindows,k)) break;
		} 
		catch(std::exception e)
		{
			printf("Caught exception! %s\n",e.what());
			exit(1);
		}
    }
    system.sps->SynchronizeEx();

	return 0;
}
