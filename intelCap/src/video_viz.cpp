/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "video_viz.h"
#include "PXC_CaptureSystem.h"
#include "HandEvaluator.h"

void video_viz()
{
	// init...
	vector<FrameCapture> video;
	PXC_CaptureSystem system;
	PXCSmartPtr<PXCSession> session;
	assert(PXCSession_Create(&session) == PXC_STATUS_NO_ERROR);
	Evaluator ev(session);

	for(int iter = 0; true; iter++)
	{
		printf("Capturing Frame %d\n",iter);
		// capture the next frame.
		FrameCapture frame = system.next();
		video.push_back(frame);

		// show the video of the visualization.
		cv::imshow("RGB",frame.rgb); 
		if(cvWaitKey(20) != -1)
			break;
		//blobs.push_back(det.blob);
	}
	int frame_ct = video.size(); // 30  seconds?

	for(int iter = 0; iter < frame_ct; iter++)
	{
		// dump the frame
		ostringstream oss;
		oss << "cap/video" << iter << ".yml.gz";
		printf("Writing frame %d of %d\n",iter,frame_ct);
		cv::imshow("RGB",video[iter].rgb); cvWaitKey(1);
		DumpPXC(oss.str(),video[iter].rgb,video[iter].z_z, video[iter].z_conf, video[iter].z_uv, system.capture->QueryDevice());

		// dump the detection
		printf("Beginning Detection\n");
		PXCDetection det = ev.detect(video[iter].z_z, video[iter].z_conf, video[iter].z_uv, video[iter].rgb,1);
		printf("Detection Complete\n");
		assert(!det.blob.empty());
		cv::imshow("Blob",det.blob); cvWaitKey(1);
		ostringstream doss;
		doss << "cap/detection" << iter << ".png";
		cv::imwrite(doss.str(),det.blob);
	}
}
