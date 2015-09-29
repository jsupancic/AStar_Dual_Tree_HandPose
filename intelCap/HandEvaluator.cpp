/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "HandEvaluator.h"
#include "using.hpp"
#include <cassert>
#include "util.hpp"
#include "Video.hpp"
#include "ONI_Video.hpp"
#include <opencv2\opencv.hpp>

void PXCtoCV_C1(PXCImage*&im_PXC,Mat&im)
{
	PXCImage::ImageInfo info;
	PXCImage::ImageData data;
	im_PXC->QueryInfo(&info);
	im_PXC->AcquireAccess(PXCImage::ACCESS_READ,PXCImage::COLOR_FORMAT_GRAY,&data);
	im = Mat(info.height,info.width,DataType<unsigned char>::type,data.planes[0]);
	im = im.clone();
	im_PXC->ReleaseAccess(&data);
}

PXCDetection blobalBBOfHand(PXCGesture*gestureDetector, PXCGesture::GeoNode::Label handLabel)
{
	// get the data
	PXCGesture::Blob bdata;
	gestureDetector->QueryBlobData(PXCGesture::Blob::LABEL_SCENE,0,&bdata);
	PXCImage *bimage;
	gestureDetector->QueryBlobImage(PXCGesture::Blob::LABEL_SCENE,0,&bimage);
	Mat blobIm; PXCtoCV_C1(bimage,blobIm);
	printf("blobalBBOfHand: aquired blob data from PXC\n");

	// what value indicates hand pixels?
	int targetVal = (handLabel == PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT)?bdata.labelLeftHand:bdata.labelRightHand;

	PXCDetection detection;
	detection.bb = rectOfBlob(blobIm,targetVal);
	detection.blob = (blobIm == targetVal);
	assert(!detection.blob.empty());
	return detection;
}

// extract the given finger tip, if it was found.
PXCDetection get_one_finger_tip(PXCGesture*gestureDetector,PXCGesture::GeoNode::Label finger_label,string name)
{
	pxcStatus status;
	PXCGesture::GeoNode finger_tip;

	// determine if the finger exists
	if ((status = gestureDetector->QueryNodeData(0, finger_label,&finger_tip)) == PXC_STATUS_NO_ERROR)
	{
		printf("finger %s found!\n",name.c_str());

		// sinse it exists, try to find the point/location.
		Point center(finger_tip.positionImage.x,finger_tip.positionImage.y);

		// now convert to rect and return based on wrold coordinate sizes.
		float x = finger_tip.positionImage.x;
		float y = finger_tip.positionImage.y;
		float z = 100*finger_tip.positionWorld.y; // convert m to cm 
		printf("input params: %f %f %f\n",x,y,z);
		Rect_<double> bb = deformable_depth::RGBCamera().bbForDepth(z, y, x, 2.5, 2.5);
		printf("finger BB: %d %d %d %d\n",bb.x,bb.y,bb.width,bb.height);
		PXCDetection detection;
		detection.bb = bb;
		return detection;
	}
	else
	{
		printf("finger %s NOT found!\n",name.c_str());
		return PXCDetection(); // not found, empty state
	}
}

// extract the fingers for given hand detection
map<string,PXCDetection> get_finger_detections(PXCGesture*gestureDetector,PXCGesture::GeoNode::Label hand_label)
{
	map<string,PXCDetection> part_detections;

	// Pinky
	part_detections["dist_phalan_1"] = get_one_finger_tip(gestureDetector,hand_label | PXCGesture::GeoNode::LABEL_FINGER_PINKY,"dist_phalan_1");
	// Ring
	part_detections["dist_phalan_2"] = get_one_finger_tip(gestureDetector,hand_label | PXCGesture::GeoNode::LABEL_FINGER_RING,"dist_phalan_2");
	// MIddle
	part_detections["dist_phalan_3"] = get_one_finger_tip(gestureDetector,hand_label | PXCGesture::GeoNode::LABEL_FINGER_MIDDLE,"dist_phalan_3");
	// Index
	part_detections["dist_phalan_4"] = get_one_finger_tip(gestureDetector,hand_label | PXCGesture::GeoNode::LABEL_FINGER_INDEX,"dist_phalan_4");
	// Thumb
	part_detections["dist_phalan_5"] = get_one_finger_tip(gestureDetector,hand_label | PXCGesture::GeoNode::LABEL_FINGER_THUMB,"dist_phalan_5");

	return part_detections;
}

vector<PXCDetection> get_detections(PXCGesture*gestureDetector,PXCGesture::ProfileInfo&profile,Mat&Z)
{
	// display detected hands
	//print_detections(*gestureDetector);
	PXCGesture::GeoNode leftHand, rightHand;
	PXCDetection leftBB, rightBB;
	leftBB.blob = Mat(Z.rows,Z.cols,DataType<uchar>::type,Scalar::all(0));
	rightBB.blob = Mat(Z.rows,Z.cols,DataType<uchar>::type,Scalar::all(0));
	pxcStatus get_hand_status_left, get_hand_status_right;
	PXCGesture::GeoNode::Label 
		leftLabel = PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT,
		rightLabel = PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT;
	// two possible return values: PXC_STATUS_ITEM_UNAVAILABLE and PXC_STATUS_NO_ERROR
	if ((get_hand_status_left = gestureDetector->QueryNodeData(0,leftLabel ,&leftHand)) == PXC_STATUS_NO_ERROR)
	{
		printf("test: found the left hand\n");
		//leftBB = bbOfHand(gestureDetector,leftLabel); //
		//leftBB = bodyUnion(bbOfNode(leftHand,Z),gestureDetector,leftLabel);
		leftBB = blobalBBOfHand(gestureDetector, leftLabel);
		leftBB.parts = get_finger_detections(gestureDetector,leftLabel);
		leftBB.resp = leftHand.confidence;
		leftBB.notes = "left_hand";
	}
	else
	{
		printf("test: didn't find left hand\n");
		assert(get_hand_status_left == PXC_STATUS_ITEM_UNAVAILABLE);
	}
	if ((get_hand_status_right = gestureDetector->QueryNodeData(0,rightLabel,&rightHand)) == PXC_STATUS_NO_ERROR)
	{
		printf("test: found the right hand\n");
		//rightBB = bbOfHand(gestureDetector,rightLabel);//
		//rightBB = bodyUnion(bbOfNode(rightHand,Z),gestureDetector,rightLabel);
		rightBB = blobalBBOfHand(gestureDetector,rightLabel);
		rightBB.parts = get_finger_detections(gestureDetector,rightLabel);
		rightBB.resp = rightHand.confidence;
		rightBB.notes = "right_hand";
	}
	else
	{
		printf("test: didn't find right hand\n");
		assert(get_hand_status_right == PXC_STATUS_ITEM_UNAVAILABLE);
	}

	printf("test: detections enumerated\n");
	vector<PXCDetection> detections;
	if(leftBB.bb != Rect())
		detections.push_back(leftBB);
	if(rightBB.bb != Rect())
		detections.push_back(rightBB);
	return detections;
}

PXCDetection choose_best_detection(vector<PXCDetection>&all_dets)
{
	PXCDetection*selected_bb = NULL;
	// choose which BB to return.
	double best_score = -inf;
	for(int iter = 0; iter < all_dets.size(); ++iter)
		if(all_dets[iter].resp > best_score)
		{
			best_score = all_dets[iter].resp;
			selected_bb = &all_dets[iter];
		}

	if(selected_bb)
		return *selected_bb;
	else
		return PXCDetection();
}

// returns the detection BB in *depth image* coordinates.
PXCDetection get_detection(PXCGesture*gestureDetector,PXCGesture::ProfileInfo&profile,Mat&Z)
{
	vector<PXCDetection> all_dets = get_detections(gestureDetector,profile,Z);
	return choose_best_detection(all_dets);
}

void show_blobs(PXCGesture*gestureDetector,UtilRender*renderer_blob,PXCGesture::ProfileInfo&profile)
{
	// display the blog image
	if (profile.blobs)
	{
		PXCSmartPtr<PXCImage> blobImage;
		assert(gestureDetector->QueryBlobImage(PXCGesture::Blob::LABEL_SCENE,0,&blobImage) == PXC_STATUS_NO_ERROR);
		renderer_blob->RenderFrame(blobImage);
	}
}

struct ExampleData
{
	Rect BB;
	Mat Z, Conf, UV, RGB;
	PXCSmartPtr<PXCImage> loaded_PXC;
};

Mat fixUV(Mat UV)
{
	Mat invalid(UV.rows,UV.cols,DataType<unsigned char>::type,Scalar::all(0));
	for(int rIter = 0; rIter < UV.rows; rIter++)
		for(int cIter = 0; cIter < UV.cols; cIter++)
		{
			float u = UV.at<Vec2f>(rIter,cIter)[0];
			float v = UV.at<Vec2f>(rIter,cIter)[1];

			if(u <= 0 || v <= 0)
			{
				//printf("missing UV at (%d, %d)\n",rIter,cIter);
				invalid.at<unsigned char>(rIter,cIter) = 1;
			}
		}

	return fillHoles<Vec2f>(UV,invalid);
}

void CVtoPXC(Mat&Z,Mat&Conf,Mat&UV,PXCAccelerator*accel,PXCSmartPtr<PXCImage>&im_PXC)
{
	bool intel_hack_cheat = false;
	PXCImage::ImageInfo imInfo; 
	imInfo.width = Z.cols; imInfo.height = Z.rows;
	imInfo.format = PXCImage::COLOR_FORMAT_DEPTH;
	accel->CreateImage(&imInfo,&im_PXC);
	PXCImage::ImageData imData;
	im_PXC->AcquireAccess(PXCImage::ACCESS_WRITE,&imData);
	// convert the Depth (Z) Map
	if(!Z.empty())
	{
		// we may need to first convert to PXC depth format.
		if(Z.type() == DataType<float>::type)
		{
			Z *= 10;
			Z.convertTo(Z,DataType<uint16_t>::type);
		}

		// cheat to help Intel's PXC on Video
		if(intel_hack_cheat)
		{
			Rect roi(Point(0,0),Point(.2*Z.cols,Z.rows-1));
			Mat(roi.height,roi.width,DataType<uint16_t>::type,1000).copyTo(Z(roi));
		}

		memcpy(imData.planes[0],Z.ptr(),Z.size().area()*Z.elemSize());
	}
	// convert the confidence map
	if(!Conf.empty())
		memcpy(imData.planes[1],Conf.ptr(),Conf.size().area()*Conf.elemSize());
	else
	{
		Mat Conf_Default(Z.rows,Z.cols,DataType<uint16_t>::type,5000);
		
		if(intel_hack_cheat)
		{
			// cheat to help Intel's PXC on video
			//cv::rectangle(Conf_Default,Point(0,0),Point(Z.rows-1,.2*Z.cols),Scalar::all(32002),CV_FILLED);
			Rect roi(Point(0,0),Point(.2*Z.cols,Z.rows-1));
			Mat(roi.height,roi.width,DataType<uint16_t>::type,32002).copyTo(Conf_Default(roi));
			//cout << Conf_Default << endl;
			imshow("Conf_Default",deformable_depth::imagesc("Conf_Default",Conf_Default));
		}

		memcpy(imData.planes[1],Conf_Default.ptr(),Conf_Default.size().area()*Conf_Default.elemSize());
	}
	// convert the UV map
	if(!UV.empty())
		memcpy(imData.planes[2],UV.ptr(),UV.size().area()*UV.elemSize());
	im_PXC->ReleaseAccess(&imData);
	printf("test: loaded and drew example input\n");
}

// loadExample(dir + test_stems[testIter] + ".gz",dir + test_stems[testIter] + ".labels.yml");
void loadExample(
	PXCAccelerator*accel,
	string image_file,
	string label_file,
	ExampleData&ex)
{
	// load a random sample
	// load the image data
	deformable_depth::loadPCX_RawZ(image_file,ex.Z,ex.Conf,ex.UV);
	deformable_depth::loadPCX_RGB(image_file,ex.RGB);
	//convert to PXCImage and display
	CVtoPXC(ex.Z,ex.Conf,ex.UV,accel,ex.loaded_PXC);

	// try loading the RGB BB directly
	ex.BB = loadBB_PCX(label_file);
	if(ex.BB == Rect())
	{
		// if that fails, try loading the Z BB and mapping it to the RGB space
		ex.BB = loadBB_PCX(label_file,"HandBB_ZCoords");
		ex.BB = rectZtoRGB(ex.BB, ex.UV, ex.RGB.size());
	}
}

Evaluator::Evaluator(PXCSmartPtr<PXCSession>&session) : 
	renderer_depth(L"Loaded Depth Data"),
	renderer_blob(new GestureRender(L"Blobs")),
	session(session),
	cmdline(session),
	capture(session, cmdline.m_recordedFile, cmdline.m_bRecord)
	//log(fopen("PXC_Fired.txt","a"))
{
	// status variable
	pxcStatus sts;

	// to render raw PXC data...
	// create a session to create the accleerator
	// configuration from command line?
	// to create/allocate implementation images
	assert(session->CreateAccelerator(&accel) == PXC_STATUS_NO_ERROR);
	// to detect hands?
	assert(session->CreateImpl(cmdline.m_iuid, PXCGesture::CUID, (void**)&gestureDetector) == PXC_STATUS_NO_ERROR);
	// configure the profile
	//chooseProfile(detector_profile,*gestureDetector);
	//gestureDetector->SetProfile(&detector_profile);
	// Choose the configuration which matches our camera
	for (int i=0;;i++) 
	{
		sts = gestureDetector->QueryProfile(i,&detector_profile);
		if (sts<PXC_STATUS_NO_ERROR) break;
		sts = capture.LocateStreams(&detector_profile.inputs);
		if (sts>=PXC_STATUS_NO_ERROR) break;
	}
	detector_profile.activationDistance = 500;
	gestureDetector->SetProfile(&detector_profile);
	printf("initialized PXC SDK\n");
	// subscribe to detections
	//PXCSmartPtr<onDetect> ghandler(new onDetect);
	//gestureDetector->SubscribeGesture(100,ghandler);
	//PXCSmartPtr<onAction> ahandler(new onAction);
	//sts=gestureDetector->SubscribeAlert(ahandler);
}

Evaluator::~Evaluator()
{
	//fclose(log);
}

vector<PXCDetection> Evaluator::detect_all(PXCSmartPtr<PXCImage> & loaded_PXC, Mat Z, int iterations)
{
	// synchronizeation point
	PXCSmartSPArray sps(2);

	PXCCapture::VideoStream::Images gesture_images;
	//capture.MapImages(0,images,gesture_images);
	for(int streamIter = 0; streamIter < PXCCapture::VideoStream::STREAM_LIMIT; streamIter++)
		gesture_images[streamIter] = loaded_PXC;//im_PXC; // align images in array to match gesture model's expectations.

	printf("test: Calling gestureDetector->ProcessImageAsync\n");
	for(int iter = 0; iter < iterations; iter++)
	{
		assert(gestureDetector->ProcessImageAsync(gesture_images,&sps[1]) == PXC_STATUS_NO_ERROR);
		sps.SynchronizeEx();
		assert(sps[1]->Synchronize(0) == PXC_STATUS_NO_ERROR);
	}
	printf("test: detector finished\n");
	show_blobs(gestureDetector,renderer_blob,detector_profile);
	vector<PXCDetection> detection = get_detections(gestureDetector,detector_profile,Z);
	//
	return detection;
}

PXCDetection Evaluator::detect(PXCSmartPtr<PXCImage> & loaded_PXC, Mat Z, int iterations)
{
	vector<PXCDetection> detections = Evaluator::detect_all(loaded_PXC, Z, iterations);
	if(detections.size() > 0)
	{
		PXCDetection detection = choose_best_detection(detections);
		assert(!detection.blob.empty());
		return detection;
	}
	else
		return PXCDetection();
}

PXCDetection Evaluator::detect(Mat Z, Mat Conf, Mat UV, Mat RGB, int iterations)
{
	PXCSmartPtr<PXCImage> im_PXC;
	CVtoPXC(Z,Conf,UV,accel,im_PXC);
	PXCDetection detection = detect(im_PXC,Z,iterations);
	assert(!detection.blob.empty());
	return detection;
}

void Evaluator::vis_detection(Mat&RGB,Mat&Z,Mat&UV,Rect&bb_gt,PXCDetection&detection)
{
	rectangle(RGB,bb_gt.tl(),bb_gt.br(),Scalar(0,255,0)); // show ground truth
	if(detection.bb != Rect())
	{
		cout << "FOUND A DETECTION" << endl;
		// fill holes in the UV image
		if(!UV.empty())
			UV = fixUV(UV);

		// display the root BB
		rectangle(Z,detection.bb.tl(),detection.bb.br(),Scalar::all(0));
		if(!UV.empty())
			detection.bb = rectZtoRGB(detection.bb,UV,RGB.size());
		printf("mapped detection to (%d, %d) to (%d, %d)\n",
			detection.bb.tl().y,detection.bb.tl().x,detection.bb.br().y,detection.bb.br().x);
		rectangle(RGB,detection.bb.tl(),detection.bb.br(),Scalar(0,0,255));

		// display the parts
		for(map<string,PXCDetection>::iterator iter = detection.parts.begin();
			iter != detection.parts.end(); iter++)
		{
			PXCDetection& part_det = iter->second;
			if(!UV.empty())
				part_det.bb = rectZtoRGB(part_det.bb,UV,RGB.size());
			rectangle(RGB,part_det.bb.tl(),part_det.bb.br(),Scalar(255,0,0));
		}
	}
	else
		cout << "NO DETECTION FOUND" << endl;
	imshow("PXC Detection - Z",Z);
	imshow("PXC Detection - RGB",RGB); 
	//cvWaitKey(10); // show the bb then continue...
}

void Evaluator::eval_video(string video_file)
{
	cout << "+load video" << endl;
	deformable_depth::ONI_Video video(video_file);
	int n_frames = video.getNumberOfFrames();
	cout << "-load video" << endl;
	VideoWriter viz(video_file + ".PXC.avi",CV_FOURCC('F','M','P','4'),15,Size(320,240),true);

	vector<BaselineDetection> track;
	for(int iter = 0; iter < n_frames; ++iter)
	{
		// 
		printf("Frame %d of %d\n",(int)iter,(int)n_frames);

		// load the current frame
		shared_ptr<MetaData> metadata = video.getFrame(iter,true);
		shared_ptr<ImRGBZ> im = metadata->load_im();
		PXCSmartPtr<PXCImage> im_PXC;
		CVtoPXC(im->Z,Mat(),Mat(), accel, im_PXC);

		// Debug, render the frame
		renderer_depth.RenderFrame(im_PXC); 

		// perform gesture recognition
		PXCDetection detection = detect(im_PXC,im->Z,1/*only one needed for video*/);
		track.push_back(detection);

		// show the result for now
		vis_detection(im->RGB,im->Z,Mat(),Rect(),detection);
		viz.write(im->RGB);
	}
	viz.release();

	// write the result.
	FileStorage store(video_file + ".PXC.yml",FileStorage::WRITE);
	for(int iter = 0; iter < track.size(); iter++)
		store << printfpp("frame%d",(int)iter) << track[iter];
	store.release();

	cout << "PXC Video Benchmark Complete!" << endl;
}

void Evaluator::eval_bin(string directory)
{
	vector<string> test_stems = allStems(directory,".exr");

	cv::FileStorage detection_store("PXC_Detections.yml",cv::FileStorage::WRITE);
	for(int testIter = 1; testIter <= test_stems.size(); ++testIter)
	{
		// load the binary file
		string stem = printfpp("frame%d.mat",testIter); // test_stems[testIter]
		string filename = directory + stem + ".exr";
		cout << "reading: " << filename << endl;
		Mat frame_depth = imread(filename,CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
		frame_depth = frame_depth.t();

		// visualize it
		Mat frame_depth_vis;
		frame_depth.convertTo(frame_depth_vis,DataType<float>::type);
		frame_depth_vis /= 10.0;
		cout << "channels: " << frame_depth_vis.channels() << endl;
		Mat RGB = imageeq("imageeq(frame_depth)",frame_depth_vis,true,false);
		//cout << "im: " << frame_depth << endl;
		waitKey(10);

		// convert to pxc format
		PXCSmartPtr<PXCImage> im_PXC;
		CVtoPXC(frame_depth_vis,Mat(),Mat(), accel, im_PXC);

		// visualize with pxc
		renderer_depth.RenderFrame(im_PXC); 

		// evaluate
		vector<PXCDetection> detections = detect_all(im_PXC,frame_depth_vis,1/*only one needed for video*/);

		// display detection results
		for(int iter = 0; iter < detections.size(); ++iter)
			vis_detection(RGB,frame_depth_vis,Mat(),Rect(),detections[iter]);
		waitKey(10);

		// store evalaution
		for(int iter = 0; iter < detections.size(); ++iter)
			detection_store << printfpp("frame_%d_%d",(int)testIter+1,iter) << detections[iter];
	}

	detection_store.release();
}

void Evaluator::eval(string person)
{
	string dir = "Z:\\dropbox\\data\\" + person + "\\";
	vector<string> test_stems = allStems(dir,".gz");

	Scores score;
	cv::FileStorage detection_store(printfpp("PXC_%s_Detections.yml",person.c_str()),cv::FileStorage::WRITE);
	for(int testIter = 0/*DEBUG*/; testIter < test_stems.size(); testIter++)
	//while(true)
	{
		printf("******** Test Iteration = %d **********\n",testIter);
		// get the example
		string im_file = dir + test_stems[testIter] + ".gz", label_file = dir + test_stems[testIter] + ".labels.yml";
		ExampleData ex; loadExample(accel,	im_file,label_file,ex);
		//if(!PXCFile_PXCFired(label_file))
		//	continue;
		
		// DEBUG
		//im_PXC->CopyData(loaded_PXC);
		//loaded_PXC->SetTimeStamp(im_PXC->QueryTimeStamp());
		renderer_depth.RenderFrame(ex.loaded_PXC); 

		// perform gesture recognition
		PXCDetection detection = detect(ex.loaded_PXC,ex.Z);

		// save the detection for later analysis
		ostringstream oss; oss << "detection" << testIter;
		detection.filename = dir + test_stems[testIter];
		detection_store << oss.str() << detection;

		// display the detection results
		vis_detection(ex.RGB,ex.Z,ex.UV,ex.BB,detection);

		// store the blob
		// cv::imwrite(test_stems[testIter] + ".png",detection.blob);

		// evaluate the quality of the result.
		// update tp = 0, fp = 0, fn = 0;
		rectScore(ex.BB, detection.bb, inf, score);
		printf("p = %f r = %f\n",score.p(-inf),score.r(-inf));
		logToFiles(im_file,label_file,detection.bb);

		// debug
		// cvWaitKey(0);
	}

	detection_store.release();
}

void Evaluator::logToFiles(string im_file, string label_file, Rect BB)
{
	// check if we already labeled this information
	FileStorage store;
	store.open(label_file,FileStorage::READ);
	bool unknownPXCFired = store["PXCFired"].empty();
	store.release();
		
	if(unknownPXCFired)
	{
		store.open(label_file,FileStorage::APPEND);
		store << "PXCFired" << (BB != Rect());
		store.release();
	}
}

