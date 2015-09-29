// James Steven Supancic III, Copyright 2012
#include "using.hpp"
#include "HandEvaluator.h"

void print_cap(int capLabel,int capVal)
{
	if(capLabel >= PXCCapture::Device::PROPERTY_CUSTOMIZED)
	{
		printf("customized ");
		capLabel = capLabel & ~PXCCapture::Device::PROPERTY_CUSTOMIZED;
	}

	switch(capLabel)
	{
	case PXCCapture::Device::PROPERTY_COLOR_EXPOSURE: printf("PROPERTY_COLOR_EXPOSURE\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_BRIGHTNESS: printf("PROPERTY_COLOR_BRIGHTNESS\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_CONTRAST: printf("PROPERTY_COLOR_CONTRAST\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_SATURATION: printf("PROPERTY_COLOR_SATURATION\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_HUE: printf("PROPERTY_COLOR_HUE\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_GAMMA: printf("PROPERTY_COLOR_GAMMA\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_WHITE_BALANCE: printf("PROPERTY_COLOR_WHITE_BALANCE\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_SHARPNESS: printf("PROPERTY_COLOR_SHARPNESS\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_BACK_LIGHT_COMPENSATION: printf("PROPERTY_COLOR_BACK_LIGHT_COMPENSATION\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_GAIN: printf("PROPERTY_COLOR_GAIN\n");break;
	case PXCCapture::Device::PROPERTY_AUDIO_MIX_LEVEL: printf("PROPERTY_AUDIO_MIX_LEVE\n");break;

	case PXCCapture::Device::PROPERTY_DEPTH_SATURATION_VALUE: printf("PROPERTY_DEPTH_SATURATION_VALUE\n");break;
	case PXCCapture::Device::PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE: printf("PROPERTY_DEPTH_LOW_CONFIDENCE_VALUE\n");break;
	case PXCCapture::Device::PROPERTY_DEPTH_CONFIDENCE_THRESHOLD: printf("PROPERTY_DEPTH_CONFIDENCE_THRESHOLD\n");break;
	case PXCCapture::Device::PROPERTY_DEPTH_SMOOTHING: printf("PROPERTY_DEPTH_SMOOTHING\n");break;

            /* Two value properties */
	case PXCCapture::Device::PROPERTY_COLOR_FIELD_OF_VIEW: printf("PROPERTY_COLOR_FIELD_OF_VIEW\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_SENSOR_RANGE: printf("PROPERTY_COLOR_SENSOR_RANGE\n");break;
	case PXCCapture::Device::PROPERTY_COLOR_FOCAL_LENGTH: printf("PROPERTY_COLOR_FOCAL_LENGTH \n");break;
	case PXCCapture::Device::PROPERTY_COLOR_PRINCIPAL_POINT: printf("PROPERTY_COLOR_PRINCIPAL_POINT\n");break;

	case PXCCapture::Device::PROPERTY_DEPTH_FIELD_OF_VIEW: printf("PROPERTY_DEPTH_FIELD_OF_VIEW\n");break;
	case PXCCapture::Device::PROPERTY_DEPTH_SENSOR_RANGE: printf("PROPERTY_DEPTH_SENSOR_RANGE\n");break;
	case PXCCapture::Device::PROPERTY_DEPTH_FOCAL_LENGTH: printf("PROPERTY_DEPTH_FOCAL_LENGTH\n");break;
	case PXCCapture::Device::PROPERTY_DEPTH_PRINCIPAL_POINT: printf("PROPERTY_DEPTH_PRINCIPAL_POINT\n");break;

            /* Three value properties */
	case PXCCapture::Device::PROPERTY_ACCELEROMETER_READING: printf("PROPERTY_ACCELEROMETER_READING\n");break;

            /* Customized properties */
	case PXCCapture::Device::PROPERTY_CUSTOMIZED: printf("PROPERTY_CUSTOMIZED\n"); break;
	case NULL: break;
	default: printf("unknown cap = 0x%x\n", capLabel);
	}
}

// I'm not sure what the correct criteria is, they both look the same to me...
void chooseProfile(PXCGesture::ProfileInfo&detector_profile,PXCGesture&gestureDetector)
{
	for (int iter=0;;iter++) 
	{
        pxcStatus sts=gestureDetector.QueryProfile(iter,&detector_profile);
        if (sts != PXC_STATUS_NO_ERROR) break;
	
		// print the profile
		printf("===== PXC Profile Found =====\n");
		for(int jter = 0; jter < PXCCapture::VideoStream::STREAM_LIMIT; jter++)
		{
			PXCCapture::VideoStream::DataDesc::StreamDesc&desc = detector_profile.inputs.streams[jter];
			if(desc.format != 0)
			{
				switch(desc.format)
				{
				case PXCImage::IMAGE_TYPE_DEPTH:
					printf("image format = PXCImage::IMAGE_TYPE_DEPTH\n"); break;
				default: printf("input format = %d\n",desc.format);
				}
				printf("sizeMax = (%d, %d)\n",desc.sizeMax.height,desc.sizeMax.width);
				printf("sizeMin = (%d, %d)\n",desc.sizeMin.height,desc.sizeMin.width);
				printf("options = 0x%x\n",desc.options);
			}
		}
		for(int jter = 0; jter < PXCCapture::VideoStream::DEVCAP_LIMIT; jter++)
		{
			PXCCapture::VideoStream::DataDesc::DeviceCap&cap = detector_profile.inputs.devCaps[jter];
			if(cap.label != 0)
			{
				print_cap(cap.label,cap.value);
			}
		}

		// TODO compare against the correct profile choice?
		break;
	}

	//detector_profile.blobs = PXCGesture::Blob::Label::LABEL_ANY;
	detector_profile.activationDistance = 500;
}

void print_detections(PXCGesture&gestureDetector)
{
	printf("===== all detections =====\n");
	for(int iter = 0; true; iter++)
	{
		PXCGesture::Gesture data;
		if(gestureDetector.QueryGestureData(0,PXCGesture::GeoNode::LABEL_ANY, iter, &data) == PXC_STATUS_ITEM_UNAVAILABLE)
			break;
		printf("Found a gesture...\n");
	}
}

Rect bbOfNode(PXCGesture::GeoNode&node, Mat&Z)
{
	printf("+bbOfNode\n");
	printf("Node body = 0x%x\n",(int)node.body);
	double im_radius = node.radiusImage;
	if(im_radius == 0)
	{
		printf("bbOfNode: computing size myself\n");
		// this is only set for fingertips, so we must give a default value for hands...
		float x = node.positionImage.x;
		float y = node.positionImage.y;
		float rawZ = node.positionImage.z;
		//float z = node.positionImage.z/100; // m to cm
		printf("bbOfNode: reading out local depth value\n");
		float z = Z.at<unsigned short>(y,x)/10; // so, apparently PXC doesn't have this implemented yet
		printf("test: bbForDepth input = (x,y,z,rawZ) = (%f,%f,%f,%f)\n",x,y,z,rawZ);
		printf("bbOfNode: calling bbForDepth\n");
		Rect bb = DepthCamera().bbForDepth(z,Size(320,240),y,x,25,25);
		printf("test: bb = (%d, %d) to (%d, %d)\n",bb.tl().y,bb.tl().x,bb.br().y,bb.br().x);
		printf("bbOfNode: had to find the detection size myself\n");
		return bb;
	}
	else
	{
		// if the SDK give a size, convert the circle to a BB
		printf("test: im_radius = %f\n",(float)im_radius);
		double area = deformable_depth::params::PI*im_radius*im_radius; 
		double width = sqrt(area);
		printf("test: width = %f\n",(float)width);
		printf("bbOfNode: used Intel's PXC SDK to get the detection size\n");
		return rectFromCenter(Point(node.positionImage.x,node.positionImage.y),Size(width,width));
	}
}

Rect bodyUnion(Rect prev, PXCGesture*gestureDetector, PXCGesture::GeoNode::Label body)
{
	Rect bb = prev;

	PXCGesture::GeoNode::Label parts[9] = 
	{PXCGesture::GeoNode::LABEL_HAND_FINGERTIP,PXCGesture::GeoNode::LABEL_HAND_UPPER,PXCGesture::GeoNode::LABEL_HAND_MIDDLE,PXCGesture::GeoNode::LABEL_HAND_LOWER,
	PXCGesture::GeoNode::LABEL_FINGER_THUMB,PXCGesture::GeoNode::LABEL_FINGER_INDEX,PXCGesture::GeoNode::LABEL_FINGER_MIDDLE,PXCGesture::GeoNode::LABEL_FINGER_RING,
	PXCGesture::GeoNode::LABEL_FINGER_PINKY};

	PXCGesture::GeoNode node;
	for(int iter = 0; iter < 9; iter++)
		if(gestureDetector->QueryNodeData(0,body | parts[iter] ,&node) == PXC_STATUS_NO_ERROR)
		{
			Point p(node.positionImage.x,node.positionImage.y);
			bb = bb | Rect(p,p);
		}

	return bb;
}

class onDetect : public PXCGesture::Gesture::Handler 
{
	public:
    virtual void PXCAPI OnGesture(PXCGesture::Gesture *data) 
	{
		//printf("Gesture detected!\n");
    }
};

class onAction : public PXCGesture::Alert::Handler 
{
public:
    virtual void PXCAPI OnAlert(PXCGesture::Alert *alert) 
	{
		//printf("Action occured\n");
    }
};

Point pointZtoRGB(Point zpos, Mat&UV, Size&RGB)
{
	Vec2f uv = UV.at<Vec2f>(zpos.y,zpos.x);
	int x = clamp<int>(0,uv[0]*RGB.width +  .5,RGB.width-1);
	int y = clamp<int>(0,uv[1]*RGB.height + .5,RGB.height-1);
	return Point(x,y);
}

void evaluate_PXC_Hand_detection_old()
{
	PXCSmartPtr<PXCSession> session;
	assert(PXCSession_Create(&session) == PXC_STATUS_NO_ERROR);
	Evaluator ev(session);
	//ev.eval("Yi_Test");
	//ev.eval("Golnaz_Test");
	//ev.eval("James_Test");
	//ev.eval("Sam_Test");
	//ev.eval("Bailey_Test");
	//ev.eval("Raul_Test");
	//ev.eval("Xiangxin_Test");
	//ev.eval("Vivian_Test");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\test_data11.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\dennis_test_video1.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\greg.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\sam.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\xianxin_test_1.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\library1.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\sequence5.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\sequence1.oni");
	//ev.eval_video("Z:\\dropbox\\data\\depth_video\\sequence1.oni");
	ev.eval_video("Z:\\dropbox\\data\\depth_video\\home_office1.oni");
	//ev.eval_bin("Z:\\dropbox\\data\\videos\\Marga\\");
}
