/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "stdafx.h"
#include "NiTE2_Detector.hpp"
#include "opencv2\opencv.hpp"
#include "MetaData.hpp"
#include <vector>
#include <string>
#include "util.hpp"

using namespace std;
using namespace deformable_depth;

int _tmain(int argc, _TCHAR* argv[])
{
	cout << "NiTE2 benchmark begins" << endl;

	// convert the image into a .ONI file
	string prefix = "C:\\Users\\James\\Dropbox\\data\\depth_video\\";
	
	// set #1
	//string ONI_File = prefix + "test_data11.oni"; bool right_hand = true;
	//string ONI_File = prefix + "sam.oni"; bool right_hand = false;
	//string ONI_File = prefix + "greg.oni"; bool right_hand = false;
	//string ONI_File = prefix + "dennis_test_video1.oni"; bool right_hand = false; int start_frame = 355;
	//string ONI_File = prefix + "xianxin_test_1.oni"; bool right_hand = false; int start_frame = 0;
	
	// set two
	//string ONI_File = prefix + "library1.oni"; bool right_hand = false; int start_frame = 0;
	//string ONI_File = prefix + "home_office1.oni"; bool right_hand = false; int start_frame = 0;
	//string ONI_File = prefix + "sequence1.oni"; bool right_hand = true; int start_frame = 50;
	string ONI_File = prefix + "sequence5.oni"; bool right_hand = true; int start_frame = 50;

	// feed it to NiTE2 and get a result
	NiTE2_Detector hand_detector(ONI_File, start_frame);
	return 0;
}

