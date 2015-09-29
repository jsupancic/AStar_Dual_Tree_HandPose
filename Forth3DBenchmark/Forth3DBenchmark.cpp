// Forth3DBenchmark.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Forth3DTracker.hpp"
#include <sstream>

// The .exe is generated in C:\Users\James\Documents\GitHub\deformable_depth\intelCap\x64\Release
// it needs to be copied into and run in C:\Users\James\Desktop\HandTracker_0.4b_win_x64\Bin

using namespace std;

int _tmain(int argc, wchar_t* argv[])
{
	// convert the image into a .ONI file
	// string prefix = "C:\\Users\\James\\Documents\\"
	string prefix = "C:\\Users\\James\\Dropbox\\data\\depth_video\\"; 
	//string ONI_File = prefix + "test_data11.oni"; bool right_hand = true; int start_frame = 0;
	//string ONI_File = prefix + "sam.oni"; bool right_hand = false; int start_frame = 0;
	//string ONI_File = prefix + "greg.oni"; bool right_hand = false; int start_frame = 0;
	//string ONI_File = prefix + "dennis_test_video1.oni"; bool right_hand = false; int start_frame = 355;
	//string ONI_File = prefix + "xianxin_test_1.oni"; bool right_hand = false; int start_frame = 0;

	//string ONI_File = prefix + "library1.oni"; bool right_hand = false; int start_frame = 0;
	//string ONI_File = prefix + "home_office1.oni"; bool right_hand = false; int start_frame = 0;
	//string ONI_File = prefix + "sequence1.oni"; bool right_hand = true; int start_frame = 50;
	//string ONI_File = prefix + "sequence5.oni"; bool right_hand = true; int start_frame = 200;

	// get arguments
	assert(argc == 4);
	wstring wONI_File(argv[1]);
	string ONI_File(wONI_File.begin(),wONI_File.end());
	bool right_hand = (wstring(L"true") == wstring(argv[2]));
	// 
	wstring wStrNumber(argv[3]);
	istringstream iss(string(wStrNumber.begin(),wStrNumber.end()));
	int start_frame; iss >> start_frame;

	// print arguments
	cout << "oni file = " << ONI_File << endl;

	// feed it to Forth3DHandTracker and get a result
	Forth3DTracker forth_tracker(prefix + ONI_File,right_hand,start_frame);
	return 0;
}

