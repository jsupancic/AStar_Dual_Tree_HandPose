// James Steven Supancic III, Copyright 2012
#include <stdio.h>
#include <conio.h>
#include <windows.h>
#include <wchar.h>
#include <vector>
#include <tchar.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>

#undef RGB
#include "util_render.h"
#include "util_capture_file.h"
#include "util_cmdline.h"
#include "util_rect.hpp"
#include "util_real.hpp"
#include "util_depth.hpp"
#include "util_file.hpp"
#include "PXCSupport.hpp"
#include "params.hpp"

#include "pxcsession.h"
#include "pxcsmartptr.h"
#include "pxccapture.h"
#include "pxcgesture.h"
#include "gesture_render.h"

using namespace deformable_depth;
using namespace cv;
using namespace std;

void evaluate_PXC_Hand_detection_old();
int capture_data();
pxcF32 getProp(PXCCapture::Device*devInfo,PXCCapture::Device::Property prop);