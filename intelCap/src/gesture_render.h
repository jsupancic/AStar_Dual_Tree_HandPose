/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2011 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#ifndef __GESTURE_RENDER_H__
#define __GESTURE_RENDER_H__
#include <list>
#include <map>
#include "pxcgesture.h"
#include "util_render.h"

class GestureRender: public UtilRender {
public:

	GestureRender(pxcCHAR *title=0):UtilRender(title) {}
	bool RenderFrame(PXCImage *rgbImage, PXCGesture *detector, PXCGesture::Gesture *gdata, PXCImage *depthImage=0);

protected:

    virtual void DrawMore(HDC hdc, double scale_x, double scale_y);

    struct Line {
        int x0, y0;
        int x1, y1;
    };

    struct Node {
        int x, y;
        int radius;
        COLORREF color;
    };

    struct Gesture {
        int bmp;        // resource
        int count;      // frame count
        pxcUID user;
        PXCGesture::GeoNode::Label label;
    };

    std::list<Line>     m_lines;
    std::map<std::pair<pxcUID,PXCGesture::GeoNode::Label>,Node> m_nodes; 
    std::list<Gesture>  m_gestures;
	int m_openness[2];
	PXCGesture::GeoNode::Openness m_last_openness[2];
};

#endif
