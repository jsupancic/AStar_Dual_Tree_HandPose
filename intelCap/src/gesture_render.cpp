/*******************************************************************************

INTEL CORPORATION PROPRIETARY INFORMATION
This software is supplied under the terms of a license agreement or nondisclosure
agreement with Intel Corporation and may not be copied or disclosed except in
accordance with the terms of that agreement
Copyright(c) 2012 Intel Corporation. All Rights Reserved.

*******************************************************************************/
#include <windows.h>
#include <resource.h>
#include "gesture_render.h"

#define GESTURE_BOX_SIZE 40
#define GESTURE_COUNT    30
#define COLOR_RED        RGB(255,0,0)
#define COLOR_GREEN      RGB(0,255,0)
#define COLOR_BLUE       RGB(0,0,255)
#define COLOR_YELLOW     RGB(255,255,0)
#define OPENNESS_WIDTH   4

static struct {
    PXCGesture::GeoNode::Label label;
    COLORREF color;
    int raduis;
} skeletons[]={
            { PXCGesture::GeoNode::LABEL_BODY_ELBOW_LEFT, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_ELBOW_RIGHT, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT, COLOR_RED, 8 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT, COLOR_BLUE, 8 },

			{ PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_HAND_FINGERTIP, COLOR_YELLOW, 2 },
			{ PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_HAND_UPPER, COLOR_BLUE, 2 },
			{ PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_HAND_MIDDLE, COLOR_BLUE, 5 },
			{ PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_HAND_LOWER, COLOR_BLUE, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_THUMB, COLOR_GREEN, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_INDEX, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_MIDDLE, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_RING, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_PINKY, COLOR_RED, 2 },

			{ PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_HAND_FINGERTIP, COLOR_YELLOW, 2 },
			{ PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_HAND_UPPER, COLOR_GREEN, 2 },
			{ PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_HAND_MIDDLE, COLOR_GREEN, 5 },
			{ PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_HAND_LOWER, COLOR_GREEN, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_THUMB, COLOR_BLUE, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_INDEX, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_MIDDLE, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_RING, COLOR_RED, 2 },
            { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_PINKY, COLOR_RED, 2 },
};

static struct {
    PXCGesture::GeoNode::Label  from;
    PXCGesture::GeoNode::Label  to;
} skeletonLines[] = 
{
    { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT, PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_THUMB },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT, PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_INDEX },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT, PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_MIDDLE },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT, PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_RING },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT, PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT|PXCGesture::GeoNode::LABEL_FINGER_PINKY },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT, PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_THUMB },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT, PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_INDEX },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT, PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_MIDDLE },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT, PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_RING },
    { PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT, PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT|PXCGesture::GeoNode::LABEL_FINGER_PINKY },
};

static struct {
    PXCGesture::Gesture::Label label;
    int bmp;
} gesturePics[]={
    { PXCGesture::Gesture::LABEL_POSE_THUMB_UP,     IDB_POSE_THUMB_UP   },
	{ PXCGesture::Gesture::LABEL_POSE_THUMB_DOWN,	IDB_POSE_THUMB_DOWN },
	{ PXCGesture::Gesture::LABEL_POSE_PEACE,        IDB_POSE_PEACE      },
	{ PXCGesture::Gesture::LABEL_HAND_WAVE,			IDB_HAND_WAVE		},
    { PXCGesture::Gesture::LABEL_NAV_SWIPE_LEFT,    IDB_NAV_SWIPE_LEFT  },
    { PXCGesture::Gesture::LABEL_NAV_SWIPE_RIGHT,   IDB_NAV_SWIPE_RIGHT },
};

static void MapXY(float &x, float &y, PXCImage::ImageData *data, PXCImage::ImageInfo *depthInfo, PXCImage::ImageInfo *rgbInfo) {
    if (data->planes[2]) {
        if (x>=0 && y>=0 &&  x<depthInfo->width && y<depthInfo->height) {
          int index=((int)y)*depthInfo->width+x;
            int index2=2*index;
            x=((float*)data->planes[2])[index2]*rgbInfo->width;
            y=((float*)data->planes[2])[index2+1]*rgbInfo->height;
        }
    } else {
        if (depthInfo->width>0 && depthInfo->height>0) {
            x=x/depthInfo->width*rgbInfo->width;
            y=y/depthInfo->height*rgbInfo->height;
        }
    }
}

bool GestureRender::RenderFrame(PXCImage *rgbImage, PXCGesture *detector, PXCGesture::Gesture *gdata, PXCImage *depthImage) {
    m_lines.clear();
    m_nodes.clear();
    if (detector) {
        PXCImage::ImageData depthData;
        memset(&depthData,0,sizeof(depthData));

        PXCImage::ImageInfo rgbInfo, depthInfo={0,0,0};
        rgbImage->QueryInfo(&rgbInfo);
        if (depthImage) {
            depthImage->QueryInfo(&depthInfo);
            pxcStatus sts=depthImage->AcquireAccess(PXCImage::ACCESS_READ,&depthData);
        }

        for (int u=0;;u++) {
            pxcUID user=0;
            pxcStatus sts= detector->QueryUser(u,&user);
            if (sts<PXC_STATUS_NO_ERROR) break;

            for (int i = 0; i < sizeof(skeletonLines)/sizeof(skeletonLines[0]); i++) {
                PXCGesture::GeoNode dataFrom;
                if (detector->QueryNodeData(user,skeletonLines[i].from,&dataFrom)<PXC_STATUS_NO_ERROR) continue;
                PXCGesture::GeoNode dataTo;
                if (detector->QueryNodeData(user,skeletonLines[i].to,&dataTo)<PXC_STATUS_NO_ERROR) continue;
                
                float x0=dataFrom.positionImage.x;
                float y0=dataFrom.positionImage.y;
                float x1=dataTo.positionImage.x;
                float y1=dataTo.positionImage.y;
                MapXY(x0,y0,&depthData,&depthInfo,&rgbInfo);
                MapXY(x1,y1,&depthData,&depthInfo,&rgbInfo);
                if (x0 > 0 && x0 < rgbInfo.width && y0 > 0 && y0 < rgbInfo.height &&
                    x1 > 0 && x1 < rgbInfo.width && y1 > 0 && y1 < rgbInfo.height) {
                    Line line={x0,y0,x1,y1};
                    m_lines.push_back(line);
                }
            }

            for (int j=0;j<sizeof(skeletons)/sizeof(skeletons[0]);j++) {
                PXCGesture::GeoNode data;
                if (detector->QueryNodeData(user,skeletons[j].label,&data)<PXC_STATUS_NO_ERROR) continue;

                float x=data.positionImage.x;
                float y=data.positionImage.y;

                MapXY(x,y,&depthData,&depthInfo,&rgbInfo);
                Node node={x,y,(data.radiusImage?data.radiusImage:skeletons[j].raduis), skeletons[j].color};
                m_nodes[std::pair<pxcUID,PXCGesture::GeoNode::Label>(user,skeletons[j].label)]=node;
            }
        }
        if (depthImage) depthImage->ReleaseAccess(&depthData);

		PXCGesture::GeoNode ndata;
		m_openness[0]=m_openness[1]=0;
		if (detector->QueryNodeData(0,PXCGesture::GeoNode::LABEL_BODY_HAND_LEFT,&ndata)>=PXC_STATUS_NO_ERROR) {
			m_openness[0]=ndata.openness;
			if (ndata.opennessState!=m_last_openness[0]) {
				m_last_openness[0]=ndata.opennessState;
				wprintf_s(L"Openness(L): %s (%d%%)\n", ndata.opennessState==PXCGesture::GeoNode::LABEL_OPEN?L"Open":
											(ndata.opennessState==PXCGesture::GeoNode::LABEL_CLOSE?L"CLOSE":L"Unknown"),
											ndata.openness);
			}
		}
		if (detector->QueryNodeData(0,PXCGesture::GeoNode::LABEL_BODY_HAND_RIGHT,&ndata)>=PXC_STATUS_NO_ERROR) {
			m_openness[1]=ndata.openness;
			if (ndata.opennessState!=m_last_openness[1]) {
				m_last_openness[1]=ndata.opennessState;
				wprintf_s(L"Openness(R): %s (%d%%)\n", ndata.opennessState==PXCGesture::GeoNode::LABEL_OPEN?L"Open":
											(ndata.opennessState==PXCGesture::GeoNode::LABEL_CLOSE?L"CLOSE":L"Unknown"),
											ndata.openness);
			}
		}
    }

    while (m_gestures.size()>0) {
        if (m_gestures.front().count>1) break;
        m_gestures.pop_front();
    }
	if (gdata->timeStamp>0) {
		gdata->timeStamp=0;
        Gesture gesture={0,GESTURE_COUNT,gdata->user,gdata->body};
        for (int k=0;k<sizeof(gesturePics)/sizeof(gesturePics[0]);k++) if (gesturePics[k].label==gdata->label) {
            gesture.bmp=gesturePics[k].bmp;
            break;
        }
        if (gesture.bmp) m_gestures.push_back(gesture);
    }

    return UtilRender::RenderFrame(rgbImage);
}

void GestureRender::DrawMore(HDC hdc, double sx, double sy) {
    HPEN red=CreatePen(PS_SOLID, 4, COLOR_RED);
    SelectObject(hdc, red);
    for (std::list<Line>::iterator itrl=m_lines.begin(); itrl!=m_lines.end(); itrl++) {
        MoveToEx(hdc, (int)(itrl->x0*sx), (int)(itrl->y0*sy), NULL); 
        LineTo(hdc, (int)(itrl->x1*sx), (int)(itrl->y1*sy)); 
    }

    for (std::map<std::pair<pxcUID,PXCGesture::GeoNode::Label>,Node>::iterator itrn=m_nodes.begin(); itrn!=m_nodes.end(); itrn++) {
        HPEN color=CreatePen(PS_SOLID, 4, itrn->second.color);
        SelectObject(hdc, color);

        int x=(int)(itrn->second.x*sx);
        int y=(int)(itrn->second.y*sy);
        int rx=(int)(itrn->second.radius*sx);
        int ry=(int)(itrn->second.radius*sy);
        MoveToEx(hdc, x, y, NULL);
        Arc(hdc,x-rx,y-ry,x+rx,y+ry,x+rx,y+ry,x+rx,y+ry);
        DeleteObject(color);
    }

    HPEN green = CreatePen(PS_SOLID, 4, COLOR_GREEN);
    SelectObject(hdc, green);
    HINSTANCE hInst= GetModuleHandle (0);
    HDC hdc2= CreateCompatibleDC(hdc);
    int y=0, g=255;
    for (std::list<Gesture>::reverse_iterator itrg=m_gestures.rbegin();itrg!=m_gestures.rend();itrg++) {
        itrg->count--;
        HBITMAP bitmap=LoadBitmap(hInst,MAKEINTRESOURCE((itrg->bmp)));
        SelectObject(hdc2, bitmap);
        BITMAP bm; 
        GetObject(bitmap, sizeof(BITMAP), &bm);
        TransparentBlt(hdc,0,y,GESTURE_BOX_SIZE*sx,GESTURE_BOX_SIZE*sy,hdc2,0,0,bm.bmWidth,bm.bmHeight,RGB(255,255,255));
        DeleteObject(bitmap);

        std::map<std::pair<pxcUID,PXCGesture::GeoNode::Label>,Node>::iterator itrn=m_nodes.find(std::pair<pxcUID,PXCGesture::GeoNode::Label>(itrg->user,itrg->label));
        if (itrn!=m_nodes.end() && itrg==m_gestures.rbegin()) {
            MoveToEx(hdc, (int)(GESTURE_BOX_SIZE*sx), (int)(y+(GESTURE_BOX_SIZE/2)*sy), NULL); 
            LineTo(hdc, (int)(itrn->second.x*sx), (int)(itrn->second.y*sy)); 
        }
        y+=GESTURE_BOX_SIZE*sy;
    }
    DeleteDC(hdc2);
    DeleteObject(red);

	/* Draw openness */
	HPEN blue=CreatePen(PS_SOLID,OPENNESS_WIDTH, COLOR_BLUE);
	SelectObject(hdc, blue);
	MoveToEx(hdc,(int)((m_info.bmiHeader.biWidth-OPENNESS_WIDTH*3/2)*sx),(int)(-m_info.bmiHeader.biHeight*sy),0);
	LineTo(hdc,(int)((m_info.bmiHeader.biWidth-OPENNESS_WIDTH*3/2)*sx),(int)((100-m_openness[0])*(-m_info.bmiHeader.biHeight*sy)/100));
	SelectObject(hdc, green);
	MoveToEx(hdc,(int)((m_info.bmiHeader.biWidth-OPENNESS_WIDTH/2)*sx),(int)(-m_info.bmiHeader.biHeight*sy),0);
	LineTo(hdc,(int)((m_info.bmiHeader.biWidth-OPENNESS_WIDTH/2)*sx),(int)((100-m_openness[1])*(-m_info.bmiHeader.biHeight*sy)/100));
	DeleteObject(blue);
    DeleteObject(green);
}
