/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_BaselineDetection
#define DD_BaselineDetection

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

#ifdef DD_CXX11
#include "Detection.hpp"
#include "InverseKinematics.hpp"
#endif

namespace deformable_depth
{
	using cv::Mat;
	using cv::Rect;
	using std::string;
	using std::map;
	using cv::FileNode;
	using std::vector;
	using cv::FileStorage;
        class PoseRegressionPoint;

	struct BaselineDetection
	{
	public:
		// members
		Mat blob;
		Rect bb;
		map<string,BaselineDetection> parts;
		string filename;
		double resp;
		string notes;
		shared_ptr<PoseRegressionPoint> pose_reg_point;
	  bool remapped;
		
		// methods
		void draw(Mat&im,cv::Scalar color = cv::Scalar(255,0,0));
		void scale(float factor);
		BaselineDetection();
#ifdef DD_CXX11
		string respString() const;
		BaselineDetection(const Detection&copyMe);
		void interpolate_parts(const BaselineDetection&source);
		bool include_part(string part_name) const;
#endif
	};
	void write(cv::FileStorage &,const std::string &,const BaselineDetection&writeme);
        void write(cv::FileStorage &,const std::string &,const shared_ptr<BaselineDetection>&writeme);
	void read(const FileNode&, BaselineDetection&, BaselineDetection);
        void read(const FileNode&, shared_ptr<BaselineDetection>&, shared_ptr<BaselineDetection>);
	vector<BaselineDetection> loadBaseline(string filename, int length = -1);
	// apply multiplication to the bounding boxes, recursive to parts.
	BaselineDetection operator* (double weight, const BaselineDetection& mult);
	BaselineDetection operator+ (const BaselineDetection& lhs, const BaselineDetection& rhs);
	
	// replace any BB = Rect() == No Detection results
	// with a linear interpolation. Fill any gaps to get full recall.
	void dilate(vector<BaselineDetection>&track,int bandwidth);
	void interpolate(vector<BaselineDetection>&track);
	void post_interpolate_parts(vector<BaselineDetection>&track);
	void interpolate_ik_regress_full_hand_pose(vector<BaselineDetection>&track);
	
	// show baselines on a video
	void show_baseline_on_video();
}

#endif
