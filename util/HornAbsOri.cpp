/**
 * Copyright 2013: James Steven Supancic III
 **/

// implements http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
#ifdef DD_ENABLE_HAND_SYNTH
#include "HornAbsOri.hpp"
#include "util.hpp"
#include <boost/concept_check.hpp>
#include <Quaternion.hpp>

namespace deformable_depth
{
  using namespace cv;
  
  Point3d centroid(const vector<Point3d>& pts)
  {
    Point3d centroid(0,0,0);
    for(const Point3d & p : pts)
      centroid += p;
    centroid.x /= pts.size();
    centroid.y /= pts.size();
    centroid.z /= pts.size();
    return centroid;
  }
  
  // this value should be decreased by the Horn operation.
  double distance(const vector<Point3d> & r_l,
		  const vector<Point3d> & r_r)
  {
    double dist = 0;
    
    for(int iter = 0; iter < r_l.size(); ++iter)
    {
      Point3d offset = r_l[iter] - r_r[iter];
      dist += offset.ddot(offset);
    }
    
    return dist;
  }
  
  void computeS(double n,
		vector<Point3d>&centered_l,
		vector<Point3d>&centered_r,
		Point3d centroid_l,
		Point3d centroid_r,
		Mat&S
 	      )
  {
    const int IDX_X = 0, IDX_Y = 1, IDX_Z = 2;
    auto addPtProduct = [](Mat&M,Point3d&pt1,Point3d&pt2)
    {
      M.at<double>(IDX_X,IDX_X) += pt1.x * pt2.x; 
      M.at<double>(IDX_X,IDX_Y) += pt1.x * pt2.y;
      M.at<double>(IDX_X,IDX_Z) += pt1.x * pt2.z; 
      M.at<double>(IDX_Y,IDX_X) += pt1.y * pt2.x; 
      M.at<double>(IDX_Y,IDX_Y) += pt1.y * pt2.y;
      M.at<double>(IDX_Y,IDX_Z) += pt1.y * pt2.z;      
      M.at<double>(IDX_Z,IDX_X) += pt1.z * pt2.x; 
      M.at<double>(IDX_Z,IDX_Y) += pt1.z * pt2.y;
      M.at<double>(IDX_Z,IDX_Z) += pt1.z * pt2.z;      
    };

    Mat muLmuR(3,3,DataType<double>::type,Scalar::all(0));
    addPtProduct(muLmuR,centroid_l,centroid_r);

    S = Mat(3,3,DataType<double>::type,Scalar::all(0));    
    for(int iter = 0; iter < n; ++iter)
    {
      addPtProduct(S,centered_l[iter],centered_r[iter]);
    }    
    S += (muLmuR * -centered_l.size());
  }
  
  void computeN(const Mat&S,Mat&N)
  {
    const int IDX_X = 0, IDX_Y = 1, IDX_Z = 2;
    N = Mat(4,4,DataType<double>::type,Scalar::all(0));
    // row 0
    N.at<double>(0,0) = S.at<double>(IDX_X,IDX_X) + S.at<double>(IDX_Y,IDX_Y) + S.at<double>(IDX_Z,IDX_Z);
    N.at<double>(0,1) = S.at<double>(IDX_Y,IDX_Z) - S.at<double>(IDX_Z,IDX_Y);
    N.at<double>(0,2) = S.at<double>(IDX_Z,IDX_X) - S.at<double>(IDX_X,IDX_Z);
    N.at<double>(0,3) = S.at<double>(IDX_X,IDX_Y) - S.at<double>(IDX_Y,IDX_X);
    // row 1 
    N.at<double>(1,0) = S.at<double>(IDX_Y,IDX_Z) - S.at<double>(IDX_Z,IDX_Y);
    N.at<double>(1,1) = S.at<double>(IDX_X,IDX_X) - S.at<double>(IDX_Y,IDX_Y) - S.at<double>(IDX_Z,IDX_Z);
    N.at<double>(1,2) = S.at<double>(IDX_X,IDX_Y) + S.at<double>(IDX_Y,IDX_X);
    N.at<double>(1,3) = S.at<double>(IDX_Z,IDX_X) + S.at<double>(IDX_X,IDX_Z);
    // row 2 
    N.at<double>(2,0) = S.at<double>(IDX_Z,IDX_X) - S.at<double>(IDX_X,IDX_Z);
    N.at<double>(2,1) = S.at<double>(IDX_X,IDX_Y) + S.at<double>(IDX_Y,IDX_X);
    N.at<double>(2,2) = -S.at<double>(IDX_X,IDX_X) + S.at<double>(IDX_Y,IDX_Y) - S.at<double>(IDX_Z,IDX_Z);
    N.at<double>(2,3) = S.at<double>(IDX_Y,IDX_Z) + S.at<double>(IDX_Z,IDX_Y);
    // row 3 
    N.at<double>(3,0) = S.at<double>(IDX_X,IDX_Y) - S.at<double>(IDX_Y,IDX_X);
    N.at<double>(3,1) = S.at<double>(IDX_Z,IDX_X) + S.at<double>(IDX_X,IDX_Z);
    N.at<double>(3,2) = S.at<double>(IDX_Y,IDX_Z) + S.at<double>(IDX_Z,IDX_Y);
    N.at<double>(3,3) = -S.at<double>(IDX_X,IDX_X) - S.at<double>(IDX_Y,IDX_Y) + S.at<double>(IDX_Z,IDX_Z);
  }
  
  vector<Point3d> operator - (const vector<Point3d>&pts, Point3d centroid)
  {
    vector<Point3d> result;
    for(const Point3d&pt : pts )
      result.push_back(pt - centroid);
    return result;
  }
  
  AbsoluteOrientation distanceAbsoluteOrientation(
    const vector< Point3d >& r_l, 
    const vector< Point3d >& r_r)
  {
    // check the pre-conditions
    double dist1 = distance(r_l,r_r);
    double n = r_l.size();
    assert(r_l.size() == r_r.size());
    
    // first we find the centroids
    Point3d centroid_l = centroid(r_l);
    Point3d centroid_r = centroid(r_r);
    
    // subtract the centroids
    vector<Point3d> centered_l = r_l - centroid_l;
    vector<Point3d> centered_r = r_r - centroid_r;
    
    // compute the matrix S
    Mat S;
    computeS(n,centered_l,centered_r,centroid_l,centroid_r,S);
    
    // compute the 4x4 matrix N
    Mat N; computeN(S,N);
    
    // compute the largest eigenvector of N and convert it to quaterion form
    Mat eig_values, eig_vectors;
    cv::eigen(N,eig_values,eig_vectors);
    // normalize to unit length
    //double v_norm = std::sqrt(((Mat)(eig_vectors.row(0) * eig_vectors.row(0).t())).at<double>(0));
    //eig_vectors.row(0) /= v_norm;
    double q0 = eig_vectors.at<double>(0,0);
    double qx = eig_vectors.at<double>(0,1);
    double qy = eig_vectors.at<double>(0,2);
    double qz = eig_vectors.at<double>(0,3);
    Quaternion q(q0,qx,qy,qz);
    q.normalize();
    //cout << printfpp("q = [%f %f %f %f]",q0,qx,qy,qz) << endl;
    //cout << printfpp("q = [%f %f %f %f]",q.get()[0],q.get()[1],q.get()[2],q.get()[3]) << endl;
    
    // now we can compute the rotation matrix R
    Mat R = q.rotation_matrix();
    
    // now compute the optimal scale
    double v1 = 0, v2 = 0;
    for(int iter = 0; iter < n; ++iter)
    {
      v1 += centered_r[iter].ddot(centered_r[iter]);
      v2 += centered_l[iter].ddot(centered_l[iter]);
    }
    double scale = std::sqrt(v1/v2);
    
    // compute the translation
    // scale*R*(Mat)centroid_l?
    Mat T = (Mat)centroid_r - scale*R*(Mat)centroid_l;
    
    // translate the points
    vector<Point3d> registered_l;
    for(int iter = 0; iter < n; ++iter)
    {
      Mat reg = scale*R*(Mat)r_l[iter] + T;
      assert(reg.type() == DataType<double>::type);
      registered_l.push_back(Point3d(reg.at<double>(0),
	reg.at<double>(1),reg.at<double>(2)));
    }
    
    // compute and return the checked new distance
    double dist2 = distance(r_r,registered_l);
    if(dist2 >= dist1)
    {
      cout << "T: " << T << endl;
      cout << "R: " << R << endl;
      cout << "S: " << scale << endl;
      cout << printfpp("dist1 = %f dist2 = %f",dist1,dist2) << endl;
      //assert(false);
    }
    
    AbsoluteOrientation abs_ori;
    abs_ori.R = R;
    abs_ori.scale = scale;
    abs_ori.T = T;
    abs_ori.quaternion = q;
    abs_ori.distance = dist2;
    return abs_ori;
  }
  
  AbsoluteOrientation distHornAO(const vector< Vec3d >& xs, const vector< Vec3d >& ys)
  {
    vector<Point3d> xs_pt;
    vector<Point3d> ys_pt;
    
    for(const Vec3d& vec : xs)
      xs_pt.push_back(Point3d(vec[0],vec[1],vec[2]));
    
    for(const Vec3d& vec : ys)
      ys_pt.push_back(Point3d(vec[0],vec[1],vec[2]));
    
    return distanceAbsoluteOrientation(xs_pt,ys_pt);
  }
}
#endif

