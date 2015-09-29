/**
 * Copyright 2012: James Steven Supancic III
 **/

#include "Renderer.hpp"
#include <cv.h>
#include <highgui.h>
#include <memory>
#include <GL/glut.h>
#include "HandSynth.hpp"
#include "params.hpp"
#include "VisHog.hpp"
#include <vector>
#include <string>
#include "util.hpp"

namespace deformable_depth
{
  using cv::Scalar;
  using cv::Mat_;
  using cv::Mat;
  using cv::VideoCapture;
  using cv::waitKey;
  using cv::Vec3f;
  using cv::Point;
  using cv::circle;
  using cv::imwrite;
  using cv::FileStorage;
  using cv::calcBackProject;
  using cv::DataType;
  using cv::HOGDescriptor;
  using cv::Size;
  
  using std::vector;
  using std::shared_ptr;
  using std::cout;
  using std::endl;
  using std::numeric_limits;
  using std::string;  
  
  Renderer::Renderer() 
  {    
  }
  
  // works in model space
  // put the geometry where it needs to be...
  void Renderer::draw()
  {
    glPushMatrix();
        
    bool drawHand = true;
    if(drawHand)
    {
      hand.draw();
    }
    else
    {
      // draw a sphere...
      glTranslatef(0.0, 0, -5); // x y z
      // wire shell
      glColor3f(0, 1, 0.0);
      glutWireSphere(1.1, 32, 32); // glutSolidSphere or glutWireSphere
      // solid core
      glColor3f(0, 0.0, 1);
      glutSolidSphere(1, 32, 32); 
    }
    
    glPopMatrix();    
  }

  // http://shitohichiumaya.blogspot.com/2012/10/how-to-get-camera-parameters-from.html
  void gl_get_camera_parameter_from_perspective_matrix(
    double & fovy_rad,
    double & fovx_rad,
    double & clip_min,
    double & clip_max)
  {
    double aspect_ratio;
    GLdouble mat[16];
    glGetDoublev(GL_PROJECTION_MATRIX, mat);
    
    GLdouble const aa = mat[0];
    GLdouble const bb = mat[5];
    GLdouble const cc = mat[10];
    GLdouble const dd = mat[14];

    aspect_ratio = bb / aa;
    fovy_rad     = 2.0f * atan(1.0f / bb);

    GLdouble const kk = (cc - 1.0f) / (cc + 1.0f);
    clip_min = (dd * (1.0f - kk)) / (2.0f * kk);
    clip_max = kk * clip_min;

    // now compute fovx
    fovy_rad = std::abs(fovy_rad);
    assert(fovy_rad > 0);
    fovx_rad = 2 * std::atan(aspect_ratio * std::tan(fovy_rad/2));    
    fovx_rad = std::abs(fovx_rad);
    assert(fovx_rad > 0);    
  }
    
  vector<float> unprojectZ(const int W, const int H, vector<float>& Z)
  {
    vector<float> Zout(W*H);
    
    // unproject Z distances to world coordinate system distances
    //GLdouble MV[4*4];
    GLdouble PROJ[4*4];
    int VP[4]; // w x w h
    // set MV matrix to identity glGetDoublev(GL_MODELVIEW_MATRIX, MV);
    Mat eye4 = Mat::eye(4,4,CV_64F);
    glGetDoublev(GL_PROJECTION_MATRIX,PROJ);
    glGetIntegerv(GL_VIEWPORT,VP);
    for(int xIter = 0; xIter < W; xIter++)
    {
      for(int yIter = 0; yIter < H; yIter++)
      {
	// pixels are row-major ordered
	float z = Z[xIter + (yIter)*W];
	if(z < 1)
	{
	  double oX, oY, oZ;
	  gluUnProject(xIter,yIter,z,
		      eye4.ptr<double>(0),PROJ,VP,&oX,&oY,&oZ);
	  Zout[xIter + (H-1-yIter)*W] = -oZ;
	}
	else
	  Zout[xIter + (H-1-yIter)*W] = numeric_limits<float>::infinity();
      }
    }    
    
    return Zout;
  }
  
  shared_ptr<float> allocRawZ(const int W, const int H, shared_ptr<float> Z)
  {
    shared_ptr<float> Zout(new float[W*H]);
    
    // copy to new format/heap location.
    for(int xIter = 0; xIter < W; xIter++)
    {
      for(int yIter = 0; yIter < H; yIter++)
      {
	float z = (&*Z)[xIter + (yIter)*W];	
	if(z >= 1)
	  (&*Zout)[xIter + (H-1-yIter)*W] = inf;
	else
	{
	  // pixels are row-major ordered
	  (&*Zout)[xIter + (H-1-yIter)*W] = z;
	}
      }
    }    
    
    return Zout;    
  }
  
  void gl_cv_read_buffers_Z(Mat&Zrnd,int W,int H)
  {
    // get the depth buffer.
    vector<float> Z(W*H);
    cout << "W = " << W << " H = " << H << endl;
    glFinish();
    glReadPixels(0,0,W,H,GL_DEPTH_COMPONENT,GL_FLOAT,&Z[0]);
    glFinish();

    Z = unprojectZ(W,H, Z);    
    //Z = allocRawZ(W,H,Z);
    
    // convet to OCV
    Zrnd = Mat_<float>(H,W,&Z[0]).clone();
  }
  
  void gl_cv_read_buffers_RGB(Mat&RGBrnd,int W,int H)
  {
    // get the RGB buffer
    shared_ptr<uchar> RGB(new uchar[3*W*H]);
    glReadPixels(0,0,W,H,GL_BGR,GL_UNSIGNED_BYTE,&*RGB);
    // convert and show with OpenCV
    RGBrnd = Mat(H,W,CV_8UC3,&*RGB).clone();
    cv::flip(RGBrnd,RGBrnd,0);
    //cout << ZM << endl;
    //imshow("Rendered: Depth",ZM);      
  }
  
  void gl_cv_read_buffers(Mat&Zrnd,Mat&RGBrnd,int W,int H)
  {      
    // 
    gl_cv_read_buffers_Z(Zrnd,W,H);
    gl_cv_read_buffers_RGB(RGBrnd,W,H);
  }
  
  void Renderer::doOpenGL(Mat&Zrnd,Mat&RGBrnd)
  {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // setup the view transformation
    glLoadIdentity();
    gluLookAt(0, 0, 0, 0, 0, -1, 0, -1, 0);
    
    // draw the posed object with OpenGL
    draw();
    
    // read the rendered object from OpenGL
    int W = glutGet(GLUT_WINDOW_WIDTH);
    int H = glutGet(GLUT_WINDOW_HEIGHT);
    gl_cv_read_buffers(Zrnd,RGBrnd,W,H);
    
    glutSwapBuffers();
  }
  
  HandData Renderer::doIteration(Mat&Zrnd,Mat&RGBrnd)
  {
    // capture the input
    cout << "doIteration" << endl;
    doOpenGL(Zrnd,RGBrnd);
    
    // show the output
    // showOutput(Zrnd,Zcap,RGBcap,RGBrnd);
    
    // animate the hand?
    //hand.animate();
    //hand.sample();
    
    // construct a hand example? 
    return HandData(RGBrnd,Zrnd);
  }
  
  HandRenderer& Renderer::getHand()
  {
    return hand;
  }
  
  /// SECTION: GLUT
  // model space => screen space...
  void reshape(int width, int height)
  {
    cout << "reshape width = " << width << " height = " << height << endl;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double fovY = 45;
    // A = tand(hfov/2)/tand(vfov/2)
    double aspect = 1.3382; // for Primesense
    // NEAR and FAR need to be set to ensure sufficent percision
    double near = 1;
    double far  = 1000; //numeric_limits<double>::max();
    gluPerspective(fovY, aspect, near, far);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();    
  }
  
  void init_glut(int argc, char**argv, void (* fn_render)( void ))
  {
    glutInit(&argc, argv);
    glutInitWindowSize(640, 480);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutCreateWindow("Rendered: Hand RGB");
    glutDisplayFunc(fn_render);
    glutIdleFunc(fn_render);
    glutReshapeFunc(reshape);
    // init OpenGL
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glShadeModel(GL_SMOOTH);    
  }  
}
