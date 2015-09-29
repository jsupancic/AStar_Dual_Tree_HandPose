/**
 * Copyright 2012: James Steven Supancic III
 **/

#include <GL/glut.h>
#include "HandSynth.hpp"
#include "util.hpp"

#include <iostream>
#include <boost/filesystem.hpp>
#include <thread>

#define use_speed_ 0
#include <cv.h>
#include <highgui.h>

namespace deformable_depth
{
  using namespace boost::filesystem;
  using namespace cv;
  using namespace std;
  
  // following function copyright Samuel R. Buss
  void cylinder(GLUquadricObj*quObj,double radiusBase, 
		double radiusTop, double height, int slices, int stacks )
  {
    // First draw the cylinder
    gluCylinder(quObj, radiusBase, radiusTop, height, slices, stacks);
    
    // Draw the top disk cap
    glPushMatrix();
    glTranslated(0.0, 0.0, height);
    gluDisk( quObj, 0.0, radiusTop, slices, stacks );
    glPopMatrix();

    // Draw the bottom disk cap
    glPushMatrix();
    glRotated(180.0, 1.0, 0.0, 0.0);
    gluDisk( quObj, 0.0, radiusBase, slices, stacks );
    glPopMatrix();
  }
  
  // CTOR
  HandRenderer::HandRenderer()
  {
    quObj = gluNewQuadric();
    setDefaultPose();
  }
  
  void HandRenderer::setDefaultPose()
  {
    // configure the hand
    posX = 0; 
    posY = 0;
    posZ = -80; // 1 meter away = -100
    rotY = 0;
    rotX = 0;
    
    // configure the fingers
    for(int fIter = 0; fIter < 4; fIter++)
      fj1[fIter] = fj2[fIter] = fj3[fIter] = fk[fIter] = 0;
    finger_joints[0] = fj1;
    finger_joints[1] = fj2;
    finger_joints[2] = fj3;
    
    // confoigure the thumb
    thumb_mci = thumb_mcj = 0;
    thumb_j2 = thumb_j3 = 0;
    
    // configure arm 
    arm_i = 0;
    arm_j = 0;    
  }
    
  void HandRenderer::setPos(float x, float y, float z)
  {
    posX = x;
    posY = y;
    posZ = z;
  }

  void HandRenderer::setRot(float x, float y)
  {
    rotX = x;
    rotY = y;
  }
  
  HandRenderer::~HandRenderer()
  {
  }
  
  void animateInRange(float&val,float min, float max)
  {
    if(val >= max)
      val = min;
    else val++;
  }
  
  float sample_in_range(float min, float max)
  {
    float U = ((float)rand())/RAND_MAX;
    float V = (max-min)*U+min;
    //std::cout << "sample = " << V << std::endl;
    return V;
  }
  
  void HandRenderer::sample()
  {
    // animate the hand location
    posX = sample_in_range(-50,50);
    posY = sample_in_range(-50,50);
    posZ = sample_in_range(-200,-80);
    rotY = sample_in_range(0,360);
    rotX = sample_in_range(0,360);
    
    // animate the finger articulation.
    for(int jointIter = 0; jointIter < 3; jointIter++)
    {
      finger_joints[jointIter][0] = sample_in_range(0,90);
      finger_joints[jointIter][1] = sample_in_range(0,90);
      finger_joints[jointIter][2] = sample_in_range(0,90);
      finger_joints[jointIter][3] = sample_in_range(0,90);
      fk[jointIter] = sample_in_range(-15,15);
    }
    
    // animate the thumb!
    thumb_mci = sample_in_range(0,90);
    thumb_mcj = sample_in_range(0,45);
    thumb_j2 = sample_in_range(0,45);
    thumb_j3 = sample_in_range(0,90);
    
    // sample the arm
    arm_i = sample_in_range(-45,45);
    arm_j = sample_in_range(-25,25);
  }
  
  void HandRenderer::animate()
  {
    // translate?
    animateInRange(posX,-50,50);
    animateInRange(posY,-50,50);
    animateInRange(posZ,-200,-80);
    
    // rotate
    rotY++;
    rotX++;
    
    // animate finger joint 1
    for(int jointIter = 0; jointIter < 3; jointIter++)
    {
      if(finger_joints[jointIter][0] >= 90)
	finger_joints[jointIter][0] = 0;
      else
	finger_joints[jointIter][0]++;
      if(finger_joints[jointIter][3] <= 0)
	finger_joints[jointIter][3] = 90;
      else
	finger_joints[jointIter][3]--;
    }
    
    // animate finger k joints
    if(fk[2] <= -15)
      fk[2] = +15;
    else
      fk[2]--;
    
    animateInRange(thumb_mci,0,90);
    animateInRange(thumb_mcj,0,45);
    animateInRange(thumb_j2,0,45);
    animateInRange(thumb_j3,0,90);
  }
  
  static void setGLColor(Scalar c)
  {
    glColor3f(c[0],c[1],c[2]);
  }
  
  Scalar HandRenderer::getPartColor(int partID)
  {
    if(partID >= 0 && partID <= 14)
    {
      // finger of some kind
      float fingerNum = partID / 3; // 0 to 5
      float phalangalIdx = partID % 3; // 0 to 2
      return Scalar(1,fingerNum/5,phalangalIdx/2);
    }
    else if(partID == 15)
    {
      // palm
      return Scalar(0,1,1);
    }
    else if(partID == -1)
    {
      return Scalar(0,0,0);
    }
    else
    {
      // error
      throw "bad part ID";
    }
  }
  
  // for this project, lets use cm
  void HandRenderer::draw()
  {
    // set the hand's position
    glTranslatef(posX, posY, posZ); // x y z
    glRotatef(rotY,0,1,0);
    glRotatef(rotX,1,0,0);   
    
    // draw the 'core'
    // 10cm in x, 10 cm in y, 2.5cm in z
    glPushMatrix();
    glScalef(1,1,.25);
    setGLColor(getPartColor(15));
    glutSolidCube(10);
    glPopMatrix();
    
    // draw the fignres. 7.5 cm long
    drawFingers();
    
    // draw the thumb
    drawThumb();
    
    // draw arm
    glPushMatrix();
    setGLColor(getPartColor(-1));
    glTranslatef(0,14,0);
    // apply rotation
    glTranslatef(0,-10,0);
    glRotatef(arm_i,1,0,0); // up down
    glRotatef(arm_j,0,0,1); // left right
    glTranslatef(0,10,0);
    // draw the cube
    glScalef(.8,2,.25);
    glutSolidCube(10);
    glPopMatrix();
  }
  
  void HandRenderer::drawFingers()
  {
    for(int fingerIter = 0; fingerIter < 4; fingerIter++)
    {
      /// BEGIN
      // figure out it's x traslation
      float xTrans = -(2.5*1.5) + fingerIter*2.5;
      glPushMatrix();
           
      /// Component 1: Proximal Phalanges
      // position it!
      glTranslatef(xTrans, -8.5 + 7.5/2, 0);
      // 7.5 tall in the Y axis...
      glRotatef(fk[fingerIter],0,0,1);
      glRotatef(90+fj1[fingerIter],1,0,0);
      // draw it!
      glBegin(GL_POLYGON);
      setGLColor(getPartColor(fingerIter*3+0));
      cylinder(quObj, 1, 1, 2.5+.1, 30, 30);
      glEnd();
      
      /// Component 2 : Intermidate Phalanges
      glTranslatef(0,0,2.5);
      glRotatef(fj2[fingerIter],1,0,0);
      glBegin(GL_POLYGON);
      setGLColor(getPartColor(fingerIter*3+1));
      cylinder(quObj, 1, 1, 2.5+.1, 30, 30);
      glEnd();           
      
      /// Component 3 : Distal Phalanges
      glTranslatef(0,0,2.5);
      glRotatef(fj3[fingerIter],1,0,0);
      glBegin(GL_POLYGON);
      setGLColor(getPartColor(fingerIter*3+2));
      cylinder(quObj, 1, 1, 2.5+.1, 30, 30);
      glEnd();       
      
      /// END
      glPopMatrix();
    }
  }

  void HandRenderer::drawThumb()
  {
    glPushMatrix();
    // thumb's position
    glTranslatef(-5, 0, 0);
    glRotatef(90,1,0,0);
    
    /// draw the metacarpal region.
    glTranslatef(2,0,0);
    glRotatef(thumb_mci,0,0,1);
    glTranslatef(-2,0,0);
    glRotatef(-thumb_mcj,0,1,0);
    glBegin(GL_POLYGON);
    setGLColor(getPartColor(14));
    cylinder(quObj, 1.5, 1.5, 3.5+.1, 30, 30);
    glEnd();
    
    /// draw the proximal phalanges
    glTranslatef(0,0,3.5);
    glRotatef(thumb_j2,1,0,0);
    glBegin(GL_POLYGON);
    setGLColor(getPartColor(13));
    cylinder(quObj, 1, 1, 2.5+.1, 30, 30);
    glEnd();  
    
    /// draw the distal phalanges
    glTranslatef(0,0,2.5);
    glRotatef(thumb_j3,1,0,0);
    glBegin(GL_POLYGON);
    setGLColor(getPartColor(12));
    cylinder(quObj, 1, 1, 2.5+.1, 30, 30);
    glEnd();      
      
    /// end
    glPopMatrix();
  }
  
  vector< Rect > HandRenderer::findDistalPhalanges(Mat& im)
  {
    Rect bb1 = bbWhere(im,
      [](Mat&im,int r,int c)
      {
	Vec3b pixel = im.at<Vec3b>(r,c);
	return pixel == Vec3b(256*1.f/3,0,0);
      });
    Rect bb2 = bbWhere(im,
      [](Mat&im,int r,int c)
      {
	Vec3b pixel = im.at<Vec3b>(r,c);
	return pixel == Vec3b(256*1.f/3,0,256*0.25);
      });
    Rect bb3 = bbWhere(im,
      [](Mat&im,int r,int c)
      {
	Vec3b pixel = im.at<Vec3b>(r,c);
	return pixel == Vec3b(256*1.f/3,0,255*.5);
      });
    Rect bb4 = bbWhere(im,
      [](Mat&im,int r,int c)
      {
	Vec3b pixel = im.at<Vec3b>(r,c);
	return pixel == Vec3b(256*1.f/3,0,255*.75);
      });
    Rect bb5 = bbWhere(im,
      [](Mat&im,int r,int c)
      {
	Vec3b pixel = im.at<Vec3b>(r,c);
	return pixel == Vec3b(256*1.f/3,0,255*1);
      });
    
    return {bb1,bb2,bb3,bb4,bb5};
  }
  
  /// HandData file IMPL
  HandData loadRandomExample(string directory,Mat&RGB,Mat_<float>&Z)
  {
    string stem;
    loadRandomRGBZ(RGB,Z,directory,&stem);
    
    // get the bounding box for the rendered hand...
    vector<Rect> parts;
    string labelFile = stem + ".labels.yml";
    FileStorage label_file(labelFile,FileStorage::READ);
    for(int pIter = 1; pIter <= HandRenderer::PART_CT; pIter++)
    {
      Rect newPart;
      vector<int> p;
      ostringstream oss; oss << "Part" << pIter;
      if(!label_file[oss.str()].isNone())
      {
	label_file[oss.str()] >> p;
	// assuming all parts have phalangal size?
	newPart = RGBCamera().bbForDepth(Z,p[1],p[0],
				  HandRenderer::distPhalangesBBSize,
				  HandRenderer::distPhalangesBBSize);
      }
      else
	newPart = Rect();
      parts.push_back(newPart);
    }
    // get the general BB
    Rect BB;
    if(!label_file["Hand"].isNone())
    {
      vector<int> p;
      label_file["Hand"] >> p;
      BB = RGBCamera().bbForDepth(Z,p[1],p[0],HandData::handBBSize,HandData::handBBSize);
    }
    else
      BB = rectUnion(parts);
    label_file.release();    
    
    return HandData(parts,BB);
  }
  
  vector< Rect > HandData::phalanges()
  {
    vector<Rect> bbs;
    bbs.insert(bbs.end(),distalPhalanges().begin(),distalPhalanges().end());
    bbs.insert(bbs.end(),interProximalPhalanges().begin(),interProximalPhalanges().end());
    bbs.insert(bbs.end(),distalThumb().begin(),distalThumb().end());
    bbs.insert(bbs.end(),intermedThumb().begin(),intermedThumb().end());
    return bbs; 
  }
  
  Rect HandData::bb()
  {
    return BB;
  }

  vector< Rect > HandData::distalPhalanges()
  {
    vector<Rect> bbs;
    bbs.push_back(parts[0]);
    bbs.push_back(parts[3]);
    bbs.push_back(parts[6]);
    bbs.push_back(parts[9]);
    return bbs;    
  }

  vector< Rect > HandData::core()
  {
    vector<Rect> bbs;
    bbs.push_back(parts[15]);
    return bbs;
  }

  vector< Rect > HandData::distalThumb()
  {
    vector<Rect> bbs;
    bbs.push_back(parts[12]);
    return bbs;
  }

  vector< Rect > HandData::intermedThumb()
  {
    vector<Rect> bbs;
    bbs.push_back(parts[13]);
    return bbs;
  }

  vector< Rect > HandData::interProximalPhalanges()
  {
    vector<Rect> bbs;
    bbs.push_back(parts[1]); bbs.push_back(parts[2]);
    bbs.push_back(parts[4]); bbs.push_back(parts[5]);
    bbs.push_back(parts[7]); bbs.push_back(parts[8]);
    bbs.push_back(parts[10]); bbs.push_back(parts[11]);
    return bbs;
  }

  vector< Rect > HandData::metacarpal()
  {
    vector<Rect> bbs;
    bbs.push_back(parts[14]);
    return bbs;
  }
  
  std::vector< float > HandData::partSizes()
  {
    vector<float> szs = {
      distBBSize,intrBBSize,intrBBSize,
      distBBSize,intrBBSize,intrBBSize,
      distBBSize,intrBBSize,intrBBSize,
      distBBSize,intrBBSize,intrBBSize,
      distBBThumb,intrBBThumb,metaBBSize,
      coreBBSize
    };
    return szs;
  }
  
  vector< Rect > HandData::getParts()
  {
    return parts;
  }
  
  HandData::HandData(vector<Rect> parts_, Rect BB_) : parts(parts_)
  {
    if(BB_ == Rect())
      BB = rectUnion(parts);
    else
      BB = BB_;
  }
  
  // construct a hand from a RGB rendering.
  bool colMatch(Mat&im,int y, int x,Scalar tc)
  {
    Vec3b pixB = im.at<Vec3b>(y,x);
    //printf("im.size = %dx%d y = %d x = %d\n",im.rows,im.cols,y,x);
    Vec3f pix(pixB[2],pixB[1],pixB[0]); pix /= 255;
    Vec3f tar(tc[0],tc[1],tc[2]);
    Vec3f d = pix-tar;
    float dist = ::sqrt(d.dot(d));
    //printf("colMatch dist = %f\n",dist);
    return dist < 5.0/255;
  };  
  
  bool HandData::isOccluded()
  {
    for(Rect r : parts)
      if(r != Rect(0,0,0,0))
	return false;
    return true;
  }
  
  HandData::HandData(Mat RGB, cv::Mat_< float > Z)
  {
    for(int partIter = 0; partIter < HandRenderer::PART_CT; partIter++)
    {
      Scalar tc = HandRenderer::getPartColor(partIter);
      Rect part = bbWhere(RGB,[&tc](Mat&im,int x, int y){return colMatch(im,x,y,tc);});
      if(!(part.x == 0 && part.y == 0 && part.width >= Z.cols -1 && part.height >= Z.rows - 1))
	parts.push_back(part);
      else
	parts.push_back(Rect(0,0,0,0));
    }
    
    // print?
    printf("Loaded hand\n");
    for(int iter = 0; iter < parts.size(); iter++)
    {
      Rect p = parts[iter];
      printf("\tPart = %d %d %d %d\n",p.x,p.y,p.br().x,p.br().x);
    }
  }
  
  int HandData::getMixtureCt()
  {
    return 6;
  }

  int HandData::getMixtureOfPart(int part)
  {
    switch(part)
    {
      // distal phalanges
      case 0:
      case 3:
      case 6:
      case 9:
	return 0;
      // interProximalPhalanges
      case 1:
      case 2:
      case 4:
      case 5:
      case 7:
      case 8:
      case 10:
      case 11:
	return 1;
      case 12:
	return 2;
      case 13:
	return 3;
      case 14:
	return 4;
      case 15:
	return 5;
      default:
	throw "bad part number";
    }
  }
  
  std::vector< float > HandData::mixtureSizes()
  {
    return {distBBSize, intrBBSize, distBBThumb, intrBBThumb, metaBBSize, coreBBSize};
  }  

  vector< Rect > HandData::getParts(int mixture)
  {
    vector<int> idxs = getPartsOfMixture(mixture);
    vector<Rect> parts = getParts();
    vector<Rect> result;
    for(int idx : idxs)
      result.push_back(parts[idx]);
    return result;
  }
  
  vector< int > HandData::getPartsOfMixture(int mixture)
  {
    switch(mixture)
    {
      case 0:
	return {0,3,6,9};
      case 1:
	return {1,2,4,5,7,8,10,11};
      case 2:
	return {12};
      case 3:
	return {13};
      case 4:
	return {14};
      case 5:
	return {15};
      default:
	throw "bad mixture #";
    }
  }
}
