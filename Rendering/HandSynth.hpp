/**
 * Copyright 2012: James Steven Supancic III
 **/

#ifndef DD_HANDSYNTH
#define DD_HANDSYNTH

#include <GL/glut.h>
#include <vector>
#include <cv.h>
#include "MetaData.hpp"

namespace deformable_depth
{
  using std::vector;
  using namespace cv;
  
  class HandData
  {
  public:
    HandData(vector<Rect> parts_,Rect BB = Rect());
    HandData(Mat RGB, Mat_<float> Z);
    Rect bb();
    vector<Rect> distalPhalanges();
    vector<Rect> interProximalPhalanges();
    vector<Rect> core();
    vector<Rect> metacarpal();
    vector<Rect> intermedThumb();
    vector<Rect> distalThumb();
    vector<Rect> phalanges();
    bool isOccluded();
  public: // PBM_Example method implementations
    // part info
    virtual vector<float> partSizes();
    virtual vector<Rect> getParts();
    virtual vector<Rect> getParts(int mixture);
    // mixture info
    virtual int getMixtureOfPart(int part);
    virtual vector<int> getPartsOfMixture(int mixture);
    virtual int getMixtureCt();
    virtual vector<float> mixtureSizes();
  private:
    Rect BB;
    vector<Rect> parts;
  public:
    static constexpr int PART_CT = 16;
    static constexpr float handBBSize = 25;
    static constexpr float distBBSize = 5;
    static constexpr float intrBBSize = 5;
    static constexpr float coreBBSize = 20;
    static constexpr float metaBBSize = 10;
    static constexpr float intrBBThumb = 5;
    static constexpr float distBBThumb = 5;
  };  
  
  class HandRenderer
  {
  private:
    GLUquadricObj *quObj;
    // hand configuration
    float posX, posY, posZ;
    float rotX, rotY;
    // finger configuration
    float*finger_joints[3];
    float fj1[4]; // angles in degrees [0,90]
    float fj2[4]; // angles in degrees [0,90]
    float fj3[4]; // angles in degrees [0,90]
    float fk[4];  // angles in degrees [-15,+15]
    // thumb configuration
    float thumb_mci, thumb_mcj; // metacarpal joints [0,90] [0,45]
    float thumb_j2, thumb_j3; // [0,45] [0,90]
    // arm cfg...
    float arm_i; // [-45,45] 
    float arm_j; // [-25,25]
  protected:
    void drawFingers();
    void drawThumb();
  public:
    HandRenderer();
    virtual ~HandRenderer();
    void draw();
    void animate();
    void sample();
    vector<Rect> findDistalPhalanges(Mat&im);
    void setDefaultPose();
    void setPos(float x, float y, float z);
    void setRot(float x, float y);
    static Scalar getPartColor(int partID);
  public:
    static constexpr float distPhalangesBBSize = HandData::distBBSize;
    static constexpr int PART_CT = HandData::PART_CT;
  };
    
  HandData loadRandomExample(string directory,Mat&RGB,Mat_<float>&Z);
}

#endif
