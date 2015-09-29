/**
 * Copyright 2014: James Steven Supancic III
 **/

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Qt>
#include <QApplication>
#include <QClipboard>
#include <QMimeData>
#include <QImage>
#include <QColor>
#include <QSize>

#include "Scripts.hpp"
#include "RegEx.hpp"
#include "params.hpp"
#include "util.hpp"
#include "RandomHoughFeature.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  using namespace Qt;

  static void script_image_eq()
  {
    string in_filename = g_params.require("FILENAME");
    string out_filename = boost::regex_replace(in_filename,boost::regex(".png"),"_eq.png");
    cout << safe_printf("% => %",in_filename,out_filename) << endl;
    
    Mat Z = imread(in_filename,CV_LOAD_IMAGE_GRAYSCALE|CV_LOAD_IMAGE_ANYDEPTH);
    Z.convertTo(Z,DataType<float>::type);
    Mat ZVis = imageeq("",Z,false,false);
    imwrite(out_filename,ZVis);

    std::exit(0);
  }

  static shared_ptr<QApplication> app;

  static Mat getImage()
  { 
    int argc = 0;
    string arg1 = "null";
    char* arg1_c = &arg1[0];
    if(not app)
      app = make_shared<QApplication>(argc,&arg1_c);
    const QClipboard *clipboard = QApplication::clipboard();
    const QMimeData *mimeData = clipboard->mimeData();

    if (mimeData->hasImage()) 
    {
      cout << "loading image from clipboard"  << endl;
      QImage im = clipboard->image();
      Mat m(im.height(),im.width(),DataType<Vec3b>::type,Scalar::all(0));
      for(int rIter = 0; rIter < im.height(); rIter++)
	for(int cIter = 0; cIter < im.width(); cIter++)
	{
	  QColor px(im.pixel(cIter,rIter));
	  m.at<Vec3b>(rIter,cIter) = Vec3b(px.blue(),px.green(),px.red());
	}
      return m;
    }
    else
    {
      cout << "loading image from FILENAME" << endl;
      string in_filename = g_params.require("FILENAME");
      Mat src = imread(in_filename);
      return src;
    }
  }

  static void putImage(const Mat&m)
  {
    imwrite(params::out_dir() + "/out.png",m);
    QImage i( QSize(m.cols,m.rows), QImage::Format_RGB888 );
    QClipboard *clipboard = QApplication::clipboard();
    for(int rIter = 0; rIter < m.rows; rIter++)
      for(int cIter = 0; cIter < m.cols; cIter++)
      {
	Vec3b pix = m.at<Vec3b>(rIter,cIter);
	QColor color(pix[2],pix[1],pix[0]);
	i.setPixel(cIter,rIter,color.rgb());
      }
    clipboard->setImage(i);
  }

  // make -j16 &&  ./deformable_depth script SCRIPT_COMMAND=inpaint CFG_FILE=scripts/hand.cfg NO_BACKGROUND_REPAINTER=TRUE
  static void script_inpaint()
  {
    //string in_filename = g_params.require("FILENAME");
    //string out_filename = boost::regex_replace(in_filename,boost::regex(".png"),"_eq.png");
    //cout << safe_printf("% => %",in_filename,out_filename) << endl;

    auto is_monocrome = [](Vec3b p) -> bool
    {
      double sum = (double)p[0] + (double)p[1] + (double)p[2];
      vector<double> vs{p[0]/sum,p[1]/sum,p[2]/sum};
      return shannon_entropy(vs) > 1.5846;
    };
    
    Mat src = getImage();
    //image_safe("src",src);
    Mat inpaintMask(src.rows,src.cols,DataType<uint8_t>::type,Scalar::all(0));
    for(int rIter = 0; rIter < src.rows; rIter++)
      for(int cIter = 0; cIter < src.cols; cIter++)
      {
	Vec3b pix = src.at<Vec3b>(rIter,cIter);
	if(not is_monocrome(pix))
	  inpaintMask.at<uint8_t>(rIter,cIter) = 255;
      }

    // INPAINT_NS or INPAINT_TELEA
    Mat dst; cv::inpaint(src,inpaintMask,dst,5,INPAINT_NS);
    for(int rIter = 0; rIter < src.rows; rIter++)
      for(int cIter = 0; cIter < src.cols; cIter++)
      {
	Vec3b pix = src.at<Vec3b>(rIter,cIter);
	if(is_monocrome(pix))
	  dst.at<Vec3b>(rIter,cIter) = src.at<Vec3b>(rIter,cIter);
      }
    putImage(dst);
    //app->exec();
    cout << "DONE" << endl;
    std::chrono::minutes dura(30);
    std::this_thread::sleep_for( dura );
    //app.reset();
    //image_safe("dst",dst);
    //waitKey_safe(0);
  }

  void dispatch_scripts()
  {
    string script_command = g_params.require("SCRIPT_COMMAND");
    if(script_command == "imageeq")      
      script_image_eq();
    else if(script_command == "inpaint")
      script_inpaint();
  }
}
