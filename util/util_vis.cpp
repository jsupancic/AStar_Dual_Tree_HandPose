/**
 * Copyright 2014: James Steven Supancic III
 **/
 
#include "util_vis.hpp"
#include "util.hpp"
#include "ThreadCompat.hpp"
#include "MetaData.hpp"
#include "BackgroundThreads.hpp"
#include "Colors.hpp"
 
namespace deformable_depth
{
  recursive_mutex exclusion_high_gui;

  Mat mm_quant(const Mat&m)
  {
    // pre quantize to 1cm resolution?
    Mat mm = m.clone();
    for(int yIter = 0; yIter < mm.rows; ++yIter)
      for(int xIter = 0; xIter < mm.cols; ++xIter)
      {
	float&z = mm.at<float>(yIter,xIter);
	z = std::round(10*z)/10;
      }
    return mm;
  }
  
  Mat eq(const Mat&m)
  {    
    return imageeq("",m,false,false,true);
  }
  
  Mat image_text(const Mat& bg, string text, Vec3b color)
  {
    // define our font information
    int font_face = FONT_HERSHEY_PLAIN ;
    double font_scale = 1;
    Scalar font_color(color[0],color[1],color[2]);
    int font_thickness = 1;
    
    // figure out how large an area we need for the text
    int baseline;
    Size text_size = getTextSize(text, font_face, font_scale, font_thickness, &baseline);
#ifdef DD_CXX11
    cout << "text_size: " << text_size << endl;
#endif
    
    Mat out = bg.clone();
    cv::resize(out,out,Size(text_size.width + 10,text_size.height + 10));
    
    // draw the text
    Point orig_bl(10/2,out.rows - 10/2);
    putText(out,text,orig_bl,font_face,font_scale,font_color,font_thickness);
    
    return out;
  }

  Mat im_merge(const Mat&r, const Mat&g, const Mat&b)
  {
    int rows = std::max(r.rows,g.rows);
    int cols = std::max(r.cols,g.cols);
    Mat im(rows,cols,DataType<Vec3b>::type,Scalar::all(0));
    for(int yIter = 0; yIter < rows; ++yIter)
      for(int xIter = 0; xIter < cols; ++xIter)
      {
	if(!b.empty())
	  im.at<Vec3b>(yIter,xIter)[0] = b.at<Vec3b>(yIter,xIter)[0];
	if(!g.empty())
	  im.at<Vec3b>(yIter,xIter)[1] = g.at<Vec3b>(yIter,xIter)[1];
	if(!r.empty())
	  im.at<Vec3b>(yIter,xIter)[2] = r.at<Vec3b>(yIter,xIter)[2];
      }
    return im;
  }
  
  Mat image_text(string text,Vec3b color,Vec3b bg)
  {
    Mat text_area(5,5,DataType<Vec3b>::type,Scalar(bg[0],bg[1],bg[2]));
    
    // return the result
    return image_text(text_area, text, color);
  }
  
  Mat image_text(vector< string > lines)
  {
    Mat text_area;
    
	for(int iter = 0; iter < lines.size(); ++iter)
      text_area = vertCat(text_area,image_text(lines[iter]));
    
    return text_area;
  }
  
  bool have_display()
  {
#ifdef WIN32
	  return true;
#else
    return getenv("DISPLAY") != NULL;
#endif
  }
  
  Mat display(string title, Mat image,bool show, bool resize)
  {
    //cout << printfpp("display: %s type: %d show: %d",title.c_str(),image.type(),show) << endl;
    
    if(image.type() == DataType<float>::type)
      return imageeq(title.c_str(),image,show,resize);
    else if(show)
    {
      image_safe(title,image,resize);
    }
    
    return image;
  }
  
  void imshow(const string& winname, InputArray mat)
  {    
    //cout << printfpp("cv::imshow called: %s %d",winname.c_str(),mat.getMat().size().area()) << endl;
    
    if(have_display())
    {
      cv::imshow(winname,mat);
    }
    else
    {
      // Display to Disk!
      std::ostringstream oss;
      oss << params::out_dir() + "display_";
      oss << winname;
      oss << ".jpg";      
      cv::imwrite(oss.str(),mat);
    }
  }

  static bool CS_WAIT_KEY_active = false;
  
  int waitKey(int delay=0)
  {
    if(have_display())
    {
      lock_guard<recursive_mutex> l(exclusion_high_gui);
      if(background_repainter_running())
	assert(CS_WAIT_KEY_active);
      static bool inited = false;
      if(not inited)
      {
	cv::namedWindow("tmp");
      }
      return cv::waitKey(delay);
    }
    else
    {
      // NOP in this case!
      return -1;
    }    
  }     
  
  
  Mat imVGA(Mat im, int interpolation)
  {
    Mat showme;
    double sf = std::sqrt(640.f*480.f/im.size().area());
    resize(im,showme,Size(),sf,sf, interpolation);
    return showme;
  }
    
  static vector<Mat> rotGlyphs(int sz,int nbins,bool contrast_sensitive)
  {
    Mat vert(sz,sz,DataType<float>::type,Scalar::all(0));
    if(contrast_sensitive)
      line(vert,Point(sz/2,0),Point(sz/2,sz/2),Scalar::all(1));
    else
      line(vert,Point(sz/2,0),Point(sz/2,sz),Scalar::all(1));
    vector<Mat> result;
    for(int iter = 0; iter < nbins; iter++)
    {
      // setup transofmr
      Point center(sz/2,sz/2);
      double scale = 1;
      double angle = 360*iter/static_cast<double>(nbins)*(contrast_sensitive?2:1);
      if(contrast_sensitive && iter >= nbins/2)
	angle += 180;
      Mat rotMat = cv::getRotationMatrix2D(center,angle,scale);
      
      // apply and add
      Mat tp; warpAffine(vert,tp,rotMat,vert.size());
      result.push_back(tp);
    }
    
    return result;
  }
  
  Mat picture_HOG_one(DepthFeatComputer& hog, std::vector< double > T,PictureHOGOptions opts)
  {
    int viz_cell_size = 10;
    int nbins = opts.contrast_sensitive?params::ORI_BINS:params::ORI_BINS/2;
    // get and show the rotation glyphs
    vector<Mat> glyphs = rotGlyphs(viz_cell_size,nbins,opts.contrast_sensitive);
//     for(int iter = 0; iter < rg.size(); iter++)
//     {
//       ostringstream oss; oss << iter;
//       imagesc(oss.str().c_str(),rg[iter]);
//     }
//     deformable_depth::waitKey(0);
   
    // figure out the dimensional breakdown
    int cells_x  = hog.getWinSize().width/hog.getCellSize().width;
    int cells_y  = hog.getWinSize().height/hog.getCellSize().height;
    int blocks_x = hog.blocks_x();
    int blocks_y = hog.blocks_y();
    int cellsPerBlock = hog.cellsPerBlock();
    
    if(opts.check_errors && T.size() != nbins*cellsPerBlock*blocks_x*blocks_y)
    {
      printf("descriptorSize = %d\n",(int)hog.getDescriptorSize());
      printf("T.size() = %d\n",(int)T.size());
      printf("nbins = %d\n",nbins);
      printf("cellsPerBlock = %d\n",cellsPerBlock);
      printf("blocks_x = %d\n",blocks_x);
      printf("blocks_y = %d\n",blocks_y);
      assert(T.size() == nbins*cellsPerBlock*blocks_x*blocks_y);
    }
    
    auto at = [blocks_y,nbins,cellsPerBlock]
    (int blockX, int blockY, int cellN, int bin) -> int
    { 
      int idx = 
	blockX*(blocks_y*cellsPerBlock*nbins) + 
	blockY*(cellsPerBlock*nbins) + 
	cellN*nbins + 
	bin;
      
      return idx;
    };
    
    Mat VIS(viz_cell_size*cells_y,viz_cell_size*cells_x,DataType<float>::type,Scalar::all(0));
    for(int rIter = 0; rIter < blocks_y; rIter++)
      for(int cIter = 0; cIter < blocks_x; cIter++)
      {
	for(int eIter = 0; eIter < cellsPerBlock; eIter++)
	{
	  int cell_x, cell_y;
	  if(cellsPerBlock == 4)
	    map_blockToCell4(cIter,rIter,eIter,cell_x,cell_y);
	  else if(cellsPerBlock == 1)
	  {
	    cell_x = cIter;
	    cell_y = rIter;
	  }
	  else
	    assert(false);
	  assert(cell_x < cells_x && cell_y < cells_y);
	  Mat updateROI = VIS(Rect(Point(cell_x*viz_cell_size,cell_y*viz_cell_size),
			      Size(viz_cell_size,viz_cell_size)));	  
	  Mat update = Mat(viz_cell_size,viz_cell_size,DataType<float>::type,Scalar::all(0));
	  for(int bIter  = 0; bIter < nbins; bIter++)
	  {
	    double weight = T[at(cIter,rIter,eIter,bIter)];
	    update = deformable_depth::max(update,weight * glyphs[bIter]);
	  }
	  updateROI += update;
	}
      }    
      
    return VIS;
  }
  
  Mat picture_HOG(DepthFeatComputer& hog, std::vector< double > T,PictureHOGOptions opts)
  {
    FeatVis vis = picture_HOG_pn(hog,T,opts);
    return horizCat(vis.getPos(),vis.getPos());
  }
  
  void split_feat_pos_neg(const vector< double >& T, vector< double >& Tpos, vector< double >& Tneg)
  {
    
	for(size_t iter = 0; iter < T.size(); iter++)
	{
		const double& value = T[iter];

		if(value > 0)
		{
			Tpos.push_back(value);
			Tneg.push_back(0);
		}
		else
		{
			Tpos.push_back(0);
			Tneg.push_back(-value);
		}
	}
  }
  
  FeatVis picture_HOG_pn(DepthFeatComputer& hog, std::vector< double > T, PictureHOGOptions opts)
  {
    vector<double> Tpos, Tneg;
    split_feat_pos_neg(T,Tpos,Tneg);
  
    Mat posHOGvis = picture_HOG_one(hog,Tpos,opts);
    Mat negHOGvis = picture_HOG_one(hog,Tneg,opts);
    resize(posHOGvis,posHOGvis,Size(),5,5);
    posHOGvis = imagesc("",posHOGvis);
    resize(negHOGvis,negHOGvis,Size(),5,5);
    negHOGvis = imagesc("",negHOGvis);    
    return FeatVis("picture_HOG_pn",posHOGvis,negHOGvis);
  }  
  

  // we visualize HOG cells...
  Mat imagehog(std::string title, 
	       deformable_depth::DepthFeatComputer& hog, 
	       std::vector< double > T,
	       PictureHOGOptions opts)
  {
    Mat VIS = picture_HOG(hog,T,opts);
    
    // return
    return VIS;
  }
    
  void waitKey_safe(int delay)
  {
    // trick to speed things up?
    if(CS_WAIT_KEY_active && delay != 0)
      return;
    
    {
      unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
      CS_WAIT_KEY_active = true;
      //printf("CS: waitKey_safe\n");
      //printf("+waitKey_safe delay = %d\n",delay);
      deformable_depth::waitKey(delay);
      //printf("-waitKey_safe\n");
      CS_WAIT_KEY_active = false;
    }
  }
  
  Mat image_safe(string title, Mat im,bool resize)
  {
    {
      unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
      // all images should be about 640 by 480 :-)
      if(resize)
      {
	im = im.clone();
	double sf = std::sqrt((640*480.0)/(im.rows*im.cols));
	cv::resize(im,im,Size(),sf,sf,params::DEPTH_INTER_STRATEGY);      
      }
      deformable_depth::imshow(title,im);
      return im;
    }
  }
  
  // accutally reify the equalized image
  static Mat imageeq_realize(
    cv::Mat_< float > im,vector<float>&values)
  {
    Mat showMe(im.rows,im.cols,DataType<Vec3b>::type,Scalar(255,255,255));
    if(showMe.size().area() == 0)
      return showMe;
    if(values.size() < 2)
      return showMe;
    
    // discover the quantiles
    float oldQuant = 0;
    std::multimap<float,int> quantiles; // should containe 256 - 1 values
    for(int qIter = 1; qIter < 256; qIter++)
    {
      float quantile = static_cast<float>(qIter)/256;    
      int high_idx = clamp<int>(0,quantile*(values.size() - 1),values.size()-1);
      float threshold = values[high_idx];
      quantiles.insert(pair<float,int>(threshold,qIter));
	
      oldQuant = quantile;
    }    
    require_equal<int>(quantiles.size(),255);
    
    // write the image
    //printf("q = %f low = %f high = %f\n",quantile,thresh_low,thresh_high);
    for(int rIter = 0; rIter < im.rows; rIter++)
      for(int cIter = 0; cIter < im.cols; cIter++)
      {	
	float curValue = im.at<float>(rIter,cIter);	  
	
	if(std::isnan(curValue))
	  // NAN => Red
	  showMe.at<Vec3b>(rIter,cIter) = toVec3b(Colors::invalid());
	else if(curValue == inf)
	  // INF => Blue
	  showMe.at<Vec3b>(rIter,cIter) = toVec3b(Colors::inf());
	else if(curValue == -inf)
	  // -INF => Green
	  showMe.at<Vec3b>(rIter,cIter) = toVec3b(Colors::ninf());
	else
	{
	  auto quantile = quantiles.lower_bound(curValue); // equiv or after object
	  int qIter;
	  if(quantile == quantiles.end())
	    qIter = 255;
	  else
	    qIter = quantile->second - 1;
	  showMe.at<Vec3b>(rIter,cIter) = Vec3b(qIter,qIter,qIter);
	}
      }    
    
    return showMe;
  }
  
  Mat imageeq(const Mat&m)
  {
    Mat m1 = m.clone();
    return imageeq("",m1,false,false);
  }

  // darker means closer
  Mat imageeq(const char* winName, cv::Mat_< float > im, bool show, bool resize,bool verbose)
  {
    assert(im.type() == DataType<float>::type);
    // all images should be about 640 by 480 :-)
    im = im.clone();
    if(resize)
    {
      double sf = std::sqrt(static_cast<double>(640*480)/(im.rows*im.cols));
      //cout << "sf: " << sf << endl;
      //cout << "rows = " << im.rows << " cols = " << im.cols << endl;
      cv::resize(im,im,Size(),sf,sf,params::DEPTH_INTER_STRATEGY);
    }
    
    //printf("+imageeq\n");
    // compute the order statistics
    vector<float> values;
    for(int rIter = 0; rIter <im.rows; rIter++)
      for(int cIter = 0; cIter < im.cols; cIter++)
      {
	float value = im.at<float>(rIter,cIter);
	if(goodNumber(value) /*&& value < params::MAX_Z()*/)
	  values.push_back(value);
      }
    std::sort(values.begin(),values.end());
    auto newEnd = std::unique(values.begin(),values.end());
    values.erase(newEnd,values.end());
    
    // compute an equalized image
    Mat showMe = imageeq_realize(im,values);
    if(verbose and values.size() > 5)
    {
      ostringstream quantiles;
      quantiles << " " << values.front();
      for(int iter = 1; iter < 5; ++iter)
      {
	int index = static_cast<double>(iter)/5 * values.size();
	quantiles << " " << values.at(clamp<int>(0,index,values.size()-1));
      }
      quantiles << " " << values.back();
      showMe = vertCat(showMe,image_text(quantiles.str()));
    }
    else if(verbose)
      showMe = vertCat(showMe,image_text("BadQuantiles"));
    
    // safe to call from different threads?
    if(show)
    {
      { 
	unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
	//printf("CS: imageeq\n");
	deformable_depth::imshow(winName,showMe);
	//printf("-imageeq\n");
      }
    }
    
    return showMe;
  }
  
  Mat imagesc(const char* winName, Mat image, bool verbose,bool show)
  {
    Mat_<float> showMe; image.convertTo(showMe,DataType<float>::type);
    return imagesc(winName,showMe,verbose,show);
  }
 
  Mat imagesc(const char*winName,Mat_<float> image, bool verbose,bool show)
  {
    // pass 1 find min and max.
    float min, max;
    float inf = numeric_limits<float>::infinity();
    float ninf = -inf;
    minmax(image,min,max);
    if(verbose)
      cout << winName << " min = " << min << " max = " << max << endl;
      
    // PASS 2 create output image
    Mat showMe;
    cvtColor(image,showMe,CV_GRAY2RGB);
    showMe.convertTo(showMe,DataType<Vec3b>::type);
    for(int xIter = 0; xIter < image.cols; xIter++)
      for(int yIter = 0; yIter < image.rows; yIter++)
      {
	float v = image.at<float>(yIter,xIter);
	float out;
	if(v == inf)
	  out = 1;
	else if(v == ninf)
	  out = 0;
	else
	  out = (v-min)/(max-min);
	
	showMe.at<Vec3b>(yIter,xIter) = Vec3b(255*out,255*out,255*out);
      }
      
    // make the title
    string title = printfpp("%s (min,max)=(%f,%f)",winName,min,max);
    
    // show output image
    if(show)
    {
      {
	unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
	//printf("CS: imagesc\n");
	deformable_depth::imshow(winName,showMe);
	//waitKey(0);
      }
    }
    
    return showMe;
  }  
                  
  Mat imagefloat(string title, Mat image, bool show, bool resize)
  {
    Mat imsc = imagesc(title.c_str(),image,false,false);
    if(resize)
      imsc = imVGA(imsc,params::DEPTH_INTER_STRATEGY);
    Mat im = horizCat(imageeq(title.c_str(),image,false,resize),imsc);
    if(show)
      image_safe(title,im,resize);
    return im;
  }
  

  struct UserLabeledPoint
  {
  public:
    Point2i pt;
    bool*visible;
    bool set;
  public:
    UserLabeledPoint() : set(false) {};
  };
  
  void getPt_NOP(int event, int x, int y, int flags, void*data)
  {};
  
  void getPt_onMouse(int event, int x, int y, int flags, void*data)
  {
    UserLabeledPoint*pt = (UserLabeledPoint*)data;
    
    if(event != CV_EVENT_LBUTTONDOWN)
      return;
    else if(flags & CV_EVENT_FLAG_CTRLKEY)
    {
      printf("occlusion labeled\n");
      if(pt->visible != nullptr)
	*(pt->visible) = false;
    }
    else
      if(pt->visible != nullptr)
	*(pt->visible) = true;
    
    pt->set = true;
    pt->pt = Point2i(x,y);
  };
  
  Point2i getPt(string winName,bool*visible,char*abort_code)
  {
    UserLabeledPoint p;
    p.visible = visible;    
    char tmp; if(abort_code == nullptr) abort_code = &tmp;
    *abort_code = '\0';
    
    {
      unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
      setMouseCallback(winName,getPt_onMouse,&p);
      while(p.set == false && *abort_code == false)
      {
	int keycode = cv::waitKey(1);
	if(keycode != -1)
	{
	  *abort_code = (char)keycode;
	}
      }
      setMouseCallback(winName,getPt_NOP,&p);
    }
    
    return p.pt;
  }  
  
  struct UserGetRect
  {
  public:
    Mat&image; 
    string&name;
    bool set, mousedown;
    Point2i p1, p2;
    UserGetRect(Mat&im,string&nm) : mousedown(false),set(false),image(im), name(nm){};
  };
  
  void getRect_onMouse(int event, int x, int y, int flags, void*data)
  {
    //printf("getRect_onMouse\n");
    UserGetRect*state = (UserGetRect*)data;
    
    // update the box
    switch(event)
    {
      case CV_EVENT_LBUTTONDOWN:
	state->p1 = Point2i(x,y);
	//printf("setting p1\n");
	state->mousedown = true;
	return;
      case CV_EVENT_MOUSEMOVE:
	if(!state->mousedown)
	{
	  //printf("p1 not set\n");
	  return;
	}
	state->p2 = Point2i(x,y);
	break;
      case CV_EVENT_LBUTTONUP:
	state->p2 = Point2i(x,y);
	//printf("setting p2\n");
	state->mousedown = false;
	break;
      default:
	return;
    }
    
    // redraw
    state->set = true;
    Mat display = state->image.clone();
    rectangle(display,state->p1,state->p2,Scalar(255,0,0));
    deformable_depth::display(state->name,display);
    //printf("Redrawing BB\n");
  }
  
  Rect getRect(string winName, Mat bg, Rect init, bool allow_null, bool*set)
  {
    // scale to a cardinal size?
    double sf = std::sqrt(640.f*480.f/bg.size().area());
    resize(bg,bg,Size(),sf,sf);
    init = rectScale(init,sf);
    // init the dynamic state
    Mat display = deformable_depth::display("",bg,false,false);
    UserGetRect state(display,winName);
    
    {
      unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
      printf("+getRect\n");
            
      // display the rect and BG
      if(init != Rect())
      {
	printf("getRect: Using init\n");
	state.set = true;
	state.p1 = init.tl();
	state.p2 = init.br();
	printf("p1 = (%d, %d) p2 = (%d, %d)\n",
	      state.p1.y, state.p1.x, state.p2.y, state.p2.x);
	Mat canvas = display.clone();
	rectangle(canvas,state.p1,state.p2,Scalar(255,0,0));
	deformable_depth::display(state.name,canvas); deformable_depth::waitKey(1);
      }
      else
	deformable_depth::display(winName,bg);
      
      // wait for the rect
      setMouseCallback(winName,getRect_onMouse,&state);
      do
      {
	int key_code = deformable_depth::waitKey(0);
      } while (state.set == false && !allow_null);
      cout << "state.set = " << state.set << endl;
      
      destroyWindow(winName); // this unregisters the callback, I hope?
      printf("-getRect\n");
    }
    
    if(set)
      *set = state.set;
    return Rect(Point(state.p1.x/sf,state.p1.y/sf),
		Point(state.p2.x/sf,state.p2.y/sf));
  }
  
  void test_vis()
  {
    deformable_depth::imshow("String Vis",image_text("image text of string"));
    deformable_depth::waitKey_safe(0);
  }
  
  Mat rotate_colors(const Mat& orig, const Mat& rot)
  {
    assert(orig.type() == DataType<Vec3b>::type);
    Mat rotated = orig.clone();
    
    for(int rIter = 0; rIter < orig.rows; rIter++)
      for(int cIter = 0; cIter < orig.cols; cIter++)
      {
	Vec3b&color = rotated.at<Vec3b>(rIter,cIter);
	Mat cMat(3,1,DataType<double>::type);
	cMat.at<double>(0,0) = color[0];
	cMat.at<double>(1,0) = color[1];
	cMat.at<double>(2,0) = color[2];
	cMat = rot * cMat;
	color[0] = cMat.at<double>(0,0);
	color[1] = cMat.at<double>(1,0);
	color[2] = cMat.at<double>(2,0);
      }
      
    return rotated;
  }
  
  Mat replace_matches(const Mat& im0, const Mat& im1, const Mat& repl)
  {
    Mat result = repl.clone();
    
    Mat im0_blur = im0.clone();//cv::bilateralFilter(im0, im0_blur, -1, 2, 2);
    Mat im1_blur = im1.clone();//cv::bilateralFilter(im1, im1_blur, -1, 2, 2);
    
    for(int rIter = 0; rIter < im1.rows; rIter++)
      for(int cIter = 0; cIter < im1.cols; cIter++)
      {
	Vec3b pix0 = im0_blur.at<Vec3b>(rIter,cIter);
	Vec3b pix1 = im1_blur.at<Vec3b>(rIter,cIter);
	Vec3d diff = Vec3d(pix0) - Vec3d(pix1);
	double dist = std::sqrt(diff.dot(diff));
	//cout << "dist: " << dist << " between " << pix0 << " and " << pix1 << endl;
	if(dist >= 3)
	{
	  //cout << "replacing pixel" << endl;
	  result.at<Vec3b>(rIter,cIter) = pix0;
	}
      }
    
    return result;
  }
  
  Mat replace_matches(const Mat& im0, const Vec3b color, const Mat& repl)
  {
    Mat im1(im0.rows,im0.cols,DataType<Vec3b>::type,Scalar(color[0],color[1],color[2]));
    return replace_matches(im0,im1,repl);
  }
  
  Mat drawDets(
    MetaData& metadata, const DetectionSet& dets, int shape, int bg)
  {
    DetectionSet showDets;
    if(shape == 1 || dets.size() > 25)
    {
      showDets = nms_apx_w_array(dets,2000,metadata.load_im()->RGB.size());
      showDets = nms_w_list(showDets,.25);
    }
    else
      showDets = dets;
    //const DetectionSet showDets = nms_w_list(dets,.5);
    
    Mat vis = (bg==0)?
      metadata.load_im()->RGB.clone():
      imageeq("",metadata.load_im()->Z.clone(),false,false);
    // generate a color scheme
    vector<size_t> indexes = sorted_indexes(showDets); // draw low resp to high resp
    // draw the detections in a color scheme
    for(int iter = 0; iter < showDets.size(); ++iter)
    {
      int index = indexes[iter];
      DetectorResult det = showDets[index];
      double b = 255.0*iter/static_cast<double>(indexes.size());
      double r = 255 - b;
      
      if(det->resp >= -1)
      {
	//if(shape == 0)
	  rectangle(vis,det->BB.tl(),det->BB.br(),Scalar(b,0,r));
	//else
	  //circle(vis,rectCenter(det->BB),5,Scalar(b,0,r));
      }
    }
    // draw the ground truths in green
    for(auto pos : metadata.get_positives())
      rectangle(vis,pos.second.tl(),pos.second.br(),Scalar(0,255,0));
    return vis;
  }
  
  void logMissingPositives(MetaData&ex,const DetectionSet&dets)
  {
    map<string,double> correctnesses;
    map<string,Rect> closests;
    auto selected_gt = params::defaultSelectFn()(ex.get_positives());
    // compute the correctness
    for(const auto & det : dets)
    {
      // keep track of how well each gt can be explained by the
      // extracted windows. This should help avoid over pruning.
      for(auto & gt : selected_gt)
	if(gt.second.height > 30)
	{
	  if(correctnesses[gt.first] <= rectIntersect(gt.second,det->BB))
	  {
	    closests[gt.first] = det->BB;		
	  }
	  correctnesses[gt.first] = 
	    std::max<double>(
	      correctnesses[gt.first],rectIntersect(gt.second,det->BB));
	}      
    }
    
    // log the correctnesse
    ostringstream cor_log;
    cor_log << "cor_log " << ex.get_filename() << ":" << dets.size() << ": ";
    Mat missing_vis;
    for(auto & cor : correctnesses)
    {
      cor_log << cor.second << " ";
      if(cor.second < .90)
      {
	if(missing_vis.empty())
	{
	  //missing_vis = imageeq("",sample_im->Z.clone(),false,false);
	  missing_vis = ex.load_im()->RGB.clone();
	}
	rectangle(missing_vis,
		  selected_gt[cor.first].tl(),selected_gt[cor.first].br(),
		  Scalar(255,0,0));
	rectangle(missing_vis,
		  closests[cor.first].tl(),closests[cor.first].br(),
		  Scalar(0,0,255));	
      }
    }
    if(!missing_vis.empty())
      log_im("missing_vis",missing_vis);
    log_once(cor_log.str());     
  }

  ///
  /// SECTION: Progress Bars
  ///
  ProgressBar::~ProgressBar()
  {
    progressBars->set_progress(title,max,-1);
  }

  ProgressBar::ProgressBar(const std::string&title,int max) : title(title), max(max)
  {
    progressBars->set_progress(title,0,max);
  }

  void ProgressBar::set_progress(int index)
  {
    progressBars->set_progress(title,index,max);
  }

  ProgressBars*progressBars = new ProgressBars();
  static int window_id_counter = 0;

  class ProgressBars_Impl
  {
  protected:
    std::map<std::string,int> values;    
    std::map<std::string,int> totals;
    string window_name;    

  public:
    void update_progress_file()
    {
      cout << "***************************************" << endl;
      ofstream ofs(params::out_dir() + "/progress.txt");
      for(auto && value : values)
      {
	string title = value.first;
	double percentage = static_cast<double>(values[title])/totals[title];
	ofs << title << " [";
	cout << title << " ["; 
	for(double pIter = 0; pIter <= 1.00; pIter += .05)
	  if(pIter <= percentage)
	  {
	    ofs << "*";
	    cout << "*";
	  }
	  else
	  {
	    ofs << " ";
	    cout << " ";
	  }
	ofs << "] " << percentage << endl;
	cout << "] " << percentage << endl;
      }
      cout << "***************************************" << endl;
    }

    virtual void set_progress(const std::string&title,int value, int total)
    {
      if(value >= total - 1)
      {
	values.erase(title);
	totals.erase(title);
      }
      else if(values.find(title) == values.end())
      {	
	// add new trackbar
	values[title] = value;
	totals[title] = total;
      }
      else
      {
	// update existing trackbar
	values[title] = value;
	totals[title] = total;
      }     
      // 
      update_progress_file();
    }
  };

  class ProgressBars_GUI : public ProgressBars_Impl
  {
  protected:
  public:
    ProgressBars_GUI()  
    {
      if(have_display())
	create_window();      
    }

    void create_window()
    {
      if(have_display())
      {
	window_name = safe_printf("progress%d",window_id_counter);
	namedWindow(window_name,WINDOW_NORMAL | CV_WINDOW_FREERATIO);
      }
    }

    void recreate()
    {
      if(not have_display())
	return; 

      cv::destroyWindow(window_name);
      create_window();
      for(auto && trackbar : values)
      {
	cv::createTrackbar(trackbar.first,window_name,&values[trackbar.first],totals[trackbar.first]);
	cv::setTrackbarPos(trackbar.first,window_name,values[trackbar.first]);
      }
    }

    virtual void set_progress(const std::string&title,int value, int total) override
    {
      ProgressBars_Impl::set_progress(title,value,total);

      if(value >= total - 1)
      {
	// destroy the trackbar
	if(have_display())
	  recreate();
      }
      else if(values.find(title) == values.end())
      {	
	// add new trackbar
	if(have_display())
	{
	  cv::createTrackbar(title,window_name,&values[title],totals[title]);
	  cv::setTrackbarPos(title,window_name,values[title]);	
	}
      }
      else
      {
	// update existing trackbar
	if(have_display())
	{
	  cv::setTrackbarPos(title,window_name,values[title]);	
	}
      }     
    }
  };

  ProgressBars::ProgressBars() : 
    pimpl(new ProgressBars_Impl())
  {    
  }

  void ProgressBars::set_progress(const std::string&title,int value, int total)
  {
    //unique_lock<recursive_mutex> critc_sec(exclusion_high_gui);
    if(exclusion_high_gui.try_lock() == true)
    {      
      pimpl->set_progress(title,value,total);
      exclusion_high_gui.unlock();
    }
    else if(value >= total or value <= 1)
    {
      lock_guard<recursive_mutex> l(exclusion_high_gui);
      pimpl->set_progress(title,value,total); 
    }
  }

  Mat image_datum(MetaData&datum,bool write)
  {
    log_once(safe_printf("show_ICL_datum = %",datum.get_filename()));

    // load the image
    shared_ptr<const ImRGBZ> im = datum.load_im();
    Mat Zeq = imageeq("",im->Z,false,false);
    Mat viz = Zeq.clone();
    
    // draw the hand bb
    auto poss = datum.get_positives();
    Rect handBB = poss["HandBB"];
    cv::rectangle(viz,handBB.tl(),handBB.br(),Scalar(0,0,255));
    
    // draw the parts
    for(auto && part : poss)
      if(part.first != "HandBB")
	cv::rectangle(viz,part.second.tl(),part.second.br(),Scalar(255,0,0));
    
    // draw the segmentation
    Mat sem_seg = datum.getSemanticSegmentation();    

    // display
    Mat shown = image_safe("an ICL example",horizCat(viz,sem_seg));    
    waitKey_safe(20);

    // export?
    if(write)
    {
      static atomic<int> id_counter(0);
      int id = id_counter++;
      imwrite(safe_printf("%/seg_%.png",params::out_dir(),id),sem_seg);
      imwrite(safe_printf("%/depth_eq_%.png",params::out_dir(),id),Zeq);
    }

    return shown;
  }

  void line(Mat&mat,Point p1, Point p2, Scalar color1, Scalar color2)
  {
    LineIterator line_iter(mat, p1, p2, 8);
    int line_length = line_iter.count;
    for(int iter = 0; iter < line_length; ++iter, line_iter++)
    {
      Point pos = line_iter.pos();
      double w = iter/static_cast<double>(line_length - 1);
      double b = w*static_cast<double>(color1[0]) + (1-w)*static_cast<double>(color2[0]);
      double g = w*static_cast<double>(color1[1]) + (1-w)*static_cast<double>(color2[1]);
      double r = w*static_cast<double>(color1[2]) + (1-w)*static_cast<double>(color2[2]);
      if(0 <= pos.x and pos.x < mat.cols)
	if(0 <= pos.y and pos.y < mat.rows)
	  mat.at<Vec3b>(pos.y,pos.x) = Vec3b(
	    saturate_cast<uint8_t>(b),
	    saturate_cast<uint8_t>(g),
	    saturate_cast<uint8_t>(r));
    }
  }

  Mat monochrome(Mat&m,Vec3b chrome)
  {
    Mat vis = m.clone();
    for(int yIter = 0; yIter < vis.rows; ++yIter)
      for(int xIter = 0; xIter < vis.cols; ++xIter)
      {
	Vec3b&pix = vis.at<Vec3b>(yIter,xIter);
	if(pix == toVec3b(Colors::invalid()))
	  pix = BLUE;
	else
	{
	  double w = static_cast<double>(pix[0])/static_cast<double>(255);
	  pix = w*chrome;
	}
      }
    return vis;
  }

  static void dft_retile_quadrants(Mat&magI)
  {
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
  }
  
  // adapted from 
  // http://docs.opencv.org/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html?highlight=fourier
  Mat image_freqs(const Mat&m)
  {
    // calc the DFT
    Mat planes[] = {Mat_<float>(m), Mat::zeros(m.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // get magnitude
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    return vertCat(eq(magI),image_text("magnitudes"));
  }
}
 
