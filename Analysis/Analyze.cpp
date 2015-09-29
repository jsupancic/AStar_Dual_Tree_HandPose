/**
 * Copyright 2013: James Steven Supancic III
 **/
 
#include "Analyze.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include "PXCSupport.hpp"
#include "MetaData.hpp"
#include <boost/regex.hpp>
#include <boost/graph/graph_concepts.hpp>
#include "Log.hpp"
#include "Detector.hpp"
#include "Eval.hpp"
#include "ONI_Video.hpp"
#include "Video.hpp"
#include "Orthography.hpp"
#include "FingerPoser.hpp"
#include "ConvexityDefects.hpp"
#include "Scoring.hpp"
#include <boost/algorithm/string.hpp>
#include "TestModel.hpp"

#if defined(DD_ENABLE_HAND_SYNTH)

namespace deformable_depth 
{
  using namespace std;
  using namespace cv;
  
  map<string,PerformanceRecord > analyze()
  {
    // get the parameters
    map<string,string> pbm_dirs = g_params.matching_keys(".*Ours_Pose.*");
    
    // anyalize the still image pose accuracy
    vector<string> people = 
      {"James","Sam","Golnaz","Yi",
	"Vivian","Xiangxin","Bailey","Raul"};
    //vector<string> people{"HighResFace"};
    PerformanceRecord pxc_cm_displacements;
    map<string,PerformanceRecord > pbm_cm_displacements;
    TaskBlock analyze_people("analyze_people");
    for(string&person : people)
    {
      analyze_people.add_callee([&,person]()
      {
	// score the PBM
	static mutex m; 
	for(auto&&pbm_test_dir : pbm_dirs)
	{
	  auto pbm_results = analyze_pbm(pbm_test_dir.second,person);
	  // print the detection f1:
	  unique_lock<mutex> l(m);
	  PerformanceRecord & cm_disp_for_person = pbm_cm_displacements[pbm_test_dir.first];
	  cm_disp_for_person.merge(std::move(pbm_results));
	}
	
	// score the PXC
	PerformanceRecord new_pxc_displs = analyze_pxc(person);		
	
	// do a synchronized insertion
	unique_lock<mutex> l(m);
	pxc_cm_displacements.merge(std::move(new_pxc_displs));
      });
    }
    analyze_people.execute();
    
    // visualize the result
    FingerErrorCumulativeDist error_plot(params::out_dir() + "/cum_errors.py");
    for(auto && pbm_result : pbm_cm_displacements)
      error_plot.add_error_plot(pbm_result.second.getFingerErrorsCM(),pbm_result.first);
    error_plot.add_error_plot(pxc_cm_displacements.getFingerErrorsCM(),"PXC");
    
    // generate the percision-recall curve
    // TODO
    
    return pbm_cm_displacements;
  }
  
  void write(string filename, map< string, PerformanceRecord > data)
  {
    ofstream out(filename);
    assert(out.is_open());
    
    assert(false); // unimplemented
    
    out.close();
  }
  
  vector<string> default_videos(int num = -1)
  {
    vector<string> videos = test_video_filenames();
    
    if(num > 0)
      return vector<string>(videos.begin(),videos.begin()+num);
    else
      return videos;
  }
  
  string find_label_file(string directory, string video_path)
  {
    string video_title = boost::regex_replace(video_path,boost::regex(".*/"),"/");
    video_title = boost::regex_replace(video_path,boost::regex(".oni.yml.gz"),"");
    string regex_str = safe_printf(".*%.*(yml|txt)",video_title);
    // TODO boost::regex::icase breaks UCI dataset but fixes NYU dataset?    
    vector<string> matches = find_files(directory, boost::regex(regex_str,boost::regex::icase));    
    if(matches.size() > 1 or matches.size() <= 0)
    {
      ostringstream oss;
      oss << directory << ":" << video_title << "\t" << video_path << "\t\"" << regex_str << "\"\twarning: matches.size " << matches.size() << " != 1" << endl;
      for(auto && match : matches)
	oss << "match: " << match << endl;     
      cout << oss.str();
      log_file << oss.str();
      assert(false);
    }

    return matches.at(0);
  }

  void generate_test(vector<string>&videos,string stem, string title,vector<BaseLineTest>&tests)
  {    
    vector<string> label_files;
    for(string&video_filename : videos)
    {
      string vid_name = load_video(video_filename)->get_name();
      string label_filename = find_label_file("/home/jsupanci/Dropbox/out/" + stem + "/",vid_name);
      log_file << "generate_test label_filename = " << label_filename << endl;
      if(not boost::filesystem::exists(label_filename))
      {	  
	string err_msg = safe_printf("file not found %",label_filename);
	cout << err_msg << endl;
	throw std::runtime_error(err_msg);
      }
      log_once(string("will open ") + label_filename);
      label_files.push_back(label_filename);
    }
    
    BaseLineTest Ours_Test{videos,label_files,
	title,
	PerformanceRecord(),
	true,true,true,true
	};
    tests.push_back(Ours_Test);    
  }

  void default_tests_ours(vector<string> videos,vector<BaseLineTest>&tests)
  {
    cout << "+default_tests_ours" << endl;
    map<string,string> test_models = g_params.matching_keys("TEST_MODEL.*");    
    cout << "test_models.size() = " << test_models.size() << endl;
    for(auto pair : test_models)
    {      
      auto split_point = pair.second.find(":");
      string file_id(pair.second.begin(),pair.second.begin()+split_point);
      string title(pair.second.begin()+split_point+1,pair.second.end());
      generate_test(videos,file_id,title,tests);
    }  
    cout << "-default_tests_ours" << endl;
  }  
      
  vector<BaseLineTest> default_tests()
  {
    vector<string> videos = default_videos();
    vector<BaseLineTest> tests;

    auto test_battery_ids = g_params.matching_values("TEST_BATTERY.*");
    if(test_battery_ids.find("baselines") != test_battery_ids.end())
    {
      generate_test(videos,"2014.09.25-FORTH","FORTH",tests);
      generate_test(videos,"2014.09.25-RC","RC",tests);
      generate_test(videos,"2014.09.25-PXC","PXC",tests);
      generate_test(videos,"2014.09.25-NiTE2","NiTE2",tests);
    }
    if(test_battery_ids.find("ours") != test_battery_ids.end())
      default_tests_ours(videos,tests);
    
    /// 
    log_file << "===============" << endl;
    for(auto && vid : videos)
      log_file << "default_test using video: " << vid << endl;
    log_file << "===============" << endl;

    return tests;
  }
  
  void generate_finger_hand_detection_plots(vector<BaseLineTest>&tests)
  {
    // generate the ROC and percision-recall curve for fingers and hands 
    unique_ptr<PrecisionRecallPlot> pr_finger_plot(
      new PrecisionRecallPlot(params::out_dir() + "/pr_finger_plot.m","finger"));
    unique_ptr<PrecisionRecallPlot> pr_hand_plot(
      new PrecisionRecallPlot(params::out_dir() + "/pr_hand_plot.m","hand"));
    unique_ptr<PrecisionRecallPlot> pr_finger_plot_ablative(
      new PrecisionRecallPlot(params::out_dir() + "/pr_finger_plot_ablative.m","finger"));
    unique_ptr<PrecisionRecallPlot> pr_hand_plot_ablative(
      new PrecisionRecallPlot(params::out_dir() + "/pr_hand_plot_ablative.m","hand")); 
    unique_ptr<PrecisionRecallPlot> pr_joint_pose(
      new PrecisionRecallPlot(params::out_dir() + "/pr_joint_pose.m","joint_pose")); 
    unique_ptr<PrecisionRecallPlot> pr_finger_agnostic(
      new PrecisionRecallPlot(params::out_dir() + "/pr_finger_agnostic.m","finger_agnostic")); 
    vector<unique_ptr<PrecisionRecallPlot>*> plots{
      &pr_finger_plot,&pr_hand_plot,&pr_finger_plot_ablative,&pr_hand_plot_ablative,&pr_joint_pose,
	&pr_finger_agnostic};

    for(BaseLineTest & test : tests)
    {
      if(test.curve)
      {
	// pr comparative
	if(test.comparative)
	{
	  pr_hand_plot->add_plot(test.record,test.title);
	  pr_finger_plot->add_plot(test.record,test.title);
	  pr_joint_pose->add_plot(test.record,test.title);
	  pr_finger_agnostic->add_plot(test.record,test.title);
	}
	if(test.ablative)
	{
	  pr_hand_plot_ablative->add_plot(test.record,test.title);
	  pr_finger_plot_ablative->add_plot(test.record,test.title);	  
	}
      }
      else
      {
	// setup the comparative analysis plot
	if(test.comparative)
	{
	  pr_finger_agnostic->add_point(test.title,test.record);
	  pr_hand_plot->add_point(test.title,test.record);
	  pr_finger_plot->add_point(test.title,test.record);
	  if(test.eval_joint_pose)
	  {
	    pr_joint_pose->add_point(test.title,test.record);	    
	  }
	}
	if(test.ablative)
	{
	  pr_hand_plot_ablative->add_point(test.title,test.record);	
	  pr_finger_plot_ablative->add_point(test.title,test.record);	  
	}
      }
    }    

    TaskBlock gen_plots("gen_plots");
    for(auto plot_ptr_ptr : plots)
      gen_plots.add_callee([&,plot_ptr_ptr]()
			   {
			     plot_ptr_ptr->reset(nullptr);
			   });
    gen_plots.execute();
  }
  
  void score_tests(vector<BaseLineTest>&tests)
  {
    TaskBlock analyze_video_task("analyze_video_task");
    for(int iter = 0; iter < tests.size(); ++iter)
    {
      analyze_video_task.add_callee([&,iter]()
      {
	auto & test = tests[iter];
	test.record = score_video(test);
	log_file << "scored_test: " << test.title << " # vids = " << test.videos.size() << 
	  " gt count = " << test.record.getGtCount() << endl;
      });
    }
    analyze_video_task.execute();    
  }
  
  static void dump_cm_errors_to_disk(vector<BaseLineTest>&tests)
  {
    double fl_corr = fromString<float>(g_params.require("FCL_COR"));
    
    for(BaseLineTest & test : tests)
    {
      // write the raw finger displacement errors
      string&title  = test.title;  // method name
      auto & record = test.record;
      ofstream ofs(params::out_dir() + "/" + title + "finger_errors_cm.csv");
      write_vec(ofs,record.getFingerErrorsCM()/fl_corr);
      ofs << endl;

      //data[test.title] = test.record;
    }
    //write(params::out_dir() + "/vid_data.py",data);
  }

  // typically run as:
  // rm -v ./out/* ; make -j16 && gdb -ex 'catch throw' -ex 'r analyze_video CFG_FILE=scripts/hand.cfg' ./deformable_depth
  void analyze_video()
  {    
    // configure the tests
    vector<BaseLineTest> tests = default_tests();
    log_file << "analyze_video: got tests # = " << tests.size() << endl;

    // get the results
    score_tests(tests);
      
    // Generate the cumulative error plot
    FingerErrorCumulativeDist error_plot(params::out_dir() + "/video_errors.py");
    for(BaseLineTest & test : tests)
    {
      error_plot.add_error_plot(test.record.getFingerErrorsCM(),test.title);
    }
    
    // generate some curves
    generate_finger_hand_detection_plots(tests);
    
    // dump the raw data to disk for later analysis
    //map<string,PerformanceRecord> data;
    dump_cm_errors_to_disk(tests);
	
    cout << "Done Analyzing Results" << endl;
    log_file << "Done Analyzing Results" << endl;
  }
  
  void analyze_anytime()
  {
    // define the tests
    vector<string> videos = default_videos();
    vector<string> subdirs = {"13","52","156","780","3900"};
    vector<BaseLineTest> tests;
    string prefix = "/home/jsupanci/Dropbox/out/2013.11.30-ANYTIME/";
    for(string subdir : subdirs)
    {
      BaseLineTest anytime_baseline{videos,
	{
	  prefix + subdir + "/james.yml.gz.DD.yml",
	  prefix + subdir + "/sam.yml.gz.DD.yml",
	  prefix + subdir + "/greg.yml.gz.DD.yml",
	  prefix + subdir + "/dennis.yml.gz.DD.yml"
	},
	subdir,
	PerformanceRecord(),
	true,true,true,false};
      tests.push_back(anytime_baseline);
    }
    
    // compute the scores.
    score_tests(tests);
    
    // generate the plots
    generate_finger_hand_detection_plots(tests);
    
    // report completion
    cout << "DONE: analyze_anytime" << endl;
  }
  
  void analyze_egocentric()
  {
    // define the tests
    vector<string> videos{
      //"/home/jsupanci/Dropbox/data/yml_videos/Greg.yml.gz",
      //"/home/jsupanci/Dropbox/data/yml_videos/Marga.yml.gz"
      "/home/jsupanci/Dropbox/data/depth_video/egocentric/Greg",
      "/home/jsupanci/Dropbox/data/depth_video/egocentric/Marga"
    };
    vector<BaseLineTest> tests;
    
    // PXC Test
    string prefix = "/home/jsupanci/workspace/deformable_depth/data/tracks/";
    BaseLineTest pxc_baseline{videos,
      {
	prefix + "/GregPXCDetections.yml",
	prefix + "/MargaPXCDetections.yml"
      },
      "PXC",
      PerformanceRecord(),
      true,true,true,false};
    tests.push_back(pxc_baseline);
    
    // BaseLineTest RF_Keskin
    // {
    //   videos,
    //   {
    // 	"/home/jsupanci/Dropbox/out/2014.10.17-Keskin-all/Greg.yml.gz.DD.yml",
    // 	  "/home/jsupanci/Dropbox/out/2014.10.17-Keskin-all/Marga.yml.gz.DD.yml"
    //   },
    //   "RFKeskin",
    //   PerformanceRecord(),
    //   true,true,true,false
    // };
    // tests.push_back(RF_Keskin);

    // BaseLineTest RF_Xu
    // {
    //   videos,
    //   {
    // 	"/home/jsupanci/Dropbox/out/2014.10.17-Xu-all/Greg.yml.gz.DD.yml",
    // 	  "/home/jsupanci/Dropbox/out/2014.10.17-Xu-all/Marga.yml.gz.DD.yml"
    //   },
    //   "RFXu",
    //   PerformanceRecord(),
    //   true,true,true,false
    // };
    // tests.push_back(RF_Xu);

    BaseLineTest RF_XuNoGt
    {
      videos,
      {
	"/home/jsupanci/Dropbox/out/2014.10.19-XuNoGtBB/Greg.yml.gz.DD.yml",
	  "/home/jsupanci/Dropbox/out/2014.10.19-XuNoGtBB/Marga.yml.gz.DD.yml"
      },
      "RFXu-IndSkin",
      PerformanceRecord(),
      true,true,true,false
    };
    tests.push_back(RF_XuNoGt);

    BaseLineTest RF_KeskinNoGt
    {
      videos,
      {
	"/home/jsupanci/Dropbox/out/2014.10.19-KeskinNoGtBB/Greg.yml.gz.DD.yml",
	  "/home/jsupanci/Dropbox/out/2014.10.19-KeskinNoGtBB/Marga.yml.gz.DD.yml"
      },
      "RFKeskin-IndSkin",
      PerformanceRecord(),
      true,true,true,false
    };
    tests.push_back(RF_KeskinNoGt);


    BaseLineTest Kitani
    {
      videos,
      {
	"/home/jsupanci/Dropbox/out/2014.11.10-Kitani/Greg.yml.gz.DD.yml",
	  "/home/jsupanci/Dropbox/out/2014.11.10-Kitani/Marga.yml.gz.DD.yml"
      },
      "Kitani",
      PerformanceRecord(),
      true,true,true,false
    };
    tests.push_back(Kitani);

    // BaseLineTest NN
    // {
    //   videos,
    //   {
    // 	"/home/jsupanci/Dropbox/out/2014.10.27-NN-EGOCENTRIC/Greg.yml.gz.DD.yml",
    // 	  "/home/jsupanci/Dropbox/out/2014.10.27-NN-EGOCENTRIC/Marga.yml.gz.DD.yml"
    //   },
    //   "NN",
    //   PerformanceRecord(),
    //   true,true,true,false
    // };
    // tests.push_back(NN);

    // BaseLineTest NNSynth
    // {
    //   videos,
    //   {
    // 	"/home/jsupanci/Dropbox/out/2014.10.30-Synth-Ego/Greg.yml.gz.DD.yml",
    // 	  "/home/jsupanci/Dropbox/out/2014.10.30-Synth-Ego/Marga.yml.gz.DD.yml"
    //   },
    //   "NNSynth",
    //   PerformanceRecord(),
    //   true,true,true,false
    // };
    // tests.push_back(NNSynth);

    // BaseLineTest NNICL
    // {
    //   videos,
    //   {
    // 	"/home/jsupanci/Dropbox/out/2014.11.01-NN-ICLTrain-4Test/Greg.yml.gz.DD.yml",
    // 	  "/home/jsupanci/Dropbox/out/2014.11.01-NN-ICLTrain-4Test/Marga.yml.gz.DD.yml"
    //   },
    //   "NNICL",
    //   PerformanceRecord(),
    //   true,true,true,false
    // };
    // tests.push_back(NNICL);

    // BaseLineTest NNNYU
    // {
    //   videos,
    //   {
    // 	"/home/jsupanci/Dropbox/out/2014.10.31-NN-NYUTrain-4Test/Greg.yml.gz.DD.yml",
    // 	  "/home/jsupanci/Dropbox/out/2014.10.31-NN-NYUTrain-4Test/Marga.yml.gz.DD.yml"
    //   },
    //   "NNNYU",
    //   PerformanceRecord(),
    //   true,true,true,false
    // };
    // tests.push_back(NNNYU);
    
    // Cascade Exp 1
    //vector<int> values = {1,2,3,4,12,13,14,15,16};
    vector<int> values = {1,16,144,150};
    for(int iter : values)
    {
      BaseLineTest cascade_baseline{videos,
	{
	  prefix + printfpp("/Results-Cascades-Exp%d-Greg.txt",iter),
	  prefix + printfpp("/Results-Cascades-Exp%d-Marga.txt",iter)
	},
	printfpp("Cascade Exp %d",iter),
	PerformanceRecord(),
	true,true,true,false};
      tests.push_back(cascade_baseline);
    }
    
    // 2014 cascade experiments 
    auto addCurve = [&](string greg, string marga, string title)
    {
      BaseLineTest test{videos,
	{
	  prefix + greg,
	  prefix + marga
	},
	title,
	PerformanceRecord(),
	true,true,true,false};
        tests.push_back(test);      
    };
    // addCurve(
    //   "PR-Cascades-Greg-2014-Egocentric prior-Exp148.txt",
    //   "PR-Cascades-Marga-2014-Egocentric prior-Exp148.txt",
    //   "2014 Egocentric Exp148"
    // );
    // addCurve(
    //   "PR-Cascades-Greg-2014-Generic prior-Exp150.txt",
    //   "PR-Cascades-Marga-2014-Generic prior-Exp150.txt",
    //   "2014 Generic Exp150"
    // );
    // addCurve(
    //   "PR-Cascades-Greg-2014-Viewpoint prior-Exp151.txt",
    //   "PR-Cascades-Marga-2014-Viewpoint prior-Exp151.txt",
    //   "2014 Viewpoint Prior Exp151");
    // addCurve(
    //   "PR-Cascades-Greg-Egocentric prior-Exp148.txt",
    //   "PR-Cascades-Marga-Egocentric prior-Exp148.txt",
    //   "Egocentric Exp148"
    // );
    // addCurve(
    //   "PR-Cascades-Greg-Generic prior-Exp150.txt",
    //   "PR-Cascades-Marga-Generic prior-Exp150.txt",
    //   "Generic Exp150");
    // addCurve(
    //   "PR-Cascades-Greg-Viewpoint prior-Exp151.txt",
    //   "PR-Cascades-Marga-Viewpoint prior-Exp151.txt",
    //   "Viewpoint Exp151");
    
    // compute the scores.
    score_tests(tests);
    
    // generate the plots
    generate_finger_hand_detection_plots(tests);
    
    dump_cm_errors_to_disk(tests);

    // report completion
    cout << "DONE: analyze_egocentric" << endl;
  }
  
  void regress_finger_conf()
  {
    // setup the QP
    QP learner(.00001);
    learner.prime(2);
    
    // load a hold out set to train this model...
    // we are ultimately evaluating on the videos not images so this 
    // is ok.
    map<string,PerformanceRecord> pbm_performance = analyze();
    for(auto && record : pbm_performance)
    {
      // get our finger detections
      string title = record.first;
      PerformanceRecord performance = record.second;      
      // TODO : broken
      assert(false);
      
//       // write each of our detections
//       for(FrameDet & det : finger_dets)
//       {
// 	if(!det.is_detection())
// 	  continue;
// 	
// 	double y = det.correct()?+1.0:-1.0;
// 	vector<double> x = {det.score(),det.subscore()};
// 	cout << "write x: " << x << endl;
// 	learner.write(x ,y,uuid());
//       }
    }
    
    // optimize the qp and print the w vector
    cout << printfpp("qp cache size = %d",(int)learner.cache_size()) << endl;
    learner.opt();
    cout << "w: " << learner.getW() << endl;
    cout << "b: " << learner.getB() << endl;
  }
  
  void export_responces()
  {
    string comp_file_name = g_params.require("COMP_FILE");
    string resp_prefix = g_params.require("RESP_PREFIX");
    string resp_suffix = g_params.require("RESP_SUFFIX");
    string method_name = g_params.require("METHOD");
    map<string,vector<BaselineDetection> > baseline_tracks;
    
    ifstream ifs(comp_file_name,ifstream::in);
    while(ifs.good())
    {
      string line; getline(ifs,line);
      boost::algorithm::trim(line);
      if(!line.empty())
      {
	istringstream ls(line);
	// home_office1,1000,forth,lost track
	string video; getline(ls,video,',');
	string frame; getline(ls,frame,',');
	string method; getline(ls,method,',');
	string cause; getline(ls,cause,',');
	string resp; getline(ls,resp,',');
	
	// get the responces
	if(method == method_name)
	{
	  if(baseline_tracks.find(video) == baseline_tracks.end())
	  {
	    string baseline_filename = resp_prefix + video + resp_suffix;
	    baseline_tracks[video] = loadBaseline(baseline_filename,-1);
	  }
	  double resp = baseline_tracks[video][fromString<int>(frame)].resp;
	  cout << video << "," << frame << "," << method << "," << cause << "," << resp << endl;
	}
	else
	  cout << video << "," << frame << "," << method << "," << cause << "," << resp << endl;
      }
      else
	cout << endl;
    }
  }
}
#else
namespace deformable_depth
{
  map<string,PerformanceRecord >  analyze() { throw std::logic_error("unsupported");}
  void export_responces(){ throw std::logic_error("unsupported");}
  void analyze_video(){ throw std::logic_error("unsupported");}
  void analyze_anytime(){ throw std::logic_error("unsupported");}
  void analyze_egocentric(){ throw std::logic_error("unsupported");}
  void regress_finger_conf(){ throw std::logic_error("unsupported");}
}
#endif
