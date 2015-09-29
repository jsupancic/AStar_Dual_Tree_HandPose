/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_ASTAR_NN
#define DD_ASTAR_NN

#include "Detection.hpp"
#include "HeuristicTemplates.hpp"
#include <queue>
#include <atomic>
#include <unordered_map>
#include "Cache.hpp"
#include "AStarTree.hpp"

namespace deformable_depth
{
  typedef long TemplateId;    

  struct RotatedWindow
  {
  public:
    DetectorResult window;
    double theta;

    bool operator==(const RotatedWindow&cmpTo) const
    {
      return window == cmpTo.window and theta == cmpTo.theta;
    }
  };
}

namespace std 
{
  template <>
  struct hash < pair<deformable_depth::TemplateId/*X*/,const deformable_depth::SearchTree*/*T*/> >
  {
  public :
    size_t operator()(const pair<deformable_depth::TemplateId/*X*/,const deformable_depth::SearchTree*/*T*/> &x ) const;
  };
  template <>
  struct hash < deformable_depth::RotatedWindow >
  {
  public :
    size_t operator()(const deformable_depth::RotatedWindow &x ) const;
  };
}

namespace deformable_depth
{
  // 
  struct InformedSearch
  {
  public:
    typedef function<shared_ptr<NNTemplateType> ()> XFn;
    double admissibility; 

  protected:
    // SECTION:TYPES
    struct Correlation
    {
      double exact, approx;
      operator double() const {return approx;};
    };   
    typedef function<Mat ()> FrameFn;

    // SECTION: General member variables recording search progress
    Cache<shared_ptr<NNTemplateType> > x_template_cache;
    const map<string,map<string,AnnotationBoundingBox> >&parts_per_ex;
    shared_ptr<MetaData> hint;
    Vec3d gt_up, gt_normal, gt_center;
    bool fliplr;
    const ApxNN_Model&model;
    long template_evals, boundings, rejected_dets, iterations, last_frame_made;
    DetectionSet windows;
    vector<shared_ptr<XFn> > XS;        
    int DEBUG_invalid_Xs_from_frontier;
    const ImRGBZ&im;
    double upper_bound;
    unordered_map<size_t,ImRGBZ> rotations;
    std::chrono::time_point<std::chrono::system_clock,std::chrono::nanoseconds> progress_last_reported;
    vector<FrameFn> debug_video; // nice visualization

    // maps used for pyramid shared heuristics    
    unordered_multimap<TemplateId,RotatedWindow> template_to_parent_window;
    unordered_map<RotatedWindow,TemplateId> window_to_template_id;
    unordered_map<TemplateId,RotatedWindow> template_id_to_window;
    unordered_map<pair<TemplateId/*X*/,const SearchTree*/*T*/>, double> heuristic_table;

    // SECTION:METHODS
    double bubble_correlation(long x0, long x_index,double cor);
    double calc_correlation_parents(long x0, long x_index, const SearchTree*T);
    double calc_correlation_pyramid(long x_index, const SearchTree*T);
    Correlation calc_correlation_aobb(long x_index, const SearchTree*T);
    Correlation calc_correlation(long x_index, const SearchTree*T);
    Correlation calc_correlation_simple(long x_index, const SearchTree*T);

    void update_random_upper_bound(const SearchTree*tree,NNTemplateType&XTempl);
    DetectorResult find_best(const AStarSearchNode&active,RotatedRect&rawWindow);
    bool emit_best(const AStarSearchNode&active,DetectionSet&dets);
    // log progress as we go along...
    virtual void write_progress(bool done);

    void implement_image_template(DetectorResult&window,double theta);
    double extract_image_template_random(const ImRGBZ& im,DetectorResult&window);
    double extract_image_template_directed(const ImRGBZ& im,DetectorResult&window,size_t theta_index);
    void extract_image_template(const ImRGBZ& im,DetectorResult&window);

  public:
    void init(DetectionFilter&filter);

    InformedSearch(const ImRGBZ& im,const ApxNN_Model&model,DetectionFilter&filter);    
    virtual ~InformedSearch();
    // exapnd the min-cost node from the frontier and add its
    // children back into the frontier or terminate the search
    // if it is a terminal node.
    virtual bool iteration(DetectionSet&dets) = 0;
    virtual void init_frontier(const AStarSearchNode&node) = 0;
    //
    shared_ptr<NNTemplateType> getX(TemplateId id);    
    size_t numberXS(); 
    bool suppressedX(TemplateId id);
  };

  // represents an invokation of an A* search
  struct AStarSearch : public InformedSearch
  {
  protected:
    std::priority_queue<AStarSearchNode> frontier;
    
    void iteration_admissible(shared_ptr<NNTemplateType>&XTempl,const AStarSearchNode&active);
    void iteration_inadmissible(shared_ptr<NNTemplateType>&XTempl,const AStarSearchNode&active);
    void make_frame(const AStarSearchNode&active);
    virtual void write_progress(bool done) override;

  public:
    AStarSearch(const ImRGBZ& im,const ApxNN_Model&model,DetectionFilter&filter);    
    virtual ~AStarSearch();
    // exapnd the min-cost node from the frontier and add its
    // children back into the frontier or terminate the search
    // if it is a terminal node.
    virtual bool iteration(DetectionSet&dets) override;
    virtual void init_frontier(const AStarSearchNode&node) override;
  };  

  //SMAStarSearch AStarSearch
  typedef AStarSearch DefaultSearchAlgorithm;

  // represents a node in an active A* search
  struct AStarSearchNode
  {
    // f cost, estimate of reaching solution from here
    double cost;
    // the tree node which this node corresponds to.
    const SearchTree*tree;
    double depth;
    size_t X_index;
    
    bool operator< (const AStarSearchNode&other) const
    {
      return cost > other.cost; // lowest cost comes last in sorted order
    };
  };
}

#endif


