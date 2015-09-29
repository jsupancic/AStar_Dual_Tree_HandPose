/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_ASTAR_TREE
#define DD_ASTAR_TREE

#include <atomic>
#include <unordered_map>
#include "HeuristicTemplates.hpp"

namespace deformable_depth
{
  class ApxNN_Model;

  using std::atomic;
  using std::unordered_map;

  class SearchTree;
  typedef function<double (const SearchTree&,const NNTemplateType&)> NodeHeuristicFn;

  class MetaData;

  class SecondOrderRandomTree;
  class SearchTree
  {
  protected:
    // the upper bound of the children
    NNTemplateType Templ;
    vector<shared_ptr<SearchTree> > children;
    int id;
    string pose_uuid;        
    // one or more trained heuristics for this node.
    // these optimistically estimate cost. 
    vector<shared_ptr<NodeHeuristicFn> > heuristics;
    
    friend class ApxNN_Model;
    friend struct AStarSearch;
    friend class InformedSearch;
    friend class SMAStarSearch;
    
  public:
    bool operator== (const SearchTree&other) const;
    SearchTree(NNTemplateType&templ);
    SearchTree(Vec3i templ_size = Vec3i(SpearTemplSize.width,SpearTemplSize.height,NN_DEFAULT_ZRES));
    string get_uuid() const;

    friend void merge_clusters_agglom
    (atomic<int>&seq_id,list<SearchTree >&clusters,
	     unordered_map<int, unordered_map<int,double> >&costs,
	     function<void (SearchTree&node)>&update_dists_for_node);
    friend SearchTree train_search_tree_agglomerative(map<string,NNTemplateType>&allTemplates);
    friend SearchTree train_search_kmeans(map<string,NNTemplateType>&allTemplates,int depth = 0);
    friend SearchTree train_search_tree_pyramid(map<string,NNTemplateType>&allTemplates,bool translational = false);
    friend SearchTree train_search_tree(map<string,NNTemplateType>&allTemplates);
    
    // serialization functions
    friend void log_search_tree(SearchTree&tree_root); // save for visualization
    friend void save_search_tree(SearchTree&tree_root); // save for later Loading
    friend void write(cv::FileStorage&, std::string&, const shared_ptr<deformable_depth::SearchTree>&);
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::SearchTree&);
    friend void read(const cv::FileNode&, shared_ptr<deformable_depth::SearchTree>&, shared_ptr<deformable_depth::SearchTree>);
    friend void read(const cv::FileNode&, deformable_depth::SearchTree&, deformable_depth::SearchTree);

    // train the higher order decision forest  
    friend vector<shared_ptr<SecondOrderRandomTree> > init_hord_trees(SearchTree&root,int depth);
    friend void train_hord_heuristics(ApxNN_Model&,SearchTree&tree,vector<shared_ptr<MetaData> >&training_set);
    friend class SecondOrderRandomTree;
  };
  SearchTree train_search_tree(map<string,NNTemplateType>&allTemplates);

  struct AStarSearchNode;
  struct SMAStarSearchNode;
  struct SMAStarNodePtr;
}

#endif

