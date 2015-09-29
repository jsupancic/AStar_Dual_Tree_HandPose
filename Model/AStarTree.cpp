/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "AStarTree.hpp"
#include <atomic>
#include "ApxNN.hpp"
#include "RandomHoughFeature.hpp"
#include "Pointers.hpp"

namespace deformable_depth
{
  using namespace std;
  using namespace cv;

  ///
  /// SECTION: SearchTree
  ///
  bool SearchTree::operator==(const SearchTree& other) const
  {
    return Templ == other.Templ;
  }

  SearchTree::SearchTree(NNTemplateType& templ) : 
    Templ(templ)
  {
  }

  SearchTree::SearchTree(Vec3i templ_size) : 
    Templ(templ_size)
  {
  }
  
  string SearchTree::get_uuid() const
  {
    return pose_uuid;
  }

  ///
  /// SECTION: Train a tree using Hierarchical Agglomerative Clustering
  /// 

  void merge_clusters_agglom
    (atomic<int>&seq_id,list<SearchTree >&clusters,
     unordered_map<int, unordered_map<int,double> >&costs,
     function<void (SearchTree&node)>&update_dists_for_node)
  {
    while(clusters.size() > 1)
    {
      list<SearchTree> next_layer;
      while(!clusters.empty())
      {	
	std::atomic<double> min_cost(inf);
	SearchTree merged;
	
	if(clusters.size() > 1)
	{
	  // find the pair to merge
	  TaskBlock find_merge_pair("find_merge_pair");
	  for(auto & node_i : clusters)
	  {
	    auto ptr_i = &node_i;
	    find_merge_pair.add_callee([&,ptr_i]()
	    {
	      for(auto & node_j : clusters)
	      {
		auto ptr_j = &node_j;
		if(ptr_j->id == ptr_i->id)
		  continue;
		//ptr_i->Templ.merge_cost(ptr_j->Templ);
		double merge_cost = costs.at(ptr_i->id).at(ptr_j->id); 
		if(merge_cost < min_cost)
		{
		    static mutex m; lock_guard<mutex> l(m);
		    if(merge_cost < min_cost)
		    {
		      min_cost = merge_cost;
		      merged.Templ = ptr_i->Templ.merge(ptr_j->Templ);
		      merged.children = vector<shared_ptr<SearchTree> >{
			make_shared<SearchTree>(*ptr_i),
			make_shared<SearchTree>(*ptr_j)};
		      merged.id = seq_id++;
		    }
		  }
		}
	    });
	  }
	  find_merge_pair.execute();
	}
	else
	{
	  merged = clusters.front();
	  clusters.clear();
	}
	
	// commit the merged
	for(auto && child : merged.children)
	  clusters.remove(*child);
	next_layer.push_front(merged);
	log_once(printfpp("ApxNN_Model::train_search_tree %d",(int)clusters.size()));
      }
      log_once(printfpp("ApxNN_Model::train_search_tree next_layer %d",(int)next_layer.size()));
      clusters = next_layer;
      TaskBlock compute_update_dists("compute_update_dists");
      for(SearchTree&node : clusters)
      {
	SearchTree*np = &node;
	compute_update_dists.add_callee([&,np]()
	{
	  update_dists_for_node(*np);
	});
      }
      compute_update_dists.execute();
    }    
  }

  static atomic<int> seq_id(0);

  void save_search_tree(SearchTree&tree_root)
  {
    FileStorage fs(params::out_dir() + "/search_tree_" + uuid() + ".yml.gz" ,FileStorage::WRITE);
    
    fs << "tree_root" << tree_root;

    fs.release();
  }

  void write(cv::FileStorage&fs, const deformable_depth::SearchTree&node)
  {
    string s;
    write(fs,s,node);
  }

  void read(const cv::FileNode&file, shared_ptr<deformable_depth::SearchTree>& node, shared_ptr<deformable_depth::SearchTree>)
  {
    SearchTree default_value;
    node = make_shared<SearchTree>();
    read(file,*node,default_value);
  }

  void read(const cv::FileNode&node, deformable_depth::SearchTree&tree, deformable_depth::SearchTree)
  {
    read_in_map(node["children"],tree.children);
    node["id"] >> tree.id;
    node["pose_uuid"] >> tree.pose_uuid;
    node["Templ"] >> tree.Templ;
  }

  void write(cv::FileStorage&fs, std::string&s, const shared_ptr<deformable_depth::SearchTree>&tree)
  {
    if(tree)
      // call into the reference writer
      write(fs,s,*tree);
  }

  void write(cv::FileStorage&fs, std::string&, const deformable_depth::SearchTree&node)
  {
    fs << "{";

    string s;
    fs << "children"; write_in_map(fs,s,node.children);
    fs << "id" << node.id;
    fs << "pose_uuid" << node.pose_uuid;
    fs << "Templ" << node.Templ;

    fs << "}";
  }

  void log_search_tree(SearchTree&tree_root)
  {
    if(not g_params.option_is_set("NN_LOG_TREE"))
      return;

    // for calcing the branching factor
    int internal_nodes = 0;
    int internal_node_children = 0;

    function<void (SearchTree&,string)> expandFn = [&](SearchTree&node,string prefix) -> void
    {
      Mat RGB = imageeq("",node.Templ.getTIm(),false,false);
      string filename = string("bound") + uuid() + ".jpg";
      imwrite(prefix + "/" + filename,RGB);
      
      if(node.children.size() > 0)
      {
	internal_nodes++;
	internal_node_children += node.children.size();
      }

      for(auto & child : node.children)
      {
	string new_dir = printfpp("%s/%d",prefix.c_str(),child->id);
	assert(boost::filesystem::create_directory(new_dir));
	expandFn(*child,new_dir);
      }
    };
    
    // traverse the tree.
    expandFn(tree_root,params::out_dir());

    double mean_branching_factor = static_cast<double>(internal_node_children)/internal_nodes;
    log_once(safe_printf("mean_branching_factor = %",mean_branching_factor));
  }

  // represents and OR search free for AStar
  SearchTree train_search_tree_agglomerative(map<string,NNTemplateType>&allTemplates)
  {
    SearchTree search_tree_root;

    // initiailize the bottom layer
    unordered_map<int, unordered_map<int,double> > costs;    
    list<SearchTree > clusters(allTemplates.size());
    function<void (SearchTree&node)> update_dists_for_node = [&](SearchTree&node)
    {
      static atomic<int> completed(0);
      log_once(printfpp("++computed distance %d",completed++));
      for(auto & node_k : clusters)
      {
	double cost = node.Templ.merge_cost(node_k.Templ);
	static mutex m; lock_guard<mutex> l(m);
	costs[node.id][node_k.id] = cost;
	costs[node_k.id][node.id] = cost;
      }
      
    };
    auto iter = allTemplates.begin();
    for(int cter = 0; iter != allTemplates.end(); ++cter, ++iter)
    {
      auto jter = clusters.begin();
      std::advance(jter,cter);
      SearchTree&curTree = *jter;
      curTree.Templ = iter->second;
      curTree.id = seq_id++;
      curTree.pose_uuid = iter->first;
    }
    TaskBlock compute_init_dists("compute_init_dists");
    log_once(printfpp("++Computing Initial Distances for %d clusters",(int)clusters.size()));
    for(auto & curTree : clusters)
    {
      SearchTree*tp = &curTree;
      compute_init_dists.add_callee([&,tp]()
      {
	update_dists_for_node(*tp);
      });
    }
    compute_init_dists.execute();
    log_once(printfpp("--Computing Initial Distances for %d clusters",(int)clusters.size()));
    
    // merge layers until we get the root
    merge_clusters_agglom(seq_id,clusters,costs,update_dists_for_node);
    
    // store the root
    search_tree_root = clusters.front();
    
    // print the search tree
    log_search_tree(search_tree_root);

    return search_tree_root;
  }

  ///
  /// Train a A* Search Tree using kmeans
  ///
  SearchTree train_search_kmeans(map<string,NNTemplateType>&allTemplates,int depth)
  {    
    static constexpr int K = 2;      
    // base case
    assert(allTemplates.size() > 0);
    if(allTemplates.size() == 1)
    {
      SearchTree merged;
      merged.Templ = allTemplates.begin()->second;
      merged.children = vector<shared_ptr<SearchTree> >{};
      merged.id = seq_id++;      
      merged.pose_uuid = allTemplates.begin()->first;
      return merged;
    }
    // recursive case
    else
    {
      // invoke kmeans
      cv::Mat feats(allTemplates.size(),allTemplates.begin()->second.getTIm().size().area(),
		    DataType<float>::type,Scalar::all(0));
      int iter = 0;
      for(auto && templ : allTemplates)
	templ.second.getTIm().reshape(0,1).copyTo(feats.row(iter++));
      //cout << feats << endl;
      Mat bestLabels;
      cv::TermCriteria term_crit(cv::TermCriteria::MAX_ITER,1000,1000);
      int attempts = 20;
      cv::kmeans(feats,K,bestLabels,term_crit,attempts,KMEANS_PP_CENTERS);

      // generate the partitons
      map<int,map<string,NNTemplateType> > partitions;
      iter = 0;
      for(auto && templ : allTemplates)
      {
	int id = bestLabels.at<int>(iter++);
	for(int iter = 0; iter < depth; ++iter)
	  log_file << "\t";
	log_file << id << "  " << templ.first << endl;
	partitions[id][templ.first] = templ.second;	
      }      

      // generate the child nodes
      vector<SearchTree> child_trees;
      TaskBlock kmeans_recurse("kmeans_recurse");
      for(int k = 0; k < K; ++k)
      {
	kmeans_recurse.add_callee([&,k]()
				  {
				    auto sub_tree = train_search_kmeans(partitions[k],depth+1);
				    static mutex m; lock_guard<mutex> l(m);
				    child_trees.push_back(sub_tree);
				  });
      }
      kmeans_recurse.execute();
      
      // generate the split node
      SearchTree split;
      split.Templ = child_trees[0].Templ;
      split.children = vector<shared_ptr<SearchTree > >{make_shared<SearchTree>(child_trees[0])};
      for(int k = 1; k < K; ++k)
      {
	split.Templ = split.Templ.merge(child_trees[k].Templ);
	split.children.push_back(make_shared<SearchTree>(child_trees[k]));
      }
      split.id = seq_id++;      

      if(depth == 0)
      {
	log_search_tree(split);
	//save_search_tree(split);
      }

      return split;      
    }
  }
}

namespace deformable_depth
{
  SearchTree train_search_tree_pyramid(map<string,NNTemplateType>&allTemplates,bool translational)
  {
    // initialize
    tbb::concurrent_unordered_map<VolumetricTemplate,SearchTree> free_trees;
    TaskBlock init_map("train_search_tree_pyramid init");
    for(auto && templ : allTemplates)
    {
      init_map.add_callee([&,templ]()
			  {
			    SearchTree tree;
			    tree.Templ = templ.second; 
			    tree.children = vector<shared_ptr<SearchTree> >{};
			    tree.id = seq_id++;      
			    tree.pose_uuid = templ.first;
			    free_trees[tree.Templ] = tree;
			  });
    }
    init_map.execute();
    log_once(safe_printf("train_search_tree_pyramid init % %",allTemplates.size(),free_trees.size()));

    vector<Vec2i> translational_offsets = ((translational)?
					   vector<Vec2i>{{0,0}}:
					   vector<Vec2i>{{0,0},{0,1},{1,0},{1,1}});

    // merge the layers
    double ps = params::pyramid_sharpness();
    for(int layerIter = 0; free_trees.size() > 1; layerIter++)
    {
      Vec3i res = free_trees.begin()->second.Templ.resolution();
      res = Vec3i(ps*res[0],ps*res[1],ps*res[2]); // don't use /= to avoid rounding behavior.
      log_once(safe_printf("train_search_tree_pyramid layer = % free_trees = % size = (%,%,%)",
			   layerIter,free_trees.size(),res[0],res[1],res[2]));
      
      // build the next layer, merging where necessary
      tbb::concurrent_unordered_map<VolumetricTemplate,SearchTree> next_layer;
      for(auto && free_tree : free_trees)
      {
	for(auto && translation : translational_offsets)
	{
	  // build the down-sampled tree
	  SearchTree down_tree(res);
	  NNTemplateType newDownTempl(res);
	  newDownTempl = free_tree.second.Templ.pyramid_down();
	  down_tree.Templ = newDownTempl;
	  down_tree.children = vector<shared_ptr<SearchTree > >{make_shared<SearchTree> (free_tree.second) };
	  down_tree.id = seq_id++;      
	  down_tree.pose_uuid = free_tree.second.pose_uuid;
	  
	  // merge into the next layer
	  if(next_layer.find(down_tree.Templ) == next_layer.end())
	  {
	    next_layer[down_tree.Templ] = down_tree;
	  }
	  else
	  {
	    next_layer.find(down_tree.Templ)->second.Templ.incClusterSize(free_tree.second.Templ.getClusterSize());
	    next_layer.find(down_tree.Templ)->second.children.push_back(make_shared<SearchTree>(free_tree.second));
	  }
	} // end for each translation
      } // end for each tree
      free_trees = next_layer;
    }

    assert(free_trees.size() == 1);
    auto root = free_trees.begin()->second;
    log_search_tree(root);
    return root;
  }

  SearchTree train_search_tree(map<string,NNTemplateType>&allTemplates)
  {
    string merge_alg = g_params.require("ASTAR_CLUSTERING_ALG");

    if(merge_alg == "kmeans")
      return train_search_kmeans(allTemplates);
    else if(merge_alg == "linkage")
      return train_search_tree_agglomerative(allTemplates);    
    else if(merge_alg == "pyramid")
      return train_search_tree_pyramid(allTemplates);
    else if(merge_alg == "translational_pyramid")
      return train_search_tree_pyramid(allTemplates,true);
    else
      throw std::runtime_error("invalid astar clustering algorithm");
  }

  ///
  /// SECTION: for training heuristics
  ///
  class TemplateSpacePartitionFunction
  {
  protected:
    double count_true, count_false;
    Vec2i pt1, pt2;
    double threshold;

  public:
    TemplateSpacePartitionFunction();
    TemplateSpacePartitionFunction(vector<valid_ptr<const NNTemplateType> >&Ts);
    bool partition(const NNTemplateType* const T) const;
    void count(const NNTemplateType*T);
    double entropy() const;

    // serialization
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::TemplateSpacePartitionFunction&);
  };

  void write(cv::FileStorage&fs, std::string&, const deformable_depth::TemplateSpacePartitionFunction&pf)
  {
    fs << "{";
    fs << "count_true" << pf.count_true;
    fs << "count_false" << pf.count_false;
    fs << "threshold" << pf.threshold;    
    fs << "}";
  }

  TemplateSpacePartitionFunction::TemplateSpacePartitionFunction()
  {
  }

  TemplateSpacePartitionFunction::TemplateSpacePartitionFunction(vector<valid_ptr<const NNTemplateType> >&Ts) : 
    count_true(0), 
    count_false(0),
    threshold(sample_in_range(0,(NN_DEFAULT_ZRES-1))),
    pt1(sample_in_range(0,SpearTemplSize.width-1),sample_in_range(0,SpearTemplSize.height-1)),
    pt2(sample_in_range(0,SpearTemplSize.width-1),sample_in_range(0,SpearTemplSize.height-1))
  {
    const NNTemplateType* T1 = Ts.at(thread_rand()%Ts.size()).get();
    const NNTemplateType* T2 = Ts.at(thread_rand()%Ts.size()).get();
    double d1 = T1->getTRef().at<float>(pt1[0],pt1[1]);
    double d2 = T2->getTRef().at<float>(pt1[0],pt1[1]);
    threshold = (d1 + d2)/2;
  }

  bool TemplateSpacePartitionFunction::partition(const NNTemplateType* const T) const
  {
    if(T->getTRef().empty())
      return false;
    volatile double v1 = T->getTRef().at<float>(pt1[0],pt1[1]);
    //double v2 = T->getTRef().at<float>(pt2[0],pt2[1]);
    return v1 > (threshold);
  }

  void TemplateSpacePartitionFunction::count(const NNTemplateType*T)
  {
    if(partition(T))
      count_true++;
    else
      count_false++;
  }

  double TemplateSpacePartitionFunction::entropy() const
  {
    return shannon_entropy(count_true/static_cast<double>(count_true+count_false));
  }
  
  ///
  /// SECTION: SecondOrderRandomTree
  ///

  class SecondOrderRandomTree // SORT?
  {
  protected:
    int AStarSearchNode_Identifier;
    vector<string> enforced_ids;
    vector<valid_ptr<const NNTemplateType> > active_Ts;
    vector<valid_ptr<const NNTemplateType> > all_Ts;
    shared_ptr<SecondOrderRandomTree> true_branch;
    shared_ptr<SecondOrderRandomTree> false_branch;
    mutable recursive_mutex monitor;
    TemplateSpacePartitionFunction part_fn;    

  public:
    // forms a heuristic for the give node
    bool grow();    
    double heuristic(const SearchTree&,const NNTemplateType&);
    // creates the root.
    SecondOrderRandomTree(SearchTree&searchNode);
    // creates the branches.
    SecondOrderRandomTree(vector<valid_ptr<const NNTemplateType> > all_Ts,int AStarSearchNode_Identifier);
    virtual ~SecondOrderRandomTree();
    void enforce_admissibility_against(const NNTemplateType&X);    
    void mark_admissibility_enforced(const string&id);
    void print(int indentation_level);
    
    // serialization
    void save();
    friend void write(cv::FileStorage&, std::string&, const deformable_depth::SecondOrderRandomTree&);
    friend void read(const cv::FileNode&, deformable_depth::SecondOrderRandomTree&, deformable_depth::SecondOrderRandomTree);
  };

  void SecondOrderRandomTree::mark_admissibility_enforced(const string&id)
  {
    lock_guard<recursive_mutex> l(monitor);

    enforced_ids.push_back(id);
    if(true_branch)
      true_branch->mark_admissibility_enforced(id);
    if(false_branch)
      false_branch->mark_admissibility_enforced(id);
  }

  void write(cv::FileStorage&fs, std::string&, const deformable_depth::SecondOrderRandomTree&tree)
  {
    string s;

    fs << "{";
    fs << "AStarSearchNode_Identifier" << tree.AStarSearchNode_Identifier;
    fs << "enforced_ids" << tree.enforced_ids;
    fs << "active_Ts"; write_seq_as_map(fs,s,tree.active_Ts);
    //fs << "active_Ts" << tree.active_Ts;
    fs << "all_Ts"; write_seq_as_map(fs,s,tree.all_Ts);
    if(tree.true_branch)
      fs << "true_branch" << *tree.true_branch;
    if(tree.false_branch)
      fs << "false_branch" << *tree.false_branch;
    fs << "part_fn" << tree.part_fn;    
    
    fs << "}";
  }

  void SecondOrderRandomTree::save()
  {
    lock_guard<recursive_mutex> l(monitor); // avoid possible race conditions...

    // setup the fliename
    auto now = std::chrono::system_clock::now();
    long seconds_past_epoc = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    string save_file = safe_printf("%/SecondOrderTree_%_%.yml.gz",
				   params::out_dir(),AStarSearchNode_Identifier,seconds_past_epoc);
    log_file << "SecondOrderRandomTree::save = " << save_file << endl;

    // create the file and commit to it.
    FileStorage fs(save_file,FileStorage::WRITE);
    fs << "Tree" << *this;
    fs.release();
  }

  void SecondOrderRandomTree::print(int indentation_level)
  {
    cout << "SecondOrderRandomTree::print ";
    for(int iter = 0; iter < indentation_level; ++iter)
      cout << "\t";
    cout << active_Ts.size() << " of " << all_Ts.size();
    cout << endl;

    if(true_branch)
      true_branch->print(indentation_level+1);
    if(false_branch)
      false_branch->print(indentation_level+1);
  }

  bool SecondOrderRandomTree::grow()
  {
    if(active_Ts.size() < 2)
      return false;

    vector<TemplateSpacePartitionFunction> partitions;    
    for(int iter = 0; iter < 500; ++iter)
      partitions.push_back(TemplateSpacePartitionFunction(active_Ts));

    for(auto && T : all_Ts)
      for(auto && partition : partitions)
	partition.count(T.get());
    
    // choose the one with the most entropy.
    double entropy = -inf;
    for(auto && part : partitions)
    {
      double part_h = part.entropy();
      if(part_h > entropy)
      {
	entropy = part_h;
	part_fn = part;
      }
    }
    cout << "selected partition function with entropy of " << entropy << endl;

    true_branch = make_shared<SecondOrderRandomTree>(all_Ts,AStarSearchNode_Identifier);
    false_branch = make_shared<SecondOrderRandomTree>(all_Ts,AStarSearchNode_Identifier);
    return true;
  }

  SecondOrderRandomTree::SecondOrderRandomTree(
    vector<valid_ptr<const NNTemplateType> > all_Ts,int AStarSearchNode_Identifier) : 
    all_Ts(all_Ts), AStarSearchNode_Identifier(AStarSearchNode_Identifier)
  {
  }

  SecondOrderRandomTree::SecondOrderRandomTree(SearchTree&searchNode)
  {
    // populate all_Ts
    std::function<void (SearchTree&)> populate_all_Ts = [&](SearchTree&node)
    {
      if(not node.Templ.getTRef().empty())
	all_Ts.push_back(&node.Templ);

      for(auto && child : node.children)
	populate_all_Ts(*child);
    };
    populate_all_Ts(searchNode);

    active_Ts = all_Ts;
    AStarSearchNode_Identifier = searchNode.id;
  }

  SecondOrderRandomTree::~SecondOrderRandomTree()
  {    
  }

  void SecondOrderRandomTree::enforce_admissibility_against(const NNTemplateType&X)
  {
    // we've been split! 
    if(true_branch and false_branch)
    {
      if(part_fn.partition(&X))
	true_branch->enforce_admissibility_against(X);
      else
	false_branch->enforce_admissibility_against(X);
      return;
    }

    // enforce locally. 
    const NNTemplateType* nn;
    double nn_score = -inf;
    for(auto && T : all_Ts)
    {
      double cor = T->cor(X);
      if(goodNumber(cor) and cor > nn_score)
      {
	nn_score = cor;
	nn = T.get();
      }
    }

    {
      lock_guard<recursive_mutex> l(monitor);
      if(std::find(active_Ts.begin(),active_Ts.end(),nn) == active_Ts.end())
      {
	active_Ts.push_back(nn);
      }
    }
  }
  
  double SecondOrderRandomTree::heuristic(const SearchTree&Ts,const NNTemplateType&X)
  {
    // we've been split! 
    if(true_branch and false_branch)
    {
      if(part_fn.partition(&X))
	return true_branch->heuristic(Ts,X);
      else
	return false_branch->heuristic(Ts,X);
    }

    // we want to return the max score in the active set.
    double max_score = -inf;
    
    for(auto && T : active_Ts)
    {
      double cor = T->cor(X);
      if(goodNumber(cor) && cor > max_score)
	max_score = cor;
    }

    return max_score;
  }
  
  ///
  /// Top level functions describing the creation of a second order forest.
  ///

  vector<shared_ptr<SecondOrderRandomTree> > init_hord_trees(SearchTree&root, int depth)
  {
    vector<shared_ptr<SecondOrderRandomTree> > forest;

    // allocate a new node
    auto ids = g_params.matching_values("SORT_NODES");
    string id = toString(root.id);
    int max_depth = fromString<int>(g_params.require("SORT_TO_DEPTH"));
    if(ids.find(id) != ids.end() or depth < max_depth)
    {
      log_once(safe_printf("init_hord_trees allocating %",root.id)); 
      auto tree = make_shared<SecondOrderRandomTree>(root);
      forest.push_back(tree);
      
      // set the node's heuristic into the search tree
      shared_ptr<NodeHeuristicFn> h = make_shared<NodeHeuristicFn>([tree](const SearchTree&t,const NNTemplateType&x)
								   {
								     return tree->heuristic(t,x);
								   });
      root.heuristics.push_back(h);
    }

    // recursively build heuristics for all children.
    for(auto && child : root.children)
    {
      vector<shared_ptr<SecondOrderRandomTree> > sub_forest = init_hord_trees(*child,depth+1);
      forest.insert(forest.end(),sub_forest.begin(),sub_forest.end());
    }

    return forest;
  }

  static void train_hord_heuristics_naive(ApxNN_Model&model,SearchTree&tree_root,vector<shared_ptr<MetaData> >&training_set)
  {
    // the "T", or template/database, set is defined by the leaf nodes of the tree
    // allocate the forest...
    vector<shared_ptr<SecondOrderRandomTree> > forest = init_hord_trees(tree_root,0);
    log_file << "train_hord_heuristics_naive forest.size() = " << forest.size() << endl;
    if(forest.size() == 0)
      return;

    // build a heuristic for this node.
    // train the forest... online
    for(int iter = 0; iter < 5 /*max depth*/; ++iter)
    {
      progressBars->set_progress("train_hord_heuristics level",iter,5);
      bool grew = false;
      for(auto && tree : forest)
	grew |= tree->grow();

      TaskBlock train_hord_heuristics("train_hord_heuristics");
      atomic<int> data_complete(0);
      for(auto & datum : training_set)
      {
	train_hord_heuristics.add_callee(
	  [&,datum]()
	  {
	    cout << safe_printf("train_hord_heuristics extracting %",datum->get_filename()) << endl;
	    shared_ptr<ImRGBZ> im = datum->load_im();
	    NNSearchParameters nn_params;
	    nn_params.stop_at_window = true;
	    DetectionFilter filter;
	    filter.cheat_code = datum;
	    // this defines the "X", or query, set.
	    DefaultSearchAlgorithm search(*im,model,filter);
	    search.init(filter);	      
	    
	    cout << safe_printf("train_hord_heuristics enforcing %",datum->get_filename()) << endl;
	    for(int templ_id = 0; templ_id < search.numberXS(); ++templ_id)
	    {
	      progressBars->set_progress(safe_printf("train_hord_heuristics %",datum->get_filename()),templ_id,search.numberXS());

	      if(search.suppressedX(templ_id))
		continue;
	      shared_ptr<NNTemplateType> X = search.getX(templ_id);
	      if(not X)
		continue;
	      
	      for(auto && tree : forest)
	      {
		tree->enforce_admissibility_against(*X);
	      }
	    }
	    // log our progress to avoid duplicate work in the not unlikely event of a crash.
	    for(auto && tree : forest)
	    {
	      tree->mark_admissibility_enforced(datum->get_filename());
	      tree->save();
	    }
	    progressBars->set_progress(safe_printf("train_hord_heuristics %",datum->get_filename()),1,1); // DONE
	    progressBars->set_progress("for datum in training_set",++data_complete,training_set.size());
	  });
      }
      train_hord_heuristics.execute();
      progressBars->set_progress("for datum in training_set",1,1);

      for(auto && tree : forest)
	tree->print(0);
    }
    progressBars->set_progress("train_hord_heuristics level",1,1);
  }

  void train_hord_heuristics(ApxNN_Model&model,SearchTree&tree_root,vector<shared_ptr<MetaData> >&training_set)
  {
    //train_hord_heuristics_naive(model,tree_root,training_set);
  }
}
