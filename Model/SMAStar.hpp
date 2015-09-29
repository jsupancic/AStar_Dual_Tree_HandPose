/**
 * Copyright 2014: James Steven Supancic III
 **/

#ifndef DD_SMASTAR_NN
#define DD_SMASTAR_NN

#include "AStarNN.hpp"

namespace deformable_depth
{
  // represents an invocation of SMA*, the simplified memory version of A* search
  // see [1] S. Russell, “Efficient memory-bounded search methods.”
  class SMAStarSearch : public InformedSearch
  {
  protected:
    set<SMAStarNodePtr> opened, closed;
    int max;
    int pruned;

    void prune();

  public:
    SMAStarSearch(const ImRGBZ& im,const ApxNN_Model&model,DetectionFilter&filter);
    virtual ~SMAStarSearch();
    virtual bool iteration(DetectionSet&dets) override;
    virtual void init_frontier(const AStarSearchNode&node) override;
  };

  // 
  struct SMAStarSearchNode : public AStarSearchNode
  {
    // also required for SMA
    int successor;    
    // parent
    weak_ptr<SMAStarSearchNode> parent;
    // extant successors
    map<int,SMAStarNodePtr> extant_successors;
    // the tree depth
    int search_depth;

    SMAStarSearchNode() : 
      successor(0), search_depth(-1)
    {
    }

    // note: http://stackoverflow.com/questions/1114856/stdset-with-user-defined-type-how-to-ensure-no-duplicates
    // operator== is not used by std::set. Elements a and b are considered equal iif !(a < b) && !(b < a)
    bool operator< (const SMAStarSearchNode&other) const
    {
      assert(search_depth >= 0);
      if(cost == other.cost)
      {
	if(search_depth == other.search_depth)
	  return this < &other;
	else
	  return search_depth < other.search_depth; // greatest depth comes last in sorted order
      }
      else
	// std::priority queue returns higher first.
	return cost > other.cost; // lowest cost comes last in sorted order
    };    
  };

  struct SMAStarNodePtr
  {
  public:
    shared_ptr<SMAStarSearchNode> node;

    SMAStarNodePtr(){};
    SMAStarNodePtr(const SMAStarNodePtr&) = default;

    SMAStarNodePtr(const AStarSearchNode&init)
    {
      node = make_shared<SMAStarSearchNode>();
      static_cast<AStarSearchNode&>(*node) = init;
      assert(node->tree != nullptr);
      assert(node->cost == init.cost);
      node->search_depth = 0;
    }

    bool operator==(const SMAStarNodePtr&other) const
    {
      return node.get() == other.node.get();
    }
    
    bool operator < (const SMAStarNodePtr&other) const
    {
      return *node < *other.node;
    }

    void backup(set<SMAStarNodePtr>&opened,set<SMAStarNodePtr>&closed);
  };  
}

#endif
