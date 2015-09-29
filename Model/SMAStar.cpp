/**
 * Copyright 2014: James Steven Supancic III
 **/

#include "SMAStar.hpp"

namespace deformable_depth
{
  ///
  /// SECTION: SMAStarSearch
  ///  
  SMAStarSearch::SMAStarSearch(const ImRGBZ& im,const ApxNN_Model&model,DetectionFilter&filter) : 
    InformedSearch(im,model,filter),
    max(fromString<int>(g_params.require("NN_SMA_OPENED_LIMIT"))), pruned(0)
  {
  }

  void SMAStarSearch::init_frontier(const AStarSearchNode&node)
  {
    opened.insert(SMAStarNodePtr(node));
  }

  SMAStarSearch::~SMAStarSearch()
  {
    log_once(safe_printf("% (open.size() = % pruned = %)",im.filename.c_str(),opened.size(),pruned));
  }
  
  bool SMAStarSearch::iteration(DetectionSet&dets)
  {
    // get the best node to process
    if(opened.empty())
      return false;
    assert(opened.rbegin()->node->cost <= opened.begin()->node->cost);
    SMAStarNodePtr best = *opened.rbegin();

    // get the relevant template
    shared_ptr<NNTemplateType> XTempl = (*XS[best.node->X_index])(); 
    if(!XTempl->is_valid())
    {
      ++DEBUG_invalid_Xs_from_frontier;
      return !opened.empty();
    }

    if(best.node->tree->children.size() == 0)
    {
      return emit_best(*best.node,dets);
    }

    // expand the successor
    int succ_index = best.node->successor++;    
    if(!(0 <= succ_index && succ_index < best.node->tree->children.size()))
    {
      assert(opened.erase(best) == 1);
      closed.insert(best);
      return !opened.empty();
    }      
    const SearchTree & child = *best.node->tree->children[succ_index];

    // generate the successor
    template_evals++;
    double cost = -child.Templ.cor(*XTempl,admissibility) + 0 ;//active.cost;
    //cost = std::max<double>(best.node->cost,cost);
    assert(&child != nullptr);
    AStarSearchNode top{cost,&child,best.node->depth,best.node->X_index};
    SMAStarNodePtr succ(top);
    succ.node->parent = best.node;
    succ.node->search_depth = best.node->search_depth+1;

    // if completed(best), BACKUP(best)
    if(succ_index + 1 >= best.node->tree->children.size())
    {
      closed.insert(best);
      best.backup(opened,closed);
    }

    // if S(best) all in memory, remove best from open
    if(best.node->extant_successors.size() == best.node->tree->children.size())
    {
      assert(opened.erase(best) == 1);
    }

    // Perform pruning on overflow
    prune();

    // insert succ on open
    opened.insert(succ);
    best.node->extant_successors[succ_index] = succ;    

    return !opened.empty();
  }

  void SMAStarSearch::prune()
  {  
    auto badIter = opened.begin();      
    for(int iter = 0; opened.size() > max && badIter != opened.end(); ++iter)
    {      
      const SMAStarNodePtr bad = *badIter;
      auto parent = bad.node->parent.lock();
      // only erase nodes with parents.
      if(parent)
      {
	// remove from the frontier
	opened.erase(opened.begin());
	pruned++;
	bool powOf2 = !(pruned == 0) && !(pruned & (pruned - 1));
	if(powOf2)
	  log_once(safe_printf("% (open.size() = % pruned = % iter = %)",im.filename.c_str(),opened.size(),pruned,iter));
	  
	// remove from parents successor list
    	for(auto && p_suc : parent->extant_successors)
    	  if(p_suc.second.node == bad.node)
    	  {
    	    parent->extant_successors.erase(p_suc.first);
    	    break;
    	  }
    	// if necessary re-add the parent
    	if(opened.find(*parent) == opened.end())
    	{
    	  opened.insert(*parent);
	  closed.erase(*parent);
    	}
    	parent->successor = 0;
	// reset the invalided iterator
	badIter = opened.begin();      
	if(badIter == opened.end())
	  return;
      } // end if(parent)
      else
	++badIter;
    }
  }

  void SMAStarNodePtr::backup(set<SMAStarNodePtr>&opened,set<SMAStarNodePtr>&closed)
  {
    bool changed = false;
    for(auto && succ : node->extant_successors)
    {
      double suc_cost = succ.second.node->cost;
      if(suc_cost < node->cost)
      {
	node->cost = suc_cost;
	changed = true;
      }
    }

    if(changed)
    {      
      shared_ptr<SMAStarSearchNode> p = node->parent.lock();
      if(p)
      {
	SMAStarNodePtr pp;
	pp.node = p;
	pp.backup(opened,closed);
      }
      
      // reheapify
      SMAStarNodePtr self = *this;
      if(opened.erase(self) > 0)
	opened.insert(self);
      if(closed.erase(self) > 0)
	closed.insert(self);
    }
  }
}


