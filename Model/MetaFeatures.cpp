/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "MetaFeatures.hpp"
#include "DepthFeatures.hpp"
#include "MetaData.hpp"
#include "Eval.hpp"
#include "Log.hpp"
#include <algorithm>
#include "Faces.hpp"
#include "Skin.hpp"

namespace deformable_depth
{
  /**
   * SECTION: Implementation of block cell remapper
   **/
  BlockCellRemapper::BlockCellRemapper
    (HOGComputer18x4_General::im_fun use_fun, Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    computer(use_fun,win_size,block_size,block_stride,cell_size)
  {
  }

  BlockCellRemapper::BlockCellRemapper(BlockCellRemapper& other) :
    computer(other.computer)
  {
  }
  
  void BlockCellRemapper::compute(const ImRGBZ& im, std::vector< float >& feats)
  {
    computer.compute(im,feats);
    map(feats);
  }

  Size BlockCellRemapper::getBlockStride()
  {
    return computer.getBlockStride();
  }

  Size BlockCellRemapper::getCellSize()
  {
    return computer.getCellSize();
  }

  int BlockCellRemapper::cellsPerBlock()
  {
    // #C = #B
    return 1;
  }
  
  size_t BlockCellRemapper::getDescriptorSize()
  {
    return 4*computer.getNBins()*blocks_x()*blocks_y();
  }

  int BlockCellRemapper::getNBins()
  {
    assert(computer.cellsPerBlock() == 4);
    ncells = computer.getNBins()*4;
    return ncells;
  }

  Size BlockCellRemapper::getWinSize()
  {
    return computer.getWinSize();
  }

  void BlockCellRemapper::map(std::vector< float >& block_feats)
  {
    assert(block_feats.size() == computer.getDescriptorSize());
    vector<float> cell_feats(getDescriptorSize(),0);
    
    // getIndex(int blockX, int blockY, int cellN, int bin);
    assert(computer.cellsPerBlock() == 4);
    for(int blockX = 0; blockX < computer.blocks_x(); blockX++)
      for(int blockY = 0; blockY < computer.blocks_y(); blockY++)
	for(int cellN = 0; cellN < 4; cellN++)
	  for(int bin = 0; bin < computer.getNBins(); bin++)
	  {	    
	    // get from the celly form (cells)
	    // notes
	    // dst block depends on cell # and src block
	    int cell_cell_X, cell_cell_Y;
	    map_blockToCell4(blockX, blockY, cellN, cell_cell_X, cell_cell_Y);
	    // figure out where to put it in the dest
	    int icell = getIndex(cell_cell_X,cell_cell_Y,0,bin+cellN*computer.getNBins());
	    assert(icell >= 0 && icell < cell_feats.size());
	    float&cell_val = cell_feats[icell];
	    
	    // get each piece of data from in the blocky
	    int iblk = computer.getIndex(blockX,blockY,cellN,bin);
	    assert(iblk >= 0 && iblk < block_feats.size());	
	    float&blk_val = block_feats[iblk];    
	    
	    // do the copy
	    cell_val = blk_val;	    
	  }
    
    block_feats = cell_feats;
  }

  // Cell Format to block format
  void BlockCellRemapper::unmap(std::vector< double >& cell_feats)
  {
    if(cell_feats.size() != getDescriptorSize())
    {
      cout << printfpp("%d != %d",cell_feats.size(),getDescriptorSize()) << endl;
      throw std::logic_error("fuck!!!");
    }
    vector<double> block_feats(computer.getDescriptorSize(),0);
    
    // getIndex(int blockX, int blockY, int cellN, int bin);
    assert(computer.cellsPerBlock() == 4);
    for(int blockX = 0; blockX < computer.blocks_x(); blockX++)
      for(int blockY = 0; blockY < computer.blocks_y(); blockY++)
	for(int cellN = 0; cellN < 4; cellN++)
	  for(int bin = 0; bin < computer.getNBins(); bin++)
	  {	    
	    // get from the celly form (cells)
	    // notes
	    // dst block depends on cell # and src block
	    int cell_cell_X, cell_cell_Y;
	    map_blockToCell4(blockX, blockY, cellN, cell_cell_X, cell_cell_Y);
	    // figure out where to put it in the dest
	    int icell = getIndex(cell_cell_X,cell_cell_Y,0,bin+cellN*computer.getNBins());
	    if(!(icell >= 0 && icell < cell_feats.size()))
	    {
	      printf("cellX = %d cellY = %d\n",cell_cell_X,cell_cell_Y);
	      printf("icell = %d of %d\n",icell,(int)cell_feats.size());
	    }
	    assert(icell >= 0 && icell < cell_feats.size());
	    double&cell_val = cell_feats[icell];
	    
	    // get each piece of data from in the blocky
	    int iblk = computer.getIndex(blockX,blockY,cellN,bin);
	    assert(iblk >= 0 && iblk < block_feats.size());	
	    double&blk_val = block_feats[iblk];    
	    
	    // do the copy
	    blk_val = cell_val;	    
	  }
    
    cell_feats = block_feats;
  }
  
  Mat BlockCellRemapper::show(const string& title, std::vector< double > feat)
  {
    unmap(feat);
    return computer.show(title,feat);
  }
  
  vector< FeatVis > BlockCellRemapper::show_planes(std::vector< double > feat)
  {
    unmap(feat);
    return computer.show_planes(feat);
  }
  
  Size BlockCellRemapper::getBlockSize()
  {
    return getCellSize();
  }
  
  /// SECTION: Feature Combiner
  void FeatureBinCombiner::normalize_weights()
  {
    weights.clear();
    
    for(shared_ptr<DepthFeatComputer>&computer : computers)
    {
      assert(computer->blocks_x() == computers[0]->blocks_x());
      assert(computer->blocks_y() == computers[0]->blocks_y());
    }
    
    // init the weights
    weights.push_back(1.0);
    for(int iter = 1; iter < computers.size(); iter++)
    {
      weights.push_back(
	static_cast<double>(computers[0]->getDescriptorSize())/
	static_cast<double>(computers[iter]->getDescriptorSize())
      );
      //weights.push_back(1.0);
    }
  }
  
  FeatureBinCombiner::FeatureBinCombiner(Size win_size, Size block_size, Size block_stride, Size cell_size)
  {
    // for now, use ZHOG and ZAREA
    computers.push_back(shared_ptr<DepthFeatComputer>(
      new HOGComputer18p4_General([](const ImRGBZ&im)->const Mat&{return im.Z;},win_size,block_size,block_stride,cell_size)));
    //computers.push_back(shared_ptr<DepthFeatComputer>(
      //new HOGComputer_Area(win_size,block_size,block_stride,cell_size)));
//     computers.push_back(shared_ptr<DepthFeatComputer>(
//       new HistOfNormals(win_size,block_size,block_stride,cell_size)));
    //computers.push_back(shared_ptr<DepthFeatComputer>(
      //new HOGComputer18p4_General([](const ImRGBZ&im)->const Mat&{return im.gray();},win_size,block_size,block_stride,cell_size)));
    
    normalize_weights();
  }
  
  string FeatureBinCombiner::toString() const
  {
    ostringstream oss;
    oss << "FeatureBinCombiner";
    for(auto & computer : computers)
      oss << computer->toString();
    return oss.str();
  }
  
  int FeatureBinCombiner::cellsPerBlock()
  {
    return 1;
  }

  size_t FeatureBinCombiner::getDescriptorSize()
  {
    size_t len = 0;
    for(shared_ptr<DepthFeatComputer>&computer : computers)
      len += computer->getDescriptorSize();
    return len;
  }
  
  void FeatureBinCombiner::compute(const ImRGBZ& im, std::vector< float >& feats)
  {
    // check the depth image for NaNs
    //assert(!any<float>(im.Z,[](float v){return !goodNumber(v);}));
    bool rows_match = im.rows() == blocks_y()*getCellSize().height;
    bool cols_match = im.cols() == blocks_x()*getCellSize().width;
    if(!(rows_match && cols_match))
    {
      cout << printfpp("imSize = %d %d",im.cols(),im.rows()) << endl;
      cout << printfpp("blocks_x[%d]*cell_size.width[%d] = %d",
		       blocks_x(),getCellSize().width,blocks_x()*getCellSize().width) << endl;
      cout << printfpp("blocks_y[%d]*cell_size.height[%d] = %d",
		       blocks_y(),getCellSize().height,blocks_y()*getCellSize().height) << endl;
      assert(rows_match && cols_match);
    }
    
    // precompute the features with each computer.
    vector<vector<float> > in_feats;
    int iter = 0;
    for(shared_ptr<DepthFeatComputer>&computer : computers)
    {
      in_feats.push_back(vector<float>());
      computer->compute(im,in_feats.back());
      // check for errors
      for(float&value : in_feats.back())
      {
	bool ok = goodNumber(value);
	if(!ok)
	  cout << "problem @ computer = " << iter << endl;
	assert(ok);
      }
      iter++;
    }
    
    // combine into output
    feats = vector<float>(getDescriptorSize(),0);
    for(int blockX = 0; blockX < blocks_x(); blockX++)
      for(int blockY = 0; blockY < blocks_y(); blockY++)
      {
	int outbin = 0;
	for(int compIter = 0; compIter < computers.size(); compIter++)
	{
	  shared_ptr<DepthFeatComputer> computer = computers[compIter];
	  assert(computer->cellsPerBlock() == 1);
	  for(int inbin = 0; inbin < computer->getNBins(); inbin++,outbin++)
	    feats[getIndex(blockX,blockY,0,outbin)] = 
	    in_feats[compIter][computer->getIndex(blockX,blockY,0,inbin)]*weights[compIter];
	}     
      }
  }

  Size FeatureBinCombiner::getBlockSize()
  {
    // they shall all have the same value via RAII postcondition.
    return computers[0]->getBlockStride();
  }
  
  Size FeatureBinCombiner::getBlockStride()
  {
    return computers[0]->getBlockStride();
  }

  Size FeatureBinCombiner::getCellSize()
  {
    return computers[0]->getCellSize();
  }

  int FeatureBinCombiner::getNBins()
  {
    int nbins = 0;
    for(shared_ptr<DepthFeatComputer>&computer : computers)
      nbins += computer->getNBins();
    return nbins; 
  }
  
  Size FeatureBinCombiner::getWinSize()
  {
    return computers[0]->getWinSize();
  }

  Mat FeatureBinCombiner::show(const string& title, std::vector< double > feat)
  {
    // compute the planes of the image
    vector<FeatVis> subPlanes = show_planes(feat);
      
    // display
    Mat display(0,0,DataType<Vec3b>::type); 
    for(int planeIter = 0; planeIter < subPlanes.size(); planeIter++)
    {
      Mat pnTempl = horizCat(subPlanes[planeIter].getPos(),subPlanes[planeIter].getNeg());
      display = horizCat(display,pnTempl);
    }
    return display;
  }  
  
  vector< FeatVis > FeatureBinCombiner::show_planes(std::vector< double > feat)
  {
    // preallocate the per-type feature vectors
    vector<vector<float> > out_feats;
    for(shared_ptr<DepthFeatComputer>&computer : computers)
    {
      out_feats.push_back(vector<float>(computer->getDescriptorSize(),0));
    }
    
    // combine into output
    for(int blockX = 0; blockX < blocks_x(); blockX++)
      for(int blockY = 0; blockY < blocks_y(); blockY++)
      {
	int inbin = 0;
	for(int compIter = 0; compIter < computers.size(); compIter++)
	{
	  shared_ptr<DepthFeatComputer> computer = computers[compIter];
	  assert(computer->cellsPerBlock() == 1);
	  for(int outbin = 0; outbin < computer->getNBins(); inbin++,outbin++)
	    out_feats[compIter][computer->getIndex(blockX,blockY,0,outbin)] = 
	    feat[getIndex(blockX,blockY,0,inbin)]/weights[compIter];
	}     
      }    
    
    // fill the planes with output
    vector<FeatVis> planes;
    for(int compIter = 0; compIter < computers.size(); compIter++)
    {
      vector<FeatVis> subPlanes = computers[compIter]->show_planes(vec_f2d(out_feats[compIter]));
      for(FeatVis & featVis : subPlanes)
	if(featVis.getPos().type() != DataType<Vec3b>::type || 
	   featVis.getNeg().type() != DataType<Vec3b>::type)
	{
	  log_file << "FeatureBinCombiner::show_planes failure @ " << compIter << endl;
	  cout     << "FeatureBinCombiner::show_planes failure @ " << compIter << endl;
	  assert(false);
	}
      planes.insert(planes.end(),subPlanes.begin(),subPlanes.end());
    }    
    
    return planes;
  }
  
  ///
  /// SECTION: PCA Feature Implementation
  ///
  static std::unique_ptr<cv::PCA> g_pca_reducer;
  
  void PCAFeature::ready_pca()
  {
    if(g_pca_reducer)
      return;
    
    #pragma omp critical(PCA_TRAIN)
    {
      if(!g_pca_reducer)
      {
	train_pca();
      }
    }
  }
  
  void PCAFeature::train_pca()
  {
    // prepare the PCA.
    // (1) collect training examples
    cout << "PCAFeature: collecting" << endl;
    vector<shared_ptr<MetaData> > all_data; // unsupervised so we can do this
    auto load_insert_examples = [&all_data](string direct)
    {
      vector<shared_ptr<MetaData> > some_data = metadata_build_all(direct);
      all_data.insert(all_data.end(),some_data.begin(),some_data.end());
    };
    for(string&test_dir : default_test_dirs())
      load_insert_examples(test_dir);
    //load_insert_examples(default_train_dir());    
    
    // (2) collect the cells for the PCA
    shared_ptr<const ImRGBZ> im0 = cropToCells(*all_data[0]->load_im());
    int winX = im0->cols();
    int winY = im0->rows();
    unique_ptr<SubComputer> hog_for_scale(
      new SubComputer(Size(winX,winY),block_size,block_stride,cell_size));
    int nbins = hog_for_scale->getNBins();
    Mat all_cells(0,nbins,DataType<float>::type);
    for(shared_ptr<MetaData> & metadata : all_data)
    {      
      // compute all cells in the image
      shared_ptr<const ImRGBZ> im = cropToCells(*metadata->load_im());
      vector<float> feats; hog_for_scale->compute(*im,feats);
      assert(feats.size()%nbins == 0);
      for(int cell_iter = 0; cell_iter < feats.size()/nbins; cell_iter++)
      {
	// extract each cell in the image
	auto begin = feats.begin() + nbins*cell_iter;
	auto end = begin + nbins;
	vector<float> cell(begin,end);
	Mat cell_mat(cell); cell_mat = cell_mat.t();
	assert(cell_mat.rows == 1 && cell_mat.cols == nbins);
	all_cells.push_back<float>(cell_mat);
      }
    }
    
    // (3) run the PCA
    cout << "PCAFeature: computing" << endl;
    g_pca_reducer.reset(new cv::PCA(all_cells,noArray(),CV_PCA_DATA_AS_ROW,FEAT_DIM));
    cout << "PCAFeature: ready" << endl;
    
    // (4) save the PCA?
    FileStorage cache_storage("cache/PCA.yml",FileStorage::WRITE);
    cache_storage << "pca_eig_values" << g_pca_reducer->eigenvalues;
    cache_storage << "pca_eig_vectors" << g_pca_reducer->eigenvectors;
    cache_storage << "pca_mean" << g_pca_reducer->mean;
    cache_storage.release();
  }
  
  PCAFeature::PCAFeature(Size win_size, Size block_size, Size block_stride, Size cell_size) :
    unreduced_computer(win_size,block_size,block_stride,cell_size),
    win_size(win_size),block_size(block_size),block_stride(block_stride),cell_size(cell_size)
  {
    // sanity check our assumptions
    assert(unreduced_computer.cellsPerBlock() == 1);
    
    // make sure we have a ready PCA
    ready_pca();
  }
  
  Size PCAFeature::getBlockSize()
  {
    return unreduced_computer.getBlockSize();
  }
  
  int PCAFeature::cellsPerBlock()
  {
    return unreduced_computer.cellsPerBlock();
  }
  
  Size PCAFeature::getBlockStride()
  {
    return unreduced_computer.getBlockStride();
  }
  
  int PCAFeature::getNBins()
  {
    return FEAT_DIM;
  }
  
  void PCAFeature::compute(const ImRGBZ& im, std::vector< float >& reduced_feats)
  {
    // allocate space
    vector<float> unreduced_feats; unreduced_computer.compute(im,unreduced_feats);
    // wrap with matrices
    int ncells = blocks_x()*blocks_y();
    Mat mat_unreduced_feats(
      ncells,unreduced_computer.getNBins(),DataType<float>::type,&unreduced_feats[0]);
    
    // compute the PCA
    Mat mat_reduced_feats = g_pca_reducer->project(mat_unreduced_feats);
    
    // must I now divide by the eigenvalues?
    assert(g_pca_reducer->eigenvalues.type() == DataType<float>::type);
    for(int rIter = 0; rIter < mat_reduced_feats.rows; rIter++)
      for(int cIter = 0; cIter < mat_reduced_feats.cols; cIter++)
	mat_reduced_feats.at<float>(rIter,cIter) /= g_pca_reducer->eigenvalues.at<float>(cIter);
      
    // convert to a flat feature vector
    reduced_feats = mat_reduced_feats.reshape(0,mat_reduced_feats.size().area());
    assert(reduced_feats.size() == getDescriptorSize());
  }
  
  Size PCAFeature::getCellSize()
  {
    return unreduced_computer.getCellSize();
  }

  size_t PCAFeature::getDescriptorSize()
  {
    return blocks_x()*blocks_y()*cellsPerBlock()*FEAT_DIM;
  }
  
  Size PCAFeature::getWinSize()
  {
    return unreduced_computer.getWinSize();
  }

  Mat PCAFeature::show(const string& title, std::vector< double > reduced_feat)
  {
    // wrap inputs as cv::matrices
    int ncells = reduced_feat.size()/FEAT_DIM;
    Mat mat_reduced_feats(
      ncells,FEAT_DIM,DataType<double>::type,&reduced_feat[0]);
    
    // now multiply by the eigenvalues to reverse what was done before...
    assert(g_pca_reducer->eigenvalues.type() == DataType<float>::type);
    for(int rIter = 0; rIter < mat_reduced_feats.rows; rIter++)
      for(int cIter = 0; cIter < mat_reduced_feats.cols; cIter++)
	mat_reduced_feats.at<float>(rIter,cIter) *= g_pca_reducer->eigenvalues.at<float>(cIter);     
    
    // backproject from reduced to full
    Mat full_feats = g_pca_reducer->backProject(mat_reduced_feats);
    vector<double> v_full = full_feats.reshape(0,full_feats.size().area());   
    
    return unreduced_computer.show(title,v_full);
  }
  
  ///
  ///
  /// SECTION: HoG18+4 version (Deva's version!)
  /// cells per block is 1 therefore blocks = cells in this version.
  /// 
  ///  
  int HOGComputer18p4_General::cellsPerBlock()
  {
    return 1;
  }
  
  /// pack into a form [ori1 ori2 ori .... oriN norm1 ... norm4]
  void HOGComputer18p4_General::compute(const ImRGBZ& im, std::vector< float >& feats)
  {
    vector<float> unreduced; hog18x4mapped.compute(im,unreduced);
    int raw_bins = hog18x4mapped.getNBins();
    int nbins = params::ORI_BINS;
    assert(raw_bins == nbins*4);
    
    /// now, iterate over all the cells in the remapped
    // 18x4 feature, reduce the block, and write the reduced
    // form to the 18p4 feature.
    feats = vector<float>(getDescriptorSize(),0);
    for(int yIter = 0; yIter < blocks_y(); yIter++)
      for(int xIter = 0; xIter < blocks_x(); xIter++)
      {
	// (1) compute the orientation sum
	for(int out_binIter = 0; out_binIter < nbins; out_binIter++)
	{
	  float ori_sum = 0;
	  for(int in_block = 0; in_block < 4; in_block++)
	  {
	    int bin_id = out_binIter + nbins*in_block;
	    ori_sum += unreduced[hog18x4mapped.getIndex(xIter,yIter,0,bin_id)];
	  }
	  feats[getIndex(xIter,yIter,0,out_binIter)] = .5*ori_sum;
	}
	
	// (2) compute the normalization sum
	for(int out_blockIter = 0; out_blockIter < 4; out_blockIter++)
	{
	  float block_sum = 0;
	  for(int in_oriBin = 0; in_oriBin < nbins; in_oriBin++)
	  {
	    int bin_id = in_oriBin + out_blockIter*nbins;
	    block_sum += unreduced[hog18x4mapped.getIndex(xIter,yIter,0,bin_id)];
	  }
	  feats[getIndex(xIter,yIter,0,nbins + out_blockIter)] = .2357*block_sum;
	}
      }
  }
  
  Size HOGComputer18p4_General::getBlockSize()
  {
    return hog18x4mapped.getBlockSize();
  }

  Size HOGComputer18p4_General::getBlockStride()
  {
    return hog18x4mapped.getBlockStride();
  }
  
  Size HOGComputer18p4_General::getCellSize()
  {
    return hog18x4mapped.getCellSize();
  }
  
  size_t HOGComputer18p4_General::getDescriptorSize()
  {
    return getNBins()*cellsPerBlock()*blocks_x()*blocks_y();
  }

  int HOGComputer18p4_General::getNBins()
  {
    return params::ORI_BINS + 4;
  }
  
  Size HOGComputer18p4_General::getWinSize()
  {
    return hog18x4mapped.getWinSize();
  }
  
  HOGComputer18p4_General::HOGComputer18p4_General
  (im_fun use_fun, Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    hog18x4mapped(use_fun,win_size,block_size,block_stride,cell_size)
  {
  }

  std::vector< double > HOGComputer18p4_General::undecorate_feat(std::vector< double > feat)
  {
    // 
    int num_cells = feat.size()/getNBins();
    
    // quadrupple every cell to match the 18*4 feature.
    vector<double> raw_feat(num_cells*4*18,0); 
    int raw_pos = 0;
    for(int cellIter = 0; cellIter < num_cells; cellIter++)
      // we need to write the cell from feat 4 times into
      // raw_feat
      for(int jter = 0; jter < 4; jter++)
	for(int bin = 0; bin < 18; bin++)
	{
	  int feat_idx = bin + cellIter*getNBins();
	  int raw_idx  = bin + cellIter*hog18x4mapped.getNBins() + jter*18;
	  assert(feat_idx >= 0 && feat_idx < feat.size());
	  assert(raw_idx >= 0 && raw_idx < raw_feat.size());
	  raw_feat[raw_idx] = feat[feat_idx];
	}
    return raw_feat;
  }
  
  vector< FeatVis > HOGComputer18p4_General::show_planes(std::vector< double > feat)
  {
    vector<double> raw_feat = undecorate_feat(feat);
    return hog18x4mapped.show_planes(raw_feat);
  }
  
  Mat HOGComputer18p4_General::show(const string& title, std::vector< double > feat)
  {
    vector<double> raw_feat = undecorate_feat(feat);
    return hog18x4mapped.show(title,raw_feat);
  }

  string HOGComputer18p4_General::toString() const
  {
    return "HOGComputer18p4_General";
  }
  
  auto selectDepthFn = [](const ImRGBZ&im) -> const Mat&
  {
    return im.Z;
    //return im.distInvarientDepths();
  };
  
  HOGComputer18p4_Depth::HOGComputer18p4_Depth(
    Size win_size, Size block_size, Size block_stride, Size cell_size) : 
    HOGComputer18p4_General(selectDepthFn,win_size,block_size,block_stride,cell_size)
  {
  }
  
  /// SECTION: COmbo Depth only
  constexpr static double HOA_IMPORTANCE = 2;
  
  ComboComputer_Depth::ComboComputer_Depth(Size win_size, Size block_size, Size block_stride, Size cell_size): FeatureBinCombiner(win_size, block_size, block_stride, cell_size)
  {
    computers.clear();
    
    // for now, use ZHOG and ZAREA
    // Depth HOG
    computers.push_back(shared_ptr<DepthFeatComputer>(
      new HOGComputer18p4_General(selectDepthFn,win_size,block_size,block_stride,cell_size)));
    // HoA
    //computers.push_back(shared_ptr<DepthFeatComputer>(
      //new HOGComputer_Area(win_size,block_size,block_stride,cell_size)));
    
    normalize_weights();
    
    // update the normalized weights
    //weights[1] *= HOA_IMPORTANCE; 
    //assert(std::dynamic_pointer_cast<HOGComputer_Area>(computers[1]));
    
    for(int iter = 0; iter < computers.size(); ++iter)
      log_once(printfpp("ComboComputer_Depth:%d weight = %f",
			   iter,weights[iter]));
  }

  string ComboComputer_Depth::toString() const
  {
      return "ComboComputer_Depth";
  }
  
  /// SECTION: Combo with RGB and Depth
  ComboComputer_RGBPlusDepth::ComboComputer_RGBPlusDepth(Size win_size, Size block_size, Size block_stride, Size cell_size): 
    FeatureBinCombiner(win_size, block_size, block_stride, cell_size)
  {
    computers.clear();
    
    // for now, use ZHOG and ZAREA
    computers.push_back(shared_ptr<DepthFeatComputer>(
      new HOGComputer18p4_General(selectDepthFn,win_size,block_size,block_stride,cell_size)));
    //computers.push_back(shared_ptr<DepthFeatComputer>(
      //new HOGComputer_Area(win_size,block_size,block_stride,cell_size)));
    computers.push_back(shared_ptr<DepthFeatComputer>(
      new HOGComputer18p4_General([](const ImRGBZ&im)-> const Mat&{return im.gray();},win_size,block_size,block_stride,cell_size)));
    //computers.push_back(shared_ptr<DepthFeatComputer>(
      //new FaceFeature(win_size,block_size,block_stride,cell_size)));
    // skin
    //computers.push_back(shared_ptr<DepthFeatComputer>(
      //new SkinFeatureComputer(win_size,block_size,block_stride,cell_size)));
    
    normalize_weights();
    
    //weights[1] *= HOA_IMPORTANCE; // double the weight of HoA
    //assert(std::dynamic_pointer_cast<HOGComputer_Area>(computers[1]));
    //weights[3] = 1; // don't normalize the face feature weight.
    
    for(int iter = 0; iter < computers.size(); ++iter)
      log_once(printfpp("ComboComputer_RGBPlusDepth:%d weight = %f",
			   iter,weights[iter]));
  }
  
  string ComboComputer_RGBPlusDepth::toString() const
  {
      return "ComboComputer_RGBPlusDepth";
  }
}
