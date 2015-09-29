/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_PLATT
#define DD_PLATT

#include <vector>
#include "Detector.hpp"
#include <memory>

namespace deformable_depth
{
  using std::vector;
  
  class LogisticPlatt
  {
  public:
    LogisticPlatt(vector<double>&deci, vector<bool>&label);
    LogisticPlatt() {};
    double prob(double deci);
  public:
    // model parameters
    double A,B;
  private:
    void init(vector<double>&deci, vector<bool>&label);
    void opt(vector<double>&deci, vector<bool>&label);
    double fObj(std::vector< double >& deci, double A, double B) const;
  private:
    // optimization parameters
    double*t;
    double hiTarget, loTarget;
    std::size_t len;
    int maxiter;
    double minstep;
    double sigma;
    double prior1;
    double prior0;
    double fval;
    double fval0;
  };
  const LogisticPlatt EmptyPlatt;
  
  void write(FileStorage&, string&, const std::shared_ptr<LogisticPlatt>&);
  void write(FileStorage& fs, const string& , const LogisticPlatt& platt);
  void read(const FileNode& node, LogisticPlatt& platt,
	    const LogisticPlatt& defaultValue = EmptyPlatt);  
  void read(FileNode, shared_ptr<LogisticPlatt>&, shared_ptr<LogisticPlatt>);
  
  shared_ptr<LogisticPlatt> train_platt
    (ExposedTrainingSet&model,
     vector<shared_ptr<MetaData>> pos_set, 
     vector<shared_ptr<MetaData>> neg_set,
      Model::TrainParams params);
  void platt_test();
}

#endif
