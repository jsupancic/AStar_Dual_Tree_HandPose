/**
 * Copyright 2013: James Steven Supancic III
 **/

#ifndef DD_HASHMAT
#define DD_HASHMAT

#include <string>
#define use_speed_ 0
#include <opencv2/opencv.hpp>
#include <openssl/md5.h>

namespace deformable_depth
{
  using namespace std;
  using namespace cv;
  
  struct MD5_HASH
  {
  public:
    uchar data[MD5_DIGEST_LENGTH+1/*in bytes*/];
  public:
    MD5_HASH();
    MD5_HASH operator^(const MD5_HASH&other) const;
  };  

  size_t hash_code(const Mat&m);
  string hashMat(const Mat&mat);  
  string hash(vector<string> strings);
  MD5_HASH hash(const vector<vector<double>> &vecs);
  MD5_HASH hash(const vector<double> &vec);
  string hash2string(MD5_HASH&hash_val);
}

#endif
