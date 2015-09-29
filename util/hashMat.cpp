/**
 * Copyright 2013: James Steven Supancic III
 **/

#include "hashMat.hpp"
#include <openssl/md5.h>
#include <boost/functional/hash.hpp>
#include "util.hpp"

namespace deformable_depth
{
  /// hashing with OpenSSL
  MD5_HASH::MD5_HASH()
  {
    memset(data,0,MD5_DIGEST_LENGTH+1);
  }
  MD5_HASH MD5_HASH::operator^(const MD5_HASH&other) const
  {
    MD5_HASH result;
    for(int iter = 0; iter < MD5_DIGEST_LENGTH; iter++)
      result.data[iter] = data[iter] ^ other.data[iter];
    return result;
  }
    
  static void hash(const uchar*data,size_t numel,MD5_HASH&hash_val)
  {
    MD5(data,numel,(uchar*)hash_val.data);
    hash_val.data[MD5_DIGEST_LENGTH] = 0;
  }

  static MD5_HASH md5_mat(const Mat&mat)
  {
    const void * data = mat.ptr(0);
    // mxGetNumberOfElements
    size_t numel = mat.size().area();
    // mxGetElementSize
    size_t elSz = mat.elemSize();

    // compute hash
    MD5_HASH hash_val; hash((uchar*)data,elSz*numel,hash_val);

    return hash_val;
  }
  
  string hash2string(MD5_HASH&hash_val)
  {
    // extract hash to string
    // 1 8-bit byte 
    // 4 bits per hex char.
    char hashString[2*MD5_DIGEST_LENGTH+1];
    for(int iter = 0; iter < MD5_DIGEST_LENGTH; iter++)
    {
	snprintf(hashString+2*iter,3,"%02x",hash_val.data[iter]);
	//mexPrintf("hash[iter] = %02x\n",hash[iter]);
	//mexPrintf("hashString = %s\n",hashString);
    }
    hashString[2*MD5_DIGEST_LENGTH] = 0;

    // return
    return string(hashString);    
  }

  
  
  size_t hash_code(const Mat&m)
  {
    MD5_HASH h = md5_mat(m);
    size_t t;
    for(int iter = 0; iter < MD5_DIGEST_LENGTH; ++iter)
    {
      for(int jter = 0; jter < 8; ++jter)
	t = rol(t);
      t += h.data[iter];
    }
    return t;
  }
  
  string hashMat(const Mat& mat)
  {
    // return string form
    MD5_HASH md5 = md5_mat(mat);
    return hash2string(md5);
  }  
  
  /// hashing with boost?
  string hash(vector<string> strings)
  {
    boost::hash<vector<string> > hashFn;
    ostringstream oss;
    oss << hashFn(strings);
    return oss.str();
  }
  
  MD5_HASH hash(const vector< double >& vec)
  {
    MD5_HASH hash_value; 
    hash((const uchar*)&vec[0],vec.size()*sizeof(double),hash_value);
    return hash_value;
  }
  
  MD5_HASH hash(const vector<vector<double>>&vecs)
  {
    MD5_HASH hash_value;
    
    for(const vector<double>&vec : vecs)
    {
      MD5_HASH partial_hash = hash(vec);
      hash_value = hash_value ^ partial_hash;
    }
    
    return hash_value;
  }
}
