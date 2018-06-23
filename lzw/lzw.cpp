/*
 *   lzw压缩编码的简单实现，实际应用中lzw会比这个复杂很多
 *   
 *   代码不是我写的，从下面的网站拷贝的
 *   http://rosettacode.org/wiki/LZW_compression
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <map>

using namespace std;
 
// Compress a string to a list of output symbols.
// The result will be written to the output iterator
// starting at "result"; the final iterator is returned.
template <typename Iterator>
Iterator compress(const std::string &uncompressed, Iterator result) {
  // Build the dictionary.
  int dictSize = 256;
  std::map<std::string,int> dictionary;
  for (int i = 0; i < 256; i++)
    dictionary[std::string(1, i)] = i;
 
  std::string w;
  for (std::string::const_iterator it = uncompressed.begin();
       it != uncompressed.end(); ++it) {
    char c = *it;
    std::string wc = w + c;
    if (dictionary.count(wc))
      w = wc;
    else {
      *result++ = dictionary[w];
      // Add wc to the dictionary.
      dictionary[wc] = dictSize++;
      w = std::string(1, c);
    }
  }
 
  // Output the code for w.
  if (!w.empty())
    *result++ = dictionary[w];
  return result;
}
 
// Decompress a list of output ks to a string.
// "begin" and "end" must form a valid range of ints
template <typename Iterator>
std::string decompress(Iterator begin, Iterator end) {
  // Build the dictionary.
  int dictSize = 256;
  std::map<int,std::string> dictionary;
  for (int i = 0; i < 256; i++)
    dictionary[i] = std::string(1, i);
 
  std::string w(1, *begin++);
  std::string result = w;
  std::string entry;
  for ( ; begin != end; begin++) {
    int k = *begin;
    if (dictionary.count(k))
      entry = dictionary[k];
    else if (k == dictSize)
      entry = w + w[0];
    else
      throw "Bad compressed k";
 
    result += entry;
 
    // Add w+entry[0] to the dictionary.
    dictionary[dictSize++] = w + entry[0];
 
    w = entry;
  }
  return result;
}
 
 
int test(const char *filename) {

  static unsigned char buffer[1024*1024];
  FILE * fp = fopen(filename, "rb");
  int offset = 0;
  while (1)
  {
	 int len = fread(buffer + offset, 1, sizeof(buffer)-offset, fp);
	 if (len <= 0)
	 {
		break;
	 }
	 offset += len;
  }
  fclose(fp);
  fp = NULL;
  std::string tocompress((const char*)buffer, (size_t)offset);

  std::vector<int> compressed;
  compress(tocompress, std::back_inserter(compressed));
  printf("code number:%d\n", compressed.size());

  //这个时候，根据compressed的size，就可以决定用多少bit来表示一个编码，并估计
  //压缩率、写入文件中。当然，文件头部要记录多少bit一个整数的信息

  copy(compressed.begin(), compressed.end(), std::ostream_iterator<int>(std::cout, ","));
  std::cout << std::endl;
  std::string decompressed = decompress(compressed.begin(), compressed.end());
  //std::cout << decompressed << std::endl;
  char newfilename[255];
  snprintf(newfilename, sizeof(newfilename), "%s.new", filename);
  fp = fopen(newfilename, "wb+");
  fwrite(decompressed.c_str(), 1, decompressed.length(), fp);
  fclose(fp);
 
  return 0;
}

int main(int argc, char ** argv)
{
	if (argc < 2)
	{
		return 0;
	}
	test(argv[1]);
	return 0;
}
