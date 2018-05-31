/**
 * bison 2018年写的一个huffman编码的库
 **/

#ifndef _HUFFMAN_H_INCLUDED_
#define _HUFFMAN_H_INCLUDED_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <string>


#define MAX_BITS_NUM (80)

//用来表示huffman编码的bit串
class BitString
{
private:
	unsigned char bits[MAX_BITS_NUM / 8];//如果要保存很长很长的bit，那这里改为deque<uchar>比较好，按需分配内存，可能还需要实现operator=
	uint16_t len;
public:
	BitString();
	virtual ~BitString();
	void push_back(unsigned char oneBit);//在串的末尾追加一个bit
	int read_bit(uint16_t index) const ;//读取第index个bit，成功返回0或者1
	int get_bit_num() const;//获得有效bit的个数
	void increase();//把bit当作一个二进制整数，并加一
	friend class Huffman;


};

// 用来表示huffman树的节点
// 中间节点，value和code , ccode字段没有意义
class HuffmanItem
{
private:
	
	uint32_t value;//我认为要编码的实际数据，都可以用一个整数编号起来
	BitString code;	//该节点的huffman编码
	BitString ccode; //该节点的范式huffman编码
	uint32_t frequency;//出现频率

	//编码过程中用来建树的指针
	HuffmanItem * left;
	HuffmanItem * right;
	HuffmanItem * parent;

	friend class Huffman;
	static int item_cmp1(const void*a, const void*b);//根据frequency比较，用于编码过程的排序
	static int item_cmp2(const void*a, const void*b);//根据码字长度比较，用于范式化过程中的排序

public:
	HuffmanItem();
	HuffmanItem(uint32_t value, uint32_t freq);//输入要编码的数据和它出现的频次
	
	virtual ~HuffmanItem();

	HuffmanItem & operator=(const HuffmanItem & a);
	

} ;

//哈夫曼编码
class Huffman
{
private: 
	HuffmanItem  ** leaf;//所有叶子节点的指针的数组
	uint32_t leaf_num; //叶子节点的个数，也就是要被编码的值的个数
	Huffman & operator=(const Huffman & a);
	int canonicalize();//范式化，详细见范式哈夫曼
	void print_leaf();//打印调试信息
	void release_tree(HuffmanItem* root);//释放哈夫曼树，这棵树只是在编码过程中用一下，码字确定后就可以释放了
public:
	Huffman();
	int build(const HuffmanItem items[], uint32_t item_num);//根据一系列值和他们的频次，建立范式哈夫曼
	virtual ~Huffman();
	int encode(const uint32_t values[], uint32_t val_num, 
			unsigned char  bits[], uint32_t & bit_num);//对存储在values里的value_num个值，进行编码，bit存储到bits里
	int decode(uint32_t values[], uint32_t &val_num, 
			const unsigned char  bits[], uint32_t  bit_num);//反过来，解码
	static void bits2string(const unsigned char  bits[], uint32_t  bit_num, std::string &s);//把bit数组以字符串方式表示出来
	int serialize(unsigned char * buf, uint32_t & buflen);//把范式哈夫曼序列化，方便存储和传输
	int deserialize(const unsigned char * buf, uint32_t  buflen);//反序列化

};

int test(const char * filename);
#endif
