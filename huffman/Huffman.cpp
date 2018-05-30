/**
 * bison 2018年写的一个huffman编码的库
 **/

#include "StdAfx.h"
#include "Huffman.h"

#include <list>
#include <hash_map>
#include <map>
#include <process.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
using namespace std;




BitString::BitString()
{
	len = 0;
	int i;
	for (i = 0; i < sizeof(bits); ++i)
	{
		bits[i] = 0;
	}
}
BitString::~BitString()
{
}
void BitString::push_back(unsigned char oneBit)
{
	int i = (len)/8;
	int j = (len)%8;
	unsigned char * ptr = &bits[i];
	if (oneBit)
	{
		*ptr = *ptr | (1<<j);
	}
	len++;
}
int BitString::read_bit(uint16_t index) const
{
	int i = (index)/8;
	int j = (index)%8;
	const unsigned char * ptr = &bits[i];
	if (*ptr & (1<<j))
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
int BitString::get_bit_num() const
{
	return len;
}
void BitString::increase()
{
	if (len < 1) { return;}

	bool flag = true;//当前全都是1的标记
	int i;
	for (i = 0; i < len; ++i)
	{
		if (read_bit(i) == 0)
		{
			flag = false;
			break;
		}
	}
	if (flag)//全一的话，长度会加1，变成100...0
	{
		int old_len = len;
		len = 0;
		push_back(1);
		for (i = 0; i < old_len; ++i)
		{
			push_back(0);
		}
	}
	else//非全1的情况下，加一长度不会变化
	{
		for (i = len - 1; i >= 0; --i)
		{
			if (read_bit(i) == 0)
			{
				int index = (i)/8;
				int offset = (i)%8;
				unsigned char * ptr = &bits[index];
				*ptr = *ptr | (1<<offset);
				break;
			}
			else
			{
				int index = (i)/8;
				int offset = (i)%8;
				unsigned char * ptr = &bits[index];
				*ptr = *ptr & (~((uint8_t)1<<offset));
			}
		}
	}

}

HuffmanItem::HuffmanItem(uint32_t value, uint32_t freq)
{
	this->value = value;
	this->frequency = freq;

	left = NULL;
	right = NULL;
	parent = NULL;
}
HuffmanItem::~HuffmanItem()
{
}
HuffmanItem::HuffmanItem()
{

	left = NULL;
	right = NULL;
	parent = NULL;
}
HuffmanItem & HuffmanItem::operator=(const HuffmanItem & a)
{
	this->value = a.value;
	this->frequency = a.frequency;
	//this->code = a.code;
	//this->ccode = a.ccode

	left = NULL;
	right = NULL;
	parent = NULL;

	return *this;
}
int HuffmanItem::item_cmp1(const void*a, const void*b)
{
	HuffmanItem *p1 = *(HuffmanItem **)a;
	HuffmanItem *p2 = *(HuffmanItem **)b;
	if (p1->frequency < p2->frequency)
	{
		return -1;
	}
	if (p1->frequency == p2->frequency)
	{
		return 0;
	}
	return 1;
	
}
int HuffmanItem::item_cmp2(const void*a, const void*b)
{
	HuffmanItem *p1 = *(HuffmanItem **)a;
	HuffmanItem *p2 = *(HuffmanItem **)b;
	if (p1->code.get_bit_num() < p2->code.get_bit_num())
	{
		return -1;
	}
	if (p1->code.get_bit_num() > p2->code.get_bit_num())
	{
		return 1;
	}
	if (p1->code.get_bit_num() == p2->code.get_bit_num())
	{
		if (p1->value < p2->value)
		{
			return -1;
		}
		if (p1->value == p2->value)
		{
			return 0;
		}
		if (p1->value > p2->value)
		{
			return 1;
		}
	}
	
}
Huffman::Huffman()
{
	leaf = NULL;
}
int Huffman::build(const HuffmanItem items[], uint32_t item_num)
{
	if (item_num < 2 || items == NULL) { return -1;}
	if ( leaf != NULL) { return -2;}

	//用来排序的数组，pointer，里面存储的是指针，指向树的节点
	HuffmanItem * * pointer = (HuffmanItem **)calloc(item_num, sizeof(HuffmanItem *));
	leaf = (HuffmanItem **)calloc(item_num, sizeof(HuffmanItem *));
	this->leaf_num = item_num;
	int i; 
	//创建所有叶子节点
	for (i = 0; i < item_num; ++i)
	{
		pointer[i] = new HuffmanItem();
		*(pointer[i]) = items[i];
		leaf[i] = pointer[i];
	}
	int pointer_nr = item_num;
	while (pointer_nr >= 2)
	{
		//按 frequency升序排序
		qsort(pointer, pointer_nr, sizeof(HuffmanItem *), HuffmanItem::item_cmp1);

		//取频次最低的两个节点，合并为一个子树，父节点频次等于他们的和
		HuffmanItem * parent = new HuffmanItem();
		parent->left = pointer[0];
		parent->right = pointer[1];
		pointer[0]->parent = parent;
		pointer[1]->parent = parent;
		parent->frequency = parent->left->frequency + parent->right->frequency;

		//父节点加入排序的数组中，两个子节点不参与排序了
		pointer[0] = parent;
		memmove(&pointer[1], &pointer[2], sizeof(HuffmanItem *)*(pointer_nr-2));

		pointer_nr--;
	}
	
	HuffmanItem  *root = pointer[0];

	//从每个叶子节点往根部上溯，获得0、1编码
	for (i = 0; i < item_num; ++i)
	{
		HuffmanItem * son = leaf[i];
		HuffmanItem * parent = son->parent;
		list<int> bits;

		while (parent != NULL)
		{
			if (parent->left == son)
			{
				bits.push_front(0);
			}
			else
			{
				bits.push_front(1);
			}

			son = parent;
			parent = son->parent;
		}
		list<int>::iterator it;
		for (it = bits.begin(); it != bits.end(); it++)
		{
			leaf[i]->code.push_back(*it);
		}
	

	}
	
	delete[] pointer;//释放掉排序用的数组
	release_tree(root);//树没有用了，释放掉，保留叶子节点即可

	print_leaf();
	canonicalize();//范式化
	print_leaf();
	
	return 0;
}
void Huffman::release_tree(HuffmanItem* root)//释放一棵树的中间节点，保留叶子节点
{
	list<HuffmanItem*> pointers;
	pointers.push_back(root);
	while (pointers.size() > 0)
	{
		HuffmanItem* p = pointers.back();
		pointers.pop_back();
		if (p->left != NULL)
		{
			pointers.push_back(p->left);
		}
		if (p->right != NULL)
		{
			pointers.push_back(p->right);
		}
		if (p->left != NULL || p->right != NULL)//中间节点就释放
		{
			delete p;
		}

	}
}
Huffman::~Huffman()
{
	if (leaf == NULL || leaf_num < 2) { return;}

	//释放叶子节点和指针数组本身
	int i;
	for (i = 0; i < leaf_num; ++i)
	{
		delete leaf[i];
	}
	delete[] leaf;
}

void Huffman::print_leaf()
{
	printf("print leaf info...\n");
	int i;
	for (i = 0; i < leaf_num; ++i)
	{
		printf("%d:\n\tcode: ", leaf[i]->value);
		int j;
		for (j = 0; j < leaf[i]->code.get_bit_num(); ++j)
		{
			printf("%d", leaf[i]->code.read_bit(j));
		}
		printf("\n\tccode:");
		for (j = 0; j < leaf[i]->ccode.get_bit_num(); ++j)
		{
			printf("%d", leaf[i]->ccode.read_bit(j));
		}
		printf("\n");
		
	}
}

int Huffman::canonicalize()
{
	if (leaf == NULL || leaf_num < 2) { return -1;}
	qsort(leaf, leaf_num, sizeof(HuffmanItem*), HuffmanItem::item_cmp2);
	
	//范式3：码字长度最小的第一个编码从0开始，长度对齐原码字
	int i, index;
	for (i = 0; i < leaf[0]->code.get_bit_num(); ++i)
	{
		leaf[0]->ccode.push_back(0);
	}
	for (index = 1; index < leaf_num; ++index)
	{
		BitString bs = leaf[index-1]->ccode;
		bs.increase();//范式1
		if (bs.get_bit_num() < leaf[index]->code.get_bit_num())//范式2
		{
			bs.push_back(0);
		}
		leaf[index]->ccode = bs;
	}

	return 0;
}
int Huffman::encode(const uint32_t values[], uint32_t val_num, 
			unsigned char bits[], uint32_t & bit_num)
{
	if (leaf == NULL || leaf_num < 2) { return -1;}
	if (values == NULL || bits == NULL) { return -2;}

	//建立方便速查的map，输入被编码的值，输出码字
	hash_map<uint32_t, BitString> value2code;
	int i;
	for (i = 0; i < leaf_num; ++i)
	{
		value2code.insert(pair<uint32_t, BitString>(leaf[i]->value, leaf[i]->ccode) );
	}
	uint32_t bit_index = 0;
	for (i = 0; i < val_num ; ++i)
	{
		hash_map<uint32_t, BitString>::iterator it  = value2code.find(values[i]);
		if (it == value2code.end())
		{
			return -3;
		}
		const BitString & bs = it->second;

		//把查到的码字一个一个bit追加到bits数组的末尾
		int j;
		for (j = 0; j < bs.get_bit_num(); ++j)
		{
			if (bit_index >= bit_num) { return -4;}

			int byte_index = bit_index / 8;
			int offset_in_byte = bit_index % 8;
			unsigned char * ptr = &bits[byte_index];

			int b = bs.read_bit(j);
			if (b)
			{
				*ptr = *ptr | (1 << offset_in_byte);
			}
			else
			{
				*ptr = *ptr & (~(1 << offset_in_byte));
			}
			bit_index++;
		}
	}
	bit_num = bit_index;
	return 0;
}
int Huffman::decode(uint32_t values[], uint32_t &val_num, 
			const unsigned char  bits[], uint32_t  bit_num)
{
	if (leaf == NULL || leaf_num < 2) { return -1;}
	if (values == NULL || bits == NULL) { return -2;}
	//leaf数组里的item，是按照码字长度升序排列的，相同长度的码字，
	// 按value的升序排列
	vector<int> counts((size_t)(leaf_num), (int)0); //大小为leaf_num的数组，元素初始化为0
	vector<uint32_t> symbols;
	int i;
	for (i = 0; i < leaf_num; ++i)
	{
		uint32_t sym = leaf[i]->value;
		int len = leaf[i]->ccode.get_bit_num();
		counts[len]++;

		symbols.push_back(leaf[i]->value);
	}
	
	int bit_index = 0;
	int val_index = 0;

	while (bit_index < bit_num && val_index < val_num)
	{

		int len;            /* current number of bits in code */
		int code;           /* len bits being decoded */
		int first;          /* first code of length len */
		int count;          /* number of codes of length len */
		int index;          /* index of first code of length len in symbol table */

		code = first = index = 0;
	

		int maxbitsnum = leaf[leaf_num-1]->ccode.get_bit_num();
		for (len = 1; len <= maxbitsnum ; len++) {

			if (bit_index >= bit_num) { return -3;}

			//读取下一个bit
			int byte_index = bit_index / 8;
			int byte_offset = bit_index % 8;
			int bit_value = bits[byte_index] & (1<<byte_offset);
			bit_index++;

			if (bit_value)
			{
				code = code | 1;
			}

        
			if (code - counts[len] < first)       /* if length len, return symbol */
			{
				
				values[val_index++] = symbols[index + (code - first)];
				break;
				
			}
			index += counts[len];                 /* else update for next length */
			first += counts[len];
			first <<= 1;
			code <<= 1;
		}
	}
	if (bit_index == bit_num)
	{
		val_num = val_index;
		return 0;  
	}
	else
	{
		return -4;
	}


}
int Huffman::serialize(unsigned char * buffer, uint32_t & buflen)
{
	//序列化的格式：叶子节点数 + [ 值 + 码字长度 ]
	if (leaf == NULL || leaf_num < 2) { return -1;}
	int total_len = (sizeof(uint32_t) + sizeof(uint8_t)) * leaf_num + sizeof(leaf_num);
	if (total_len > buflen) { return -2;}

	int offset = 0;

	*(uint32_t*)(buffer+offset) = leaf_num;
	offset += sizeof(uint32_t);

	int i;
	for (i = 0; i < leaf_num; ++i)
	{
		*(uint32_t*)(buffer+offset) = leaf[i]->value;
		offset += sizeof(uint32_t);

		*(uint8_t*)(buffer+offset) = leaf[i]->ccode.get_bit_num();
		offset += sizeof(uint8_t);
	}
	buflen = offset;
	return 0;
}
int Huffman::deserialize(const unsigned char * buffer, uint32_t  buflen)
{
	if (leaf != NULL) { return -1;}
	if (buffer == NULL || buflen < sizeof(uint32_t) )
	{
		return -2;
	}
	leaf_num = *(uint32_t*)buffer;
	int total_len = (sizeof(uint32_t) + sizeof(uint8_t)) * leaf_num + sizeof(leaf_num);
	if (buflen != total_len||leaf_num < 2)
	{
		return -3;
	}
	leaf = (HuffmanItem**)calloc(leaf_num, sizeof(HuffmanItem*));
	int i;
	int offset = sizeof(uint32_t);
	for (i = 0; i < leaf_num; ++i)
	{
		uint32_t value = *(uint32_t*)(buffer+offset);
		offset += sizeof(uint32_t);

		uint8_t bitnum = *(uint8_t*)(buffer+offset);
		offset += sizeof(uint8_t);

		leaf[i] = new HuffmanItem();
		leaf[i]->value = value;
		leaf[i]->code.len = bitnum;// code不是完整的状态和数据，借用一下他的长度字段
	}
#if 0
	//范式3：码字长度最小的第一个编码从0开始，长度对齐原码字
	int  index;
	for (i = 0; i < leaf[0]->code.get_bit_num(); ++i)
	{
		leaf[0]->ccode.push_back(0);
	}
	for (index = 1; index < leaf_num; ++index)
	{
		BitString bs = leaf[index-1]->ccode;
		bs.increase();//范式1
		if (bs.get_bit_num() < leaf[index]->code.get_bit_num())//范式2
		{
			bs.push_back(0);
		}
		leaf[index]->ccode = bs;
	}
#else
	canonicalize();
#endif
	for (i = 0; i < leaf_num; ++i)
	{
		leaf[i]->code.len = 0;//长度字段清掉
	}
	return 0;

}
void Huffman::bits2string(const unsigned char  bits[], uint32_t  bit_num, string &s)
{
	if (bits == NULL) { return;}

	s = "";
	int i; 
	for (i = 0; i < bit_num; ++i)
	{
		int byte_index = i / 8;
		int offset_in_byte = i % 8;
		unsigned char b = (bits[byte_index]);
		if (b & (1 << offset_in_byte))
		{
			s.append("1");
		}
		else
		{
			s.append("0");
		}
	}
}
int test()
{
	srand(_getpid());
	HuffmanItem items[5];
	for (int i =0; i  < 5; ++i)
	{
		HuffmanItem a(i, i+1);
		items[i] = a;
	}
	Huffman hh, h;
	hh.build(items, 5);

	unsigned char serialbuf[100];
	uint32_t serialbuflen = sizeof(serialbuf);
	if (hh.serialize(serialbuf, serialbuflen)) { return -1;}
	printf("序列化长度：%u\n", serialbuflen);
	if (h.deserialize(serialbuf, serialbuflen)) { return -1;}



	printf("原始串:\n");
	uint32_t v[20];
	for (int i =0; i  < sizeof(v)/sizeof(uint32_t); ++i)
	{
		v[i] = rand() % 5;
		printf("%d", v[i]);
	}
	printf("\n");

	unsigned char bits[100];
	uint32_t bitnum = 800;
	if (h.encode(v, sizeof(v)/sizeof(uint32_t), 
		     bits, bitnum) < 0) { return -1;}
	string s;
	Huffman::bits2string(bits, bitnum, s);
	printf(">>%d  %s\n", s.length(), s.c_str());

	uint32_t vn = sizeof(v)/sizeof(uint32_t);
	h.decode(v, vn, bits, bitnum);

	printf("解码串:\n");
	for (int i =0; i  < vn; ++i)
	{
		printf("%d", v[i]);
	}
	printf("\n");
}

