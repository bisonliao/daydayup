/**
 * bison 2018��д��һ��huffman����Ŀ�
 **/

#ifndef _HUFFMAN_H_INCLUDED_
#define _HUFFMAN_H_INCLUDED_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <string>


#define MAX_BITS_NUM (80)

//������ʾhuffman�����bit��
class BitString
{
private:
	unsigned char bits[MAX_BITS_NUM / 8];//���Ҫ����ܳ��ܳ���bit���������Ϊdeque<uchar>�ȽϺã���������ڴ棬���ܻ���Ҫʵ��operator=
	uint16_t len;
public:
	BitString();
	virtual ~BitString();
	void push_back(unsigned char oneBit);//�ڴ���ĩβ׷��һ��bit
	int read_bit(uint16_t index) const ;//��ȡ��index��bit���ɹ�����0����1
	int get_bit_num() const;//�����Чbit�ĸ���
	void increase();//��bit����һ������������������һ
	friend class Huffman;


};

// ������ʾhuffman���Ľڵ�
// �м�ڵ㣬value��code , ccode�ֶ�û������
class HuffmanItem
{
private:
	
	uint32_t value;//����ΪҪ�����ʵ�����ݣ���������һ�������������
	BitString code;	//�ýڵ��huffman����
	BitString ccode; //�ýڵ�ķ�ʽhuffman����
	uint32_t frequency;//����Ƶ��

	//�������������������ָ��
	HuffmanItem * left;
	HuffmanItem * right;
	HuffmanItem * parent;

	friend class Huffman;
	static int item_cmp1(const void*a, const void*b);//����frequency�Ƚϣ����ڱ�����̵�����
	static int item_cmp2(const void*a, const void*b);//�������ֳ��ȱȽϣ����ڷ�ʽ�������е�����

public:
	HuffmanItem();
	HuffmanItem(uint32_t value, uint32_t freq);//����Ҫ��������ݺ������ֵ�Ƶ��
	
	virtual ~HuffmanItem();

	HuffmanItem & operator=(const HuffmanItem & a);
	

} ;

//����������
class Huffman
{
private: 
	HuffmanItem  ** leaf;//����Ҷ�ӽڵ��ָ�������
	uint32_t leaf_num; //Ҷ�ӽڵ�ĸ�����Ҳ����Ҫ�������ֵ�ĸ���
	Huffman & operator=(const Huffman & a);
	int canonicalize();//��ʽ������ϸ����ʽ������
	void print_leaf();//��ӡ������Ϣ
	void release_tree(HuffmanItem* root);//�ͷŹ��������������ֻ���ڱ����������һ�£�����ȷ����Ϳ����ͷ���
public:
	Huffman();
	int build(const HuffmanItem items[], uint32_t item_num);//����һϵ��ֵ�����ǵ�Ƶ�Σ�������ʽ������
	virtual ~Huffman();
	int encode(const uint32_t values[], uint32_t val_num, 
			unsigned char  bits[], uint32_t & bit_num);//�Դ洢��values���value_num��ֵ�����б��룬bit�洢��bits��
	int decode(uint32_t values[], uint32_t &val_num, 
			const unsigned char  bits[], uint32_t  bit_num);//������������
	static void bits2string(const unsigned char  bits[], uint32_t  bit_num, std::string &s);//��bit�������ַ�����ʽ��ʾ����
	int serialize(unsigned char * buf, uint32_t & buflen);//�ѷ�ʽ���������л�������洢�ʹ���
	int deserialize(const unsigned char * buf, uint32_t  buflen);//�����л�

};

int test(const char * filename);
#endif
