#ifndef _int_STACK_H_
#define  _int_STACK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"


class int_Stack 
{
public:
	int_Stack();
	int_Stack(int_Stack &ss) ;
	~int_Stack();	
	int_Stack&operator=(const int_Stack&ss) ;
    	bool push(const int & ele);
	bool pop(int& ele);
	bool peek(int& ele);
	bool isEmpty();
	int getSize();
private:
	int * m_pHdr;//保存元素的缓冲区的头指针
	int m_nEleNum;//元素个数
	long m_nBufSize;//缓冲去最大能保存的元素个数   
private:   	
	int enlarge();//当缓冲区不够大时，重新分配，扩大缓冲区	
	bool isFull();//当前缓冲区是否用完了	
};

#endif
