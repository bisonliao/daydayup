#ifndef _Labels_STACK_H_
#define  _Labels_STACK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"


class Labels_Stack 
{
public:
	Labels_Stack();
	Labels_Stack(Labels_Stack &ss) ;
	~Labels_Stack();	
	Labels_Stack&operator=(const Labels_Stack&ss) ;
    	bool push(const Labels & ele);
	bool pop(Labels& ele);
	bool peek(Labels& ele);
	bool isEmpty();
	int getSize();
private:
	Labels * m_pHdr;//保存元素的缓冲区的头指针
	int m_nEleNum;//元素个数
	long m_nBufSize;//缓冲去最大能保存的元素个数   
private:   	
	int enlarge();//当缓冲区不够大时，重新分配，扩大缓冲区	
	bool isFull();//当前缓冲区是否用完了	
};

#endif
