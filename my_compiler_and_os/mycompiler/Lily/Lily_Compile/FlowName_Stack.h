#ifndef _FlowName_STACK_H_
#define  _FlowName_STACK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"


class FlowName_Stack 
{
public:
	FlowName_Stack();
	FlowName_Stack(FlowName_Stack &ss);
	~FlowName_Stack();	
	FlowName_Stack&operator=(const FlowName_Stack&ss);
    	bool push(const FlowName & ele);
	bool pop(FlowName& ele);
	bool peek(FlowName& ele);
	bool isEmpty();
	bool contain(FlowName& ele);

	int getSize();
private:
	FlowName * m_pHdr;//保存元素的缓冲区的头指针
	int m_nEleNum;//元素个数
	long m_nBufSize;//缓冲去最大能保存的元素个数   
	int m_nPeekIndex;
private:   	
	int enlarge();//当缓冲区不够大时，重新分配，扩大缓冲区	
	bool isFull();//当前缓冲区是否用完了	
};

#endif
