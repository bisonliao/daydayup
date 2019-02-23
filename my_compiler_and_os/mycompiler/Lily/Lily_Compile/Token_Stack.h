#ifndef _Token_STACK_H_
#define  _Token_STACK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"


class Token_Stack 
{
public:
	Token_Stack();
	Token_Stack(Token_Stack &ss);
	~Token_Stack();	
	Token_Stack&operator=(const Token_Stack&ss);
    	bool push(const Token & ele);
	bool pop(Token& ele);
	bool peek(Token& ele);
	bool isEmpty();

	void BeginPeekFrmTop();
	bool PeekNextFrmTop(Token& ele); 

	int getSize();
private:
	Token * m_pHdr;//保存元素的缓冲区的头指针
	int m_nEleNum;//元素个数
	long m_nBufSize;//缓冲去最大能保存的元素个数   
	int m_nPeekIndex;
private:   	
	int enlarge();//当缓冲区不够大时，重新分配，扩大缓冲区	
	bool isFull();//当前缓冲区是否用完了	
};

#endif
