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
	Token * m_pHdr;//����Ԫ�صĻ�������ͷָ��
	int m_nEleNum;//Ԫ�ظ���
	long m_nBufSize;//����ȥ����ܱ����Ԫ�ظ���   
	int m_nPeekIndex;
private:   	
	int enlarge();//��������������ʱ�����·��䣬���󻺳���	
	bool isFull();//��ǰ�������Ƿ�������	
};

#endif
