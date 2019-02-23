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
	Labels * m_pHdr;//����Ԫ�صĻ�������ͷָ��
	int m_nEleNum;//Ԫ�ظ���
	long m_nBufSize;//����ȥ����ܱ����Ԫ�ظ���   
private:   	
	int enlarge();//��������������ʱ�����·��䣬���󻺳���	
	bool isFull();//��ǰ�������Ƿ�������	
};

#endif
