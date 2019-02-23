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
	FlowName * m_pHdr;//����Ԫ�صĻ�������ͷָ��
	int m_nEleNum;//Ԫ�ظ���
	long m_nBufSize;//����ȥ����ܱ����Ԫ�ظ���   
	int m_nPeekIndex;
private:   	
	int enlarge();//��������������ʱ�����·��䣬���󻺳���	
	bool isFull();//��ǰ�������Ƿ�������	
};

#endif
