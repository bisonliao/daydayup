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
	int * m_pHdr;//����Ԫ�صĻ�������ͷָ��
	int m_nEleNum;//Ԫ�ظ���
	long m_nBufSize;//����ȥ����ܱ����Ԫ�ظ���   
private:   	
	int enlarge();//��������������ʱ�����·��䣬���󻺳���	
	bool isFull();//��ǰ�������Ƿ�������	
};

#endif
