#ifndef _Variable_STACK_H_
#define  _Variable_STACK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Variable.h"


class Variable_Stack 
{
public:
	Variable_Stack() ;
	Variable_Stack(Variable_Stack &ss) ;
	~Variable_Stack();	
	Variable_Stack&operator=(const Variable_Stack&ss) ;
    bool push(const Variable & ele);
	bool pop(Variable& ele);
	bool peek(Variable& ele);
	bool isEmpty();
	int getSize();
	/*��ջ����ջ�ײ鿴depth�����������ĳ�����������ֵ���name����ô���سɹ�*/
	bool FindVarByNameFrmTop(const AnsiString& name, int depth, Variable& ele);
	/*����FindVarByNameFrmTop, ���ǽ������ĵ�ַ��16���Ʊ�ʾ,������ַ�����*/
	bool FindVarAddrByNameFrmTop(const AnsiString& name, int depth, Variable& ele);
	bool ModifyVarByNameFrmTop(const AnsiString& name, int depth, const Variable& ele);
	void removeAll();
private:
	Variable * m_pHdr;//����Ԫ�صĻ�������ͷָ��
	int m_nEleNum;//Ԫ�ظ���
	long m_nBufSize;//����ȥ����ܱ����Ԫ�ظ���   
private:   	
	int enlarge();//��������������ʱ�����·��䣬���󻺳���	
	bool isFull();//��ǰ�������Ƿ�������	
};

#endif
