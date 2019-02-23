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
	/*从栈顶往栈底查看depth个变量，如果某个变量的名字等于name，那么返回成功*/
	bool FindVarByNameFrmTop(const AnsiString& name, int depth, Variable& ele);
	/*类似FindVarByNameFrmTop, 但是将变量的地址用16进制表示,存放在字符串里*/
	bool FindVarAddrByNameFrmTop(const AnsiString& name, int depth, Variable& ele);
	bool ModifyVarByNameFrmTop(const AnsiString& name, int depth, const Variable& ele);
	void removeAll();
private:
	Variable * m_pHdr;//保存元素的缓冲区的头指针
	int m_nEleNum;//元素个数
	long m_nBufSize;//缓冲去最大能保存的元素个数   
private:   	
	int enlarge();//当缓冲区不够大时，重新分配，扩大缓冲区	
	bool isFull();//当前缓冲区是否用完了	
};

#endif
