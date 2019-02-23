#ifndef __VARIABLE_H__
#define __VARIABLE_H__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "common.h"
#include "AnsiString.h"
#include "MemBlock.h"
#include "expr_type.h"


class Variable
{
private:
	int m_nType;			/*类型*/
	AnsiString m_asName;	/*名字*/
	/*各种不同的值*/
public:
	AnsiString m_StringValue;	/*字符串*/
	int	m_IntegerValue;			/*整数*/
	double m_FloatValue;		/*浮点数*/
	MemBlock m_MemBlockValue;	/*内存块*/

public:
	Variable();	
	Variable(const Variable &v);	
	~Variable();
	const Variable &operator=(const Variable &v);
	void clear();	/*如果为TYPE_STRING或TYPE_MEMBLOCK型，释放大部分空间*/

	const AnsiString& getString() const {return m_StringValue;};
	int	getInteger() const 			{return m_IntegerValue;};
	double getFloat() const			{return m_FloatValue;};
	const MemBlock& getMemBlock() const {return m_MemBlockValue;};

	int getType() const				{return m_nType;};
	const AnsiString& getName()	{return m_asName;};

	void setString(const AnsiString & s) 	{m_StringValue = s;};
	void setInteger(int i)			{m_IntegerValue = i;};
	void setFloat(double d)			{m_FloatValue = d;};
	void setMemBlock(const MemBlock & mb) {m_MemBlockValue = mb;};

	int setType(int type);
	void setName(const AnsiString &s)	{m_asName = s;};
	
	AnsiString toString() const;
};

#endif
