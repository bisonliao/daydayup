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
	int m_nType;			/*����*/
	AnsiString m_asName;	/*����*/
	/*���ֲ�ͬ��ֵ*/
public:
	AnsiString m_StringValue;	/*�ַ���*/
	int	m_IntegerValue;			/*����*/
	double m_FloatValue;		/*������*/
	MemBlock m_MemBlockValue;	/*�ڴ��*/

public:
	Variable();	
	Variable(const Variable &v);	
	~Variable();
	const Variable &operator=(const Variable &v);
	void clear();	/*���ΪTYPE_STRING��TYPE_MEMBLOCK�ͣ��ͷŴ󲿷ֿռ�*/

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
