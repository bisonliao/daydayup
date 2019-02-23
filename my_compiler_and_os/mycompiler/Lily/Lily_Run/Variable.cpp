#include "Variable.h"

int Variable::setType(int type)
{
	if (type == TYPE_INTEGER || type == TYPE_STRING || type == TYPE_FLOAT || type == TYPE_MEMBLOCK)
	{
		m_nType = type;
		return 0;
	}
	else
	{
		return -1;
	}
}
Variable::Variable()
{
	m_nType = TYPE_INTEGER;
	m_IntegerValue = 0;
}
Variable::~Variable()
{
}
Variable::Variable(const Variable &v)
{
	m_nType = v.m_nType;
	m_asName = v.m_asName;
	m_StringValue = v.m_StringValue;
	m_IntegerValue = v.m_IntegerValue;
	m_FloatValue = v.m_FloatValue;
	m_MemBlockValue = v.m_MemBlockValue;
}
const Variable & Variable::operator=(const Variable &v)
{
	m_nType = v.m_nType;
	m_asName = v.m_asName;
	m_StringValue = v.m_StringValue;
	m_IntegerValue = v.m_IntegerValue;
	m_FloatValue = v.m_FloatValue;
	m_MemBlockValue = v.m_MemBlockValue;

	return *this;
}
void Variable::clear()
{
	if (m_nType == TYPE_STRING)
	{
		m_StringValue.clear();
	}
	if (m_nType == TYPE_MEMBLOCK)
	{
		m_MemBlockValue.Realloc(0);
	}
}
