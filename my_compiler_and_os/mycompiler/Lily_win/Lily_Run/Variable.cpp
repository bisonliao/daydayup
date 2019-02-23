#include "Variable.h"

#undef SHOW_STEP
#if defined(_DEBUG)
	#undef SHOW_STEP
	#define SHOW_STEP printf("执行%s的第%d行...\n", __FILE__, __LINE__);
	//#define SHOW_STEP ;
#else
	#define SHOW_STEP ;
#endif

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
	if (m_nType == TYPE_MEMBLOCK)
	{
		m_MemBlockValue = v.m_MemBlockValue;
	}
	
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
AnsiString Variable::toString() const
{
	AnsiString ret("variable");
	
	char tmp[255];
	_snprintf(tmp, sizeof(tmp), " name=[%s]", m_asName.c_str());
	ret.concat(tmp);
	
	if (m_nType == TYPE_STRING)
	{
		_snprintf(tmp, sizeof(tmp), " type=[string] value=[");
		ret.concat(tmp);
		ret.concat(m_StringValue);
		ret.concat("]\n");
	}
	else if (m_nType == TYPE_INTEGER)
	{
		_snprintf(tmp, sizeof(tmp), " type=[int] value=[");
		ret.concat(tmp);
		_snprintf(tmp, sizeof(tmp), "%d]", m_IntegerValue);
		ret.concat(tmp);
		ret.concat("]\n");	
	}
	else if (m_nType == TYPE_FLOAT)
	{
		_snprintf(tmp, sizeof(tmp), " type=[float] value=[");
		ret.concat(tmp);
		_snprintf(tmp, sizeof(tmp), "%f]", m_FloatValue);
		ret.concat(tmp);
		ret.concat("]\n");	
	}
	else if (m_nType == TYPE_MEMBLOCK)
	{
		_snprintf(tmp, sizeof(tmp), " type=[memblock] value=[");
		ret.concat(tmp);
		ret.concat(m_MemBlockValue.toString());
		ret.concat("]\n");
	}
	else 
	{
		_snprintf(tmp, sizeof(tmp), " unknow type[%d] and value!\n", m_nType);
		ret.concat(tmp);
	}
	return ret;
}
