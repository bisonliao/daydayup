#ifndef _MEM_H_INCLUDED_
#define _MEM_H_INCLUDED_
/*
*实现按名字存取变量
*内部基于map实现. map的每个元素是一个deque, 通常情况下,deque只有一个元素,
*就是需要存取的变量,但是如果变量是一个数组元素,那么deque就包含了一个数组
*一个数组最大可以有2048个元素
*/

#include <map>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <deque>
using namespace std;
#include "var.h"

namespace lnb {

class CMapKey
{
public:
	string m_first;
	CVar   m_second;

	CMapKey():m_first(), m_second()
	{
	}
	CMapKey(const CMapKey & a)
	{
		m_first = a.m_first;
		m_second = a.m_second;
	}
	CMapKey(const string & first, const CVar & second)
	{
		m_first = first;
		m_second = second;
	}

	CMapKey & operator=( const CMapKey & a)
	{
		if (this == &a)
		{
			return *this;
		}
		m_first = a.m_first;
		m_second = a.m_second;
		return *this;
	}

	bool operator<(const CMapKey & a) const
	{
		if (m_first == a.m_first)
		{
			return m_second.less( a.m_second );
		}
		return m_first<a.m_first;
	}
};
class CArrayIndex
{
public:
	string m_first;
	unsigned int m_second;

	CArrayIndex():m_first(), m_second()
	{
	}
	CArrayIndex(const string & first, unsigned int second)
	{
		m_first = first;
		m_second = second;
	}
	CArrayIndex(const CArrayIndex & a)
	{
		m_first = a.m_first;
		m_second = a.m_second;
	}

	CArrayIndex & operator=( const CArrayIndex & a)
	{
		if (this == &a)
		{
			return *this;
		}
		m_first = a.m_first;
		m_second = a.m_second;
		return *this;
	}

	bool operator<(const CArrayIndex & a) const
	{
		if (m_first == a.m_first)
		{
			return m_second < a.m_second;
		}
		return m_first < a.m_first;
	}
};

class CMem
{
private:
	map<CArrayIndex , CVar> m_map1; 	//保存数组变量
	map<string, CVar> m_map2; 		//保存标量
	map<CMapKey , CVar> m_map3; 		//保存map变量

public:
	void GetScalarVar(const string &varname, CVar & var);
	void SetScalarVar(const string &varname, const CVar & var);
	void GetScalarVarPtr(const string &varname, CVar* & varp);

	void GetMapVar(const string &varname, CVar & var, const CVar & key);
	void SetMapVar(const string &varname, const CVar & var, const CVar & key);
	void GetMapVarPtr(const string &varname, CVar* & varp, const CVar & key);

	void SetArrayVar(const string &varname, const CVar & var, unsigned int subscript);
	void GetArrayVar(const string &varname, CVar & var, unsigned int subscript);
	void GetArrayVarPtr(const string &varname, CVar* & varp, unsigned int subscript);
	
	void GetMapVarsByName(const string & varname, map<CMapKey , CVar> & varlist);
	void GetArrayVarsByName(const string & varname, map<CArrayIndex , CVar> & varlist);
	
};

};

#endif
