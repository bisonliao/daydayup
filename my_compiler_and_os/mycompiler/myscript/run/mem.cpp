#include "mem.h"

/*
* 保证qqq有n个元素
*/
static void MakeSureSize(deque<CVar> & qqq, size_t n)
{
	for (int i = qqq.size(); i < n; ++i)
	{
		qqq.push_back( CVar() );
	}
}

void CMem::SetVar(const string &varname, CVar&var, unsigned short subscript)
{
	if ( subscript >= CMem::ARRAY_MAX_SIZE)
	{
		fprintf(stderr, "数组越界!\n");
		exit(-1);
	}

	map<string,deque<CVar> >::iterator it = m_map.find(varname);
	if (it == m_map.end())
	{
		deque<CVar> qqq;
		MakeSureSize(qqq, subscript+1);
		qqq[subscript] = var;
		/*插入新的*/
		m_map.insert(std::pair<string, deque<CVar> >(varname, qqq));
	}
	else
	{
		MakeSureSize( (it->second), subscript+1);
		(it->second)[subscript] = var;
	}
}
void CMem::GetVar(const string &varname, CVar&var, unsigned short subscript)
{
	if ( subscript >= CMem::ARRAY_MAX_SIZE)
	{
		fprintf(stderr, "数组越界!\n");
		exit(-1);
	}

	map<string,deque<CVar> >::iterator it = m_map.find(varname);
	if (it == m_map.end())
	{
		CVar newvar;
		deque<CVar> qqq;
		MakeSureSize(qqq, subscript+1);
		qqq[subscript] = newvar;
		/*插入新的*/
		m_map.insert(std::pair<string, deque<CVar> >(varname, qqq));
		var = newvar;
	}
	else
	{
		MakeSureSize( (it->second), subscript+1);
		var = (it->second)[subscript];
	}

}
