#include "mem.h"
using namespace lnb;

static CMem g_GlobalVars; //全局变量保存区


void CMem::SetArrayVar(const string &varname, const CVar&var, unsigned int subscript)
{
	//--------------------------------------
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.SetArrayVar(varname.substr(1), var, subscript);
		return;
	}
	//--------------------------------------

	CArrayIndex key(varname, subscript );
	map<CArrayIndex, CVar >::iterator it = m_map1.find( key );
	if (it == m_map1.end())
	{
		m_map1.insert(std::pair<CArrayIndex, CVar >(key, var));
	}
	else
	{
		it->second = var;
	}
}
void CMem::SetMapVar(const string &varname, const CVar&var, const CVar & key)
{
	//--------------------------------------
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.SetMapVar(varname.substr(1), var, key);
		return;
	}
	//--------------------------------------

	CMapKey kk(varname, key);
	map<CMapKey, CVar >::iterator it = m_map3.find(kk);
	if (it == m_map3.end())
	{
		m_map3.insert(std::pair<CMapKey, CVar> (kk, var));
	}
	else
	{
		it->second = var;
	}
}
void CMem::SetScalarVar(const string & varname, const CVar&var)
{
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.SetScalarVar(varname.substr(1), var);
		return;
	}
	//--------------------------------------

	map<string,CVar >::iterator it = m_map2.find(varname);
	if (it == m_map2.end())
	{
		/*插入新的*/
		m_map2.insert(std::pair<string, CVar >(varname, var));
	}
	else
	{
		it->second = var;
	}
}
void CMem::GetArrayVar(const string &varname, CVar&var, unsigned int subscript)
{
	//--------------------------------------
	 if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetArrayVar(varname.substr(1), var, subscript);
		return;
	}
	//--------------------------------------

	CArrayIndex index(varname, subscript);
	map<CArrayIndex, CVar >::iterator it = m_map1.find(index);
	if (it == m_map1.end())
	{
		/*插入新的*/
		CVar newvar;
		m_map1.insert(std::pair<CArrayIndex, CVar >(index, newvar ));
		var = newvar;
	}
	else
	{
		var = it->second;
	}

}
void CMem::GetMapVar(const string &varname, CVar&var, const CVar & key)
{
	//--------------------------------------
	 if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetMapVar(varname.substr(1), var, key);
		return;
	}
	//--------------------------------------

	CMapKey kkk(varname, key);
	map<CMapKey, CVar>::iterator it = m_map3.find(kkk);
	if (it == m_map3.end())
	{
		CVar newvar;
		m_map3.insert(std::pair<CMapKey, CVar >(kkk, newvar));
		var = newvar;
	}
	else
	{
		var = it->second;
	}

}
void CMem::GetScalarVar(const string &varname, CVar&var)
{
	 if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetScalarVar(varname.substr(1), var);
		return;
	}
	//--------------------------------------

	map<string,CVar >::iterator it = m_map2.find(varname);
	if (it == m_map2.end())
	{
		CVar newvar;
		/*插入新的*/
		m_map2.insert(std::pair<string, CVar >(varname, newvar));
		var = newvar;
	}
	else
	{
		var = (it->second);
	}

}
void CMem::GetArrayVarPtr(const string &varname, CVar* &varp, unsigned int subscript)
{
	//--------------------------------------
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetArrayVarPtr(varname.substr(1), varp, subscript);
		return;
	}
	//--------------------------------------

	CArrayIndex index(varname, subscript);
	map<CArrayIndex,CVar >::iterator it = m_map1.find(index);
	if (it == m_map1.end())
	{
		CVar newvar;
		/*插入新的*/
		it = m_map1.insert( std::pair<CArrayIndex, CVar >(index, newvar) ).first;
	}
	varp = &(it->second);
}
void CMem::GetMapVarPtr(const string &varname, CVar* &varp, const CVar & key)
{
	//--------------------------------------
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetMapVarPtr(varname.substr(1), varp, key);
		return;
	}
	//--------------------------------------

	CMapKey kkk(varname, key);
	map<CMapKey, CVar>::iterator it = m_map3.find(kkk);
	if (it == m_map3.end())
	{
		CVar newvar;
		/*插入新的*/
		it = m_map3.insert(std::pair<CMapKey, CVar >(kkk, newvar)).first;
	}
	varp = & ( (it->second) );
}
void CMem::GetScalarVarPtr(const string &varname, CVar* &varp)
{
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetScalarVarPtr(varname.substr(1), varp);
		return;
	}
	//--------------------------------------

	map<string,CVar>::iterator it = m_map2.find(varname);
	if (it == m_map2.end())
	{
		CVar newvar;
		/*插入新的*/
		it = m_map2.insert(std::pair<string, CVar >(varname, newvar)).first;
	}
	varp = & ( (it->second));
}

void CMem::GetArrayVarsByName(const string & varname, map<CArrayIndex , CVar> & varlist)
{
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetArrayVarsByName(varname.substr(1), varlist);
		return;
	}
	//--------------------------------------
	CArrayIndex index1, index2;
	index1.m_first = varname;
	index1.m_second = 0;
	index2.m_first = varname;
	index2.m_second = UINT_MAX;

	map<CArrayIndex, CVar>::const_iterator first = m_map1.lower_bound(index1);
	map<CArrayIndex, CVar>::const_iterator last = m_map1.upper_bound(index2);

	varlist.clear();
	if (first == m_map1.end())
	{
		return;
	}
	varlist.insert(first, last);
}
void CMem::GetMapVarsByName(const string & varname, map<CMapKey , CVar> & varlist)
{
	if (varname.data()[0] == '$' && varname.data()[1] == '$') //全局变量
	{
		g_GlobalVars.GetMapVarsByName(varname.substr(1), varlist);
		return;
	}
	//--------------------------------------
	CMapKey index1, index2;
	CVar vv;
	index1.m_first = varname;
	CVar::getTheLeast(vv);
	index1.m_second = vv;

	index2.m_first = varname;
	CVar::getTheMost(vv);
	index2.m_second = vv;

	map<CMapKey, CVar>::const_iterator first = m_map3.lower_bound(index1);
	map<CMapKey, CVar>::const_iterator last = m_map3.upper_bound(index2);

	varlist.clear();
	if (first == m_map3.end())
	{
		return;
	}
	varlist.insert(first, last);
}
