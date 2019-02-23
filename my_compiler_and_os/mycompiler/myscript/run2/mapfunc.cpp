#include <stdlib.h>
#include <stdio.h>
#include <deque>
#include <map>
#include <string.h>
#include "var.h"
#include "util.h"
#include <string>
#include "script.h"

using namespace std;
using namespace lnb;

#define MAX_MAP_NUM 10

class MAP
{
public:
	bool bUsed;
	map<string, string> mapContainer;

	MAP():bUsed(false)
	{ }
} ;


static MAP  g_stMap[ MAX_MAP_NUM ];



int CScript::mapfunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret)
{
	int i;
	if ("lnb_map_open" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 0)
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint map_open()\n", 
				funcname.c_str());
			return -1;
		}

		int iIndex = -1;
		for (i = 0; i < MAX_MAP_NUM; ++i)
		{
			if (!g_stMap[i].bUsed)
			{
				iIndex = i;
				break;
			}
		}
		if (-1 == iIndex)
		{
			fprintf(stderr, "执行函数%s失败， 打开的句柄太多\n", funcname.c_str());
			return 0;
		}
		ret.IntVal() = iIndex;
		g_stMap[iIndex].bUsed = true;

		return 0;
	}
	else if ("lnb_map_close" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 1 || 
			arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint map_close(int handle)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_MAP_NUM ||
			!g_stMap[arglist[0]->IntVal()].bUsed)
		{
			fprintf(stderr, "invalid map  handle!\n");
			return 0;
		}
		g_stMap[arglist[0]->IntVal()].mapContainer.clear();
		g_stMap[arglist[0]->IntVal()].bUsed = false;

		ret.IntVal() = 0;

		return 0;
	}
	else if ("lnb_map_size" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 1 || 
			arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint map_size(int handle)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_MAP_NUM ||
			!g_stMap[arglist[0]->IntVal()].bUsed)
		{
			fprintf(stderr, "invalid map  handle!\n");
			return 0;
		}
		ret.IntVal() = g_stMap[arglist[0]->IntVal()].mapContainer.size();


		return 0;
	}
	else if ("lnb_map_insert" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 3 || 
			arglist[0]->Type() != CVar::T_INT ||
			arglist[1]->Type() != CVar::T_STR ||
			arglist[2]->Type() != CVar::T_STR )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint map_insert(int handle, string key, string val)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_MAP_NUM ||
			!g_stMap[arglist[0]->IntVal()].bUsed)
		{
			fprintf(stderr, "invalid map  handle!\n");
			return 0;
		}
		g_stMap[arglist[0]->IntVal()].mapContainer.insert(
			std::pair<string, string>( arglist[1]->StrVal(), arglist[2]->StrVal()) );
		ret.IntVal() = 0;

		return 0;
	}
	else if ("lnb_map_erase" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 2 && arglist.size() != 1 ||
			arglist[0]->Type() != CVar::T_INT ||
			arglist.size() == 2 && arglist[1]->Type() != CVar::T_STR )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint map_erase(int handle[, string key])\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_MAP_NUM ||
			!g_stMap[arglist[0]->IntVal()].bUsed)
		{
			fprintf(stderr, "invalid map  handle!\n");
			return 0;
		}
		if ( arglist.size() == 2 )
		{
			g_stMap[arglist[0]->IntVal()].mapContainer.erase(  arglist[1]->StrVal());
		}
		else
		{
			g_stMap[arglist[0]->IntVal()].mapContainer.clear();
		}
		ret.IntVal() = 0;

		return 0;
	}
	else if ("lnb_map_find" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 3 ||
			arglist[0]->Type() != CVar::T_INT ||
			arglist[1]->Type() != CVar::T_STR )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint map_find(int handle, string key, string val)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_MAP_NUM ||
			!g_stMap[arglist[0]->IntVal()].bUsed)
		{
			fprintf(stderr, "invalid map  handle!\n");
			return 0;
		}
		map<string, string>::const_iterator it = g_stMap[arglist[0]->IntVal()].mapContainer.find( arglist[1]->StrVal() );
		if ( it == g_stMap[arglist[0]->IntVal()].mapContainer.end())
		{
			return 0;
		}

		arglist[2]->Type() = CVar::T_STR;
		arglist[2]->StrVal() = it->second;
		ret.IntVal() = 0;

		return 0;
	}
	return -1;
}
