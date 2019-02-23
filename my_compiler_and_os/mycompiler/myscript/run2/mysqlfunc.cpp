#include <stdlib.h>
#include <stdio.h>
#include <deque>
#include <string.h>
#include "var.h"
#include "util.h"
#include <string>
#include "script.h"
#include "mysql.h"

using namespace std;
using namespace lnb;

#define MAX_CONN_NUM 10

typedef struct 
{
	MYSQL handle;

	MYSQL_RES *  pres;		//结果集的指针
	int iFieldNum;			//结果集的字段个数
	string sTitle;			//结果集字段的名称

	int iAffectedRowNum;	//update/delete等影响的纪录条数
} DbLink;


static DbLink  g_stMysql[ MAX_CONN_NUM ];
static bool   g_bUsedMap[ MAX_CONN_NUM ] = {false};

static int ExecuteSql(DbLink * pLink, const char * sqlstr)
{
	MYSQL * phandle = &(pLink->handle);

	if (mysql_query(phandle, sqlstr))
	{
		fprintf(stderr, "sql执行失败![%d][%s]\n", mysql_errno(phandle), mysql_error(phandle) );
		return -1;
	}
	if (mysql_field_count(phandle) > 0)
	{
		if (pLink->pres != NULL)
		{
			mysql_free_result(pLink->pres);
			pLink->pres = NULL;
		}

		pLink->pres = mysql_use_result(phandle);

		//保存字段个数和字段名称

		pLink->iFieldNum = mysql_num_fields(pLink->pres);
		MYSQL_FIELD *fields = mysql_fetch_fields(pLink->pres);

		char szTmpFieldName[255];
		unsigned int i;
		pLink->sTitle = "";
		for(i = 0; i < pLink->iFieldNum; i++)
		{
			snprintf(szTmpFieldName, sizeof(szTmpFieldName), "%s|", fields[i].name);
			pLink->sTitle.append(szTmpFieldName);
		}
	}
	else
	{
		pLink->iAffectedRowNum  = mysql_affected_rows(phandle);
	}
	return 0;
}

void CScript::mysqlfunc_end()
{
	for (int i = 0; i < MAX_CONN_NUM; ++i)
	{
		if (g_bUsedMap[i])
		{
			if (g_stMysql[i].pres != NULL)
			{
				mysql_free_result(g_stMysql[i].pres);
			}
			mysql_close( &(g_stMysql[i].handle) );
		}
	}
	mysql_library_end();
}

int CScript::mysqlfunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret)
{
	static bool bFirstTime = true;
	int i;
	if (bFirstTime)
	{
		bFirstTime = false;

		if (mysql_library_init(0, NULL, NULL))
		{
			fprintf(stderr, "mysql_library_init() failed!\n");
			return -1;
		}
		for (i = 0; i < MAX_CONN_NUM; ++i)
		{
			if (NULL == mysql_init( &(g_stMysql[i].handle) ) )
			{
				fprintf(stderr, "mysql_init() failed!\n");
				return -1;
			}
			g_stMysql[i].pres = NULL;
			g_stMysql[i].sTitle.reserve(1024);
			g_stMysql[i].iFieldNum = 0;
			g_stMysql[i].iAffectedRowNum = 0;
		}
	}
	if ("lnb_mysql_connect" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 5 || 
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_INT || 
			arglist[2]->Type() != CVar::T_STR || 
			arglist[3]->Type() != CVar::T_STR || 
			arglist[4]->Type() != CVar::T_STR  )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint mysql_connect(string hostname, int port, string username, string password, string dbname)\n", 
				funcname.c_str());
			return -1;
		}

		int iIndex = -1;
		for (i = 0; i < MAX_CONN_NUM; ++i)
		{
			if (!g_bUsedMap[i])
			{
				iIndex = i;
				break;
			}
		}
		if (-1 == iIndex)
		{
			fprintf(stderr, "执行函数%s失败， 建立的数据库连接太多\n", funcname.c_str());
			return 0;
		}
		if (NULL == mysql_real_connect( &(g_stMysql[iIndex].handle), arglist[0]->StrVal().c_str(),
                       arglist[2]->StrVal().c_str(),
                       arglist[3]->StrVal().c_str(),
                       arglist[4]->StrVal().c_str(),
                       arglist[1]->IntVal(),
                       NULL,
                       0) )
		{
			fprintf(stderr, "连接数据库失败,%s\n", mysql_error( &(g_stMysql[iIndex].handle) ));
			return 0;
		}
		ret.IntVal() = iIndex;
		g_bUsedMap[iIndex] = true;

		return 0;
	}
	else if ("lnb_mysql_close" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 1 || 
			arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint mysql_close(int handle)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_CONN_NUM ||
			!g_bUsedMap[arglist[0]->IntVal()])
		{
			return 0;
		}
		mysql_close( &(g_stMysql[ arglist[0]->IntVal() ].handle) );
		g_bUsedMap[arglist[0]->IntVal()] = false;

		ret.IntVal() = 0;

		return 0;
	}
	else if ("lnb_mysql_query" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 2 || 
			arglist[0]->Type() != CVar::T_INT ||
			arglist[1]->Type() != CVar::T_STR )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint mysql_query(int handle, string sqlstr)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_CONN_NUM ||
			!g_bUsedMap[arglist[0]->IntVal()])
		{
			fprintf(stderr, "invalid mysql connection handle!\n");
			return 0;
		}
		if (ExecuteSql( &g_stMysql[ arglist[0]->IntVal() ], arglist[1]->StrVal().c_str() ) == 0)
		{
			ret.IntVal() = 0;
		}

		return 0;
	}
	else if ("lnb_mysql_fetchrow" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() < 1 || 
			arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint mysql_fetchrow(int handle, ...)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_CONN_NUM ||
			!g_bUsedMap[arglist[0]->IntVal()])
		{
			return 0;
		}
		MYSQL_ROW row = mysql_fetch_row(g_stMysql[ arglist[0]->IntVal() ].pres);
		if (NULL == row)
		{
			ret.IntVal() = 0;
			return 0;
		}
		int i;
		for (i = 1; i < arglist.size(); ++i)
		{
			arglist[i]->Type() = CVar::T_STR;
			if (row[i-1] == NULL)
			{
				arglist[i]->StrVal() = "<NULL>";
			}
			else
			{
				arglist[i]->StrVal() =  row[i-1];
			}
		}
		ret.IntVal() = 1;

		return 0;
	}
	else if ("lnb_mysql_getAffectedRowNum" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 1 ||
			arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint mysql_getAffectedRowNum(int handle)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_CONN_NUM ||
			!g_bUsedMap[arglist[0]->IntVal()])
		{
			return 0;
		}
		ret.IntVal() = g_stMysql[ arglist[0]->IntVal() ].iAffectedRowNum;

		return 0;
	}
	else if ("lnb_mysql_getFieldTitle" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 2 ||
			arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\nint mysql_getFieldTitle(int handle, string title)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_CONN_NUM ||
			!g_bUsedMap[arglist[0]->IntVal()])
		{
			return 0;
		}
		arglist[1]->Type() = CVar::T_STR;
		arglist[1]->StrVal() = g_stMysql[ arglist[0]->IntVal() ].sTitle;

		ret.IntVal() = 0;
		return 0;
	}
	return -1;
}
