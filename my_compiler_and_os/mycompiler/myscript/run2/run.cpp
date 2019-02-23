#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <list>
#include "run.h"
using namespace std;
using namespace lnb;


static map<string, CScript> gs_AllScript;

int lnb::AddExternFunction(const string & funcname, EXTERN_FUNC_PTR fptr)
{
	return CScript::AddExternFuncPtr(funcname, fptr);
}
int lnb::ClearPCode()
{
	gs_AllScript.clear();
}

int lnb::AddPCode(const deque<string> & PCodeList )
{
	unsigned int iPCodeIndex = 0;

	while (1)
	{
		CScript sss;
		string scriptname;
		vector<string> instructs;

		if (iPCodeIndex >= PCodeList.size())
		{
			break;
		}

		while (1)
		{
			if (iPCodeIndex >= PCodeList.size())
			{
				return -1;
			}
			string sPCode = PCodeList[iPCodeIndex++];

			instructs.push_back(sPCode);

			if (strncmp("!!!BEGIN ", sPCode.c_str(), 9) == 0)
			{
				scriptname = sPCode.c_str() + 9;	
			}
			if (strncmp("!!!END", sPCode.c_str(), 6) == 0)
			{
				break;
			}
		}
		if (instructs.empty())
		{
			break;
		}
		if (sss.PushInstruct(instructs))
		{
			fprintf(stderr, "指令非法!!!\n");
			return -1;
		}
		gs_AllScript.insert(std::pair<string, CScript>(scriptname, sss));
	}
	return 0;
}


int lnb::Run(const deque<string> & PCodeList, const deque<string> & arrArgs )
{
	atexit(CScript::atexit);

	unsigned int iPCodeIndex = 0;

	while (1)
	{
		CScript sss;
		string scriptname;
		vector<string> instructs;

		if (iPCodeIndex >= PCodeList.size())
		{
			break;
		}

		while (1)
		{
			if (iPCodeIndex >= PCodeList.size())
			{
				return -1;
			}
			string sPCode = PCodeList[iPCodeIndex++];

			instructs.push_back(sPCode);

			if (strncmp("!!!BEGIN ", sPCode.c_str(), 9) == 0)
			{
				scriptname = sPCode.c_str() + 9;	
			}
			if (strncmp("!!!END", sPCode.c_str(), 6) == 0)
			{
				break;
			}
		}
		if (instructs.empty())
		{
			break;
		}
		if (sss.PushInstruct(instructs))
		{
			fprintf(stderr, "指令非法!!!\n");
			return -1;
		}
		gs_AllScript.insert(std::pair<string, CScript>(scriptname, sss));
	}

	deque<CVar*> args;
	const char cVarMaxNum = 10;
	CVar var[cVarMaxNum];
	if (arrArgs.size() > cVarMaxNum)
	{
		fprintf(stderr, "%s %d: 参数太多!\n", __FILE__, __LINE__);
		return -1;
	}
	for (int i = 0; i < arrArgs.size(); ++i)
	{
		var[i].Type() = CVar::T_STR;
		var[i].StrVal() = arrArgs[i];
		args.push_back( &(var[i]) );
	}
	CVar rtnval;
	return lnb::RunScript("main", args, rtnval);
}

int lnb::RunScript(const string & scriptname, const deque<CVar*> &args, CVar & rtnval)
{
	map<string,CScript>::iterator script_it = gs_AllScript.find(scriptname);
	if (script_it == gs_AllScript.end())
	{
		fprintf(stderr, "Error! 没有找到函数%s!\n", scriptname.c_str());
		return -1;
	}
#ifndef  _RECURSIVE_

	//不支持第归调用流程
	if (script_it->second.Run(args, rtnval) < 0)
#else
	//支持第归调用流程
	CScript sss = script_it->second;
	if (sss.Run(args, rtnval) < 0)
#endif
	{
		return -1;
	}
	return 0;
}
