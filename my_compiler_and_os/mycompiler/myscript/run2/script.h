#ifndef _SCRIPT_H_INCLUDED_
#define _SCRIPT_H_INCLUDED_

#include <vector>
#include <string>
#include <list>
#include <map>
#include <deque>
#include "mem.h"
#include "var.h"
using namespace std;


namespace lnb {

class CScript;
typedef int (CScript::*FUNC_PTR)(const string & funcname, deque<CVar*> &arglist, CVar & ret);
typedef int (*EXTERN_FUNC_PTR)(const string & funcname, deque<CVar*> &arglist, CVar & ret, CMem & memory);

class CScript
{
public:
	CScript();
	CScript(const CScript & another);
	const CScript& operator=(const CScript & another);

	void Clear();
	int PushInstruct(const vector<string> & instructs);
	int Run(const deque<CVar*> &args, CVar& rtnval);
	int RunOld(const deque<CVar*> &args, CVar& rtnval);
	static void  atexit();
	static int AddExternFuncPtr(const string &funcname, EXTERN_FUNC_PTR fptr);

private:
	vector<string> m_vInstructs;
	map<int, int> m_mLabels; //标签的索引和语句位置
	CMem m_mem; 	//变量内存区
	CVar m_arrVar[100]; //一个临时区域

private:
	int RunFunction(const string & funcname,  deque<CVar*> &arglist2, CVar & ret);

	static FUNC_PTR GetFuncPtr(const string &funcname);
	static map<string, EXTERN_FUNC_PTR> ms_mapExternFuncs;
	static EXTERN_FUNC_PTR GetExternFuncPtr(const string &funcname);
	static void InitFuncMap( map<string, FUNC_PTR> & mapFuncs);


	int filefunc(const string & funcname, deque<CVar*> &arglist, CVar & ret);
	static void filefunc_end();
	int InnerFuncs(const string & funcname, deque<CVar*> &arglist, CVar & vvv);
	int stringfunc(const string & funcname, deque<CVar*> &arglist, CVar & ret);
	int ipcfunc(const string & funcname, deque<CVar*> &arglist, CVar & ret);
	int datetimefunc(const string & funcname, deque<CVar*> &arglist, CVar & ret);
	int mysqlfunc(const string & funcname, deque<CVar*> &arglist, CVar & ret);
	static void mysqlfunc_end();
	int mapfunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret);
};
};

#endif
