#include "script.h"
#include "util.h"
#include "action_number.h"

#include <stack>
#include "var.h"
#include "mem.h"
#include "perlc.h"

using namespace lnb;

map<string, EXTERN_FUNC_PTR> CScript::ms_mapExternFuncs;

namespace lnb {
int RunScript(const string & scriptname, const deque<CVar*> &args, CVar & rtnval);
};

CScript::CScript()
{
}
void CScript::Clear()
{
	m_vInstructs.clear();
	m_mLabels.clear();
}
CScript::CScript(const CScript & another)
{
	m_vInstructs = another.m_vInstructs;
	m_mLabels = another.m_mLabels;
	m_mem = another.m_mem;
}
const CScript & CScript::operator=(const CScript & another)
{
	m_vInstructs = another.m_vInstructs;
	m_mLabels = another.m_mLabels;
	m_mem = another.m_mem;
}
static int action2int(const string & action)
{
	int iInt = -1;

	if (action == "!!!END") iInt = ACTION_SUB_END;
	else if (action == "!!!BEGIN") iInt = ACTION_SUB_BEGIN;
	else if (action == "LABEL") iInt = ACTION_LABEL;
	else if (action == "PUSH") iInt = ACTION_PUSH;
	else if (action == "PUSARY")  iInt = ACTION_PUSARY;
	else if (action == "PUSMAP")  iInt = ACTION_PUSMAP;
	else if (action == "SAV") iInt = ACTION_SAV;
	else if (action == "SAVARY")  iInt = ACTION_SAVARY;
	else if (action == "SAVMAP")  iInt = ACTION_SAVMAP;
	else if (action == "CLEAR") iInt = ACTION_CLEAR;
	else if (action == "CALL") iInt = ACTION_CALL;
	else if (action == "GOTO") iInt = ACTION_GOTO;
	else if (action == "GOTOFALSE") iInt = ACTION_GOTOFALSE;
	else if (action == "ADD") iInt = ACTION_ADD;
	else if (action == "SUB") iInt = ACTION_SUB;
	else if (action == "MINUS") iInt = ACTION_MINUS;
	else if (action == "MUL") iInt = ACTION_MUL;
	else if (action == "DIV") iInt = ACTION_DIV;
	else if (action == "LT") iInt = ACTION_LT;
	else if (action == "GT") iInt = ACTION_GT;
	else if (action == "GE") iInt = ACTION_GE;
	else if (action == "LE") iInt = ACTION_LE;
	else if (action == "EQ") iInt = ACTION_EQ;
	else if (action == "NE") iInt = ACTION_NE;
	else if (action == "NOT") iInt = ACTION_NOT;

	return iInt;
}
int CScript::PushInstruct(const vector<string> & instructs)
{
	int size = instructs.size();
	if (strncmp("!!!BEGIN", instructs[0].c_str(), 8) != 0 ||
		strncmp("!!!END", instructs[size-1].c_str(), 6) !=0 )
	{
		return -1;
	}
	for (int i = 0; i < size; ++i)
	{
		string instr = instructs[i];
		if (instr.length() <= 0)
		{
			return -1;
		}
	
		if (instr.length() >= 7 &&
			instr.substr(0, 5) == string("LABEL") )
		{
			// LABEL l123
			int nLabelIndex = atoi(instr.substr(7).c_str());
			int nPos = m_vInstructs.size();

			//fprintf(stdout, "l%d -> %d\n", nLabelIndex, nPos);
	
			m_mLabels.insert( std::pair<int, int>(nLabelIndex, nPos) );
		}

		//-----------------------------
		string action;
		string object;
		//分解出动作
		int pos = instr.find(' ');
		if (pos == -1)
		{
			action = instr;
			object = "";
		}
		else
		{
			action = instr.substr(0, pos);
			object = instr.substr(pos+1);
		}
		int iInt = action2int(action);
		if ( iInt < 0)
		{
			fprintf(stderr, "%s %d:action2int() failed!\n", __FILE__, __LINE__);
			return -1;
		}
		instr = string( (char*)&iInt, sizeof(iInt)) + object;
		//-----------------------------
		m_vInstructs.push_back(instr);
	
	}
	return 0;
}
int CScript::RunOld(const deque<CVar*> &args, CVar & rtnval)
{
	//将该段脚本的参数初始化到内存中
	deque<CVar*>::const_iterator args_it;
	int args_idx;
	char argname[100];
	for (args_idx = 1, args_it = args.begin(); args_it != args.end(); 
		++args_it, ++args_idx)
	{
		sprintf(argname, "$%d", args_idx);
		m_mem.SetScalarVar(argname, *(*args_it));
	}

	//开始运算
	stack<CVar> runstk; //运算栈	
	int i = 0;
	for (unsigned int index = 0; index < m_vInstructs.size(); ++index, ++i)
	{
		string & instruct = m_vInstructs[index];

		//int iAction = *(int*)instruct.data();
		//const string & object = instruct.substr(4);
		const char * object = instruct.c_str() ;
		int iAction = *(int*)object;
		object += 4;

		if (iAction == ACTION_PUSH)
		{
			char substr0 = object[0];
			CVar var;
			if (substr0 == '^') //整数常量
			{
				var.Type() = CVar::T_INT;
				//var.IntVal() = atoi(object.substr(1).c_str());
				var.IntVal() = atoll(object+1);
				runstk.push(var);
			}
			else if (substr0 == 'v' ||
				substr0 == '$') //用户变量或者临时变量
			{
				CVar * vptr = NULL;	
				m_mem.GetScalarVarPtr(object, vptr);
				var.Type() = CVar::T_PTR;
				var.PtrVal() = vptr;
				runstk.push(var);
				
			}
			else if (substr0 == '#') //字符串常量
			{
				var.Type() = CVar::T_STR;
				//string s1 = object.substr(2, object.length()-3);
				string s1(object+2, strlen(object)-3);
				StringUnescape(s1, var.StrVal());
				runstk.push(var);
			}
			else if (substr0 == '%') //浮点常量
			{
				var.Type() = CVar::T_FLOAT;
				//var.FloatVal() = atof(object.substr(1).c_str());
				var.FloatVal() = atof(object+1);
				runstk.push(var);
			}
			else 
			{
				fprintf(stderr, "指令非法! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_PUSARY) //数组元素压栈
		{
			if ( object[0] == '$') //用户变量或者临时变量
			{
				CVar var;
				CVar * varp;
				CVar offset = runstk.top();
				runstk.pop();

				m_mem.GetArrayVarPtr(object, varp, (*offset).IntVal());

				var.Type() = CVar::T_PTR;
				var.PtrVal() = varp;
				runstk.push(var);
			}
			else 
			{
				fprintf(stderr, "指令非法! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_SAV)
		{
			if (object[0] == '$' ||
				object[0] == 'v')
			{
#ifdef _DEBUG
				if (runstk.empty())
				{
					fprintf(stderr, "保存变量时运算栈为空! %s\n",
						instruct.c_str());
					return -1;
				}
#endif
				CVar  var = runstk.top();
				runstk.pop();
				m_mem.SetScalarVar(object, *var);
			}
			else
			{
				fprintf(stderr, "非法指令! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_SAVARY) /*保存数组元素*/
		{
			if (object[0] == '$' )
			{
#ifdef _DEBUG
				if (runstk.size() < 2)
				{
					fprintf(stderr, "保存数组元素时运算栈内操作数个数不够! %s\n",
						instruct.c_str());
					return -1;
				}
#endif
				CVar offset = runstk.top();
				runstk.pop();
				CVar var = runstk.top();
				runstk.pop();

				m_mem.SetArrayVar(object, *var, (*offset).IntVal());
			}
			else
			{
				fprintf(stderr, "非法指令! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_CLEAR)
		{
			while (!runstk.empty())
			{
				runstk.pop();
			}
		}
		else if (iAction == ACTION_CALL)
		{
			deque<CVar*> arglist;
#ifdef _DEBUG
			if (runstk.empty())
			{
				fprintf(stderr, "函数调用时运算栈为空! %s\n",
					instruct.c_str());
				return -1;
			}
#endif
			CVar var  = runstk.top();
			runstk.pop();
#ifdef _DEBUG
			if (var.Type() != CVar::T_INT ||
				var.IntVal() < 0 ||
				var.IntVal() > 100)
			{
				fprintf(stderr, "函数调用时参数个数不合法! %s\n",
					instruct.c_str());
				return -1;
			}
			if (runstk.size() < var.IntVal())
			{
				fprintf(stderr, "函数调用时运算栈参数个数不正确! %s\n",
						instruct.c_str());
				return -1;
			}
#endif
			for (int i = 0; i < var.IntVal(); ++i)
			{
				CVar argvar = runstk.top();
				runstk.pop();

				if (argvar.Type() == CVar::T_PTR)
				{
					arglist.push_front( (argvar.PtrVal()) );
				}
				else
				{
					m_arrVar[i] = argvar;
					arglist.push_front(&m_arrVar[i]);
				}
			}
			if (RunFunction(object, arglist, var) != 0)
			{
				fprintf(stderr, "函数执行失败!\n");
				return -1;
			}
			runstk.push(var);
		}
		else if (iAction == ACTION_GOTO)
		{
			int LabIdx = atoi(object+1);
			map<int, int>::const_iterator labit = m_mLabels.find(LabIdx);
#ifdef _DEBUG
			if (labit == m_mLabels.end())
			{
				fprintf(stderr, "非法跳转语句! %s\n", instruct.c_str());
				return -1;
			}
#endif
			index = labit->second;
		}
		else if (iAction == ACTION_GOTOFALSE)
		{
#ifdef _DEBUG
			if (runstk.empty())
			{
				fprintf(stderr, "执行跳转语句时运算栈为空! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar var = *(runstk.top());
			runstk.pop();
			long long nCondition = 0;
			if (var.Type() == CVar::T_INT)
			{
				nCondition = var.IntVal();
			}
			else if (var.Type() == CVar::T_FLOAT)
			{
				nCondition = var.FloatVal();
			}
			else 
			{
				nCondition = (var.StrVal().length() > 0 ? 1:0);
			}
			if (nCondition == 0)
			{
				//int LabIdx = atoi(object.substr(1).c_str());
				int LabIdx = atoi(object+1);
				map<int, int>::const_iterator labit = m_mLabels.find(LabIdx);
#ifdef _DEBUG
				if (labit == m_mLabels.end())
				{
					fprintf(stderr, "非法跳转语句! %s\n", instruct.c_str());
					return -1;
				}
#endif
				index = labit->second;
			}
		}
		else if (iAction == ACTION_ADD)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) + (*rop) );
		}
		else if (iAction == ACTION_SUB)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) - (*rop) );
		}
		else if (iAction == ACTION_MINUS)
		{
#ifdef _DEBUG
			if (runstk.size() < 1)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar op = runstk.top();
			runstk.pop();

			runstk.push(  -(*op) );
		}
		else if (iAction == ACTION_MUL)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( *(lop) * (*rop) );
		}
		else if (iAction == ACTION_DIV)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) / (*rop) );
		}
		else if (iAction == ACTION_LT)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) < (*rop) );
		}
		else if (iAction == ACTION_GT)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) > (*rop) );
		}
		else if (iAction == ACTION_GE)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) >= (*rop) );
		}
		else if (iAction == ACTION_LE)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( (*lop) <= (*rop) );
		}
		else if (iAction == ACTION_EQ)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( *lop == *rop );
		}
		else if (iAction == ACTION_NE)
		{
#ifdef _DEBUG
			if (runstk.size() < 2)
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar rop = runstk.top();
			runstk.pop();
			CVar lop = runstk.top();
			runstk.pop();

			runstk.push( *lop != *rop );
		}
		else if (iAction == ACTION_NOT)
		{
#ifdef _DEBUG
			if (runstk.empty())
			{
				fprintf(stderr, "操作数太少! %s\n", instruct.c_str());
				return -1;
			}
#endif
			CVar op = runstk.top();
			runstk.pop();

			runstk.push( !(*op) );
		}
		else if (iAction == ACTION_SUB_END)
		{
			break;
		}
		else if (iAction == ACTION_SUB_BEGIN)
		{
		}
		else if (iAction == ACTION_LABEL)
		{
		}
		else
		{
			fprintf(stderr, "非法指令! %s\n", instruct.c_str());
			return -1;
		}
	}
	m_mem.GetScalarVar("$OUTDATA", rtnval);
	return 0;
}
int CScript::Run(const deque<CVar*> &args, CVar & rtnval)
{
	//将该段脚本的参数初始化到内存中
	int args_idx;
	for (args_idx = 0; args_idx < args.size(); ++args_idx)
	{
		m_mem.SetArrayVar("$argv", *(args[args_idx]), args_idx+1);
	}
	CVar varArgc;
	varArgc.Type() = CVar::T_INT;
	varArgc.IntVal() = args.size();
	m_mem.SetScalarVar("$argc", varArgc);

	//运算栈
	const int  STK_MAX=20	; 		//运算栈最大深度
	CVar runstk[STK_MAX];
	register unsigned int  stkp = 0; //栈指针

	for (unsigned int index = 0; index < m_vInstructs.size(); ++index)
	{
		string & instruct = m_vInstructs[index];

		//int iAction = *(int*)instruct.data();
		//const string & object = instruct.substr(4);
		const char * object = instruct.c_str() ;
		int iAction = *(int*)object;
		object += 4;

		if (iAction == ACTION_PUSH)
		{
			char substr0 = object[0];

			CVar & var = runstk[stkp++];
			if (substr0 == '^') //整数常量
			{
				var.Type() = CVar::T_INT;
				var.IntVal() = atoll(object+1);
			}
			else if (substr0 == 'v' ||
				substr0 == '$') //用户变量或者临时变量
			{
				CVar * vptr = NULL;	
				m_mem.GetScalarVarPtr(object, vptr);
				var.Type() = CVar::T_PTR;
				var.PtrVal() = vptr;
				
			}
			else if (substr0 == '#') //字符串常量
			{
				var.Type() = CVar::T_STR;
				string s1(object+2, strlen(object)-3);
				StringUnescape(s1, var.StrVal());
			}
			else if (substr0 == '%') //浮点常量
			{
				var.Type() = CVar::T_FLOAT;
				var.FloatVal() = atof(object+1);
			}
			else 
			{
				fprintf(stderr, "指令非法! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_PUSARY) //数组元素压栈
		{

			if ( object[0] == '$') //用户变量或者临时变量
			{
				CVar * varp;
				CVar & offset = runstk[--stkp];
				

				m_mem.GetArrayVarPtr(object, varp, (*offset).IntVal());

				CVar & var = runstk[stkp++];
				var.Type() = CVar::T_PTR;
				var.PtrVal() = varp;
			}
			else 
			{
				fprintf(stderr, "指令非法! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_PUSMAP) //map元素压栈
		{

			if ( object[0] == '$') //用户变量或者临时变量
			{
				CVar * varp;
				CVar & offset = runstk[--stkp];
				

				m_mem.GetMapVarPtr(object, varp, *offset );

				CVar & var = runstk[stkp++];
				var.Type() = CVar::T_PTR;
				var.PtrVal() = varp;
			}
			else 
			{
				fprintf(stderr, "指令非法! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_SAV)
		{
			if (object[0] == '$' ||
				object[0] == 'v')
			{
				CVar  & var = runstk[--stkp];
				m_mem.SetScalarVar(object, *var);
			}
			else
			{
				fprintf(stderr, "非法指令! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_SAVARY) /*保存数组元素*/
		{
			if (object[0] == '$' )
			{
				CVar & offset = runstk[--stkp];
				CVar & var = runstk[--stkp];

				m_mem.SetArrayVar(object, *var, (*offset).IntVal());
			}
			else
			{
				fprintf(stderr, "非法指令! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_SAVMAP) /*保存map元素*/
		{
			if (object[0] == '$' )
			{
				CVar & offset = runstk[--stkp];
				CVar & var = runstk[--stkp];

				m_mem.SetMapVar(object, *var, *offset );
			}
			else
			{
				fprintf(stderr, "非法指令! %s\n", instruct.c_str());
				return -1;
			}
		}
		else if (iAction == ACTION_CLEAR)
		{
			stkp = 0;
		}
		else if (iAction == ACTION_CALL)
		{
			deque<CVar*> arglist;
			CVar & var  = runstk[--stkp];
			for (int i = 0; i < var.IntVal(); ++i)
			{
				CVar & argvar = runstk[--stkp];

				if (argvar.Type() == CVar::T_PTR)
				{
					arglist.push_front( (argvar.PtrVal()) );
				}
				else
				{
					m_arrVar[i] = argvar;
					arglist.push_front(&m_arrVar[i]);
				}
			}
			if (RunFunction(object, arglist, runstk[stkp++]) != 0)
			{
				fprintf(stderr, "函数执行失败!\n");
				return -1;
			}
		}
		else if (iAction == ACTION_GOTO)
		{
			int LabIdx = atoi(object+1);
			map<int, int>::const_iterator labit = m_mLabels.find(LabIdx);
			index = labit->second;
		}
		else if (iAction == ACTION_GOTOFALSE)
		{
			CVar  & var = *(runstk[--stkp]);
			long long nCondition = 0;
			if (var.Type() == CVar::T_INT)
			{
				nCondition = var.IntVal();
			}
			else if (var.Type() == CVar::T_FLOAT)
			{
				nCondition = var.FloatVal();
			}
			else 
			{
				nCondition = (var.StrVal().length() > 0 ? 1:0);
			}
			if (nCondition == 0)
			{
				int LabIdx = atoi(object+1);
				map<int, int>::const_iterator labit = m_mLabels.find(LabIdx);
				index = labit->second;
			}
		}
		else if (iAction == ACTION_ADD)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Add((*lop), (*rop), runstk[stkp++]);
		}
		else if (iAction == ACTION_SUB)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Sub((*lop), (*rop), runstk[stkp++]);
		}
		else if (iAction == ACTION_MINUS)
		{
			CVar & op = runstk[--stkp];

			CVar::Minus((*op), runstk[stkp++]);
		}
		else if (iAction == ACTION_MUL)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Mul((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( *(lop) * (*rop) );
		}
		else if (iAction == ACTION_DIV)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Div((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( (*lop) / (*rop) );
		}
		else if (iAction == ACTION_LT)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Lt((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] =  (*lop) < (*rop) ;
		}
		else if (iAction == ACTION_GT)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Gt((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( (*lop) > (*rop) );
		}
		else if (iAction == ACTION_GE)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Ge((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( (*lop) >= (*rop) );
		}
		else if (iAction == ACTION_LE)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Le((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( (*lop) <= (*rop) );
		}
		else if (iAction == ACTION_EQ)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Eq((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( *lop == *rop );
		}
		else if (iAction == ACTION_NE)
		{
			CVar & rop = runstk[--stkp];
			CVar & lop = runstk[--stkp];

			CVar::Ne((*lop), (*rop), runstk[stkp++]);
			//runstk[stkp++] = ( *lop != *rop );
		}
		else if (iAction == ACTION_NOT)
		{
			CVar & op = runstk[--stkp];

			CVar::Not((*op),  runstk[stkp++]);
			//runstk[stkp++] = ( !(*op) );
		}
		else if (iAction == ACTION_SUB_END)
		{
			break;
		}
		else if (iAction == ACTION_SUB_BEGIN)
		{
		}
		else if (iAction == ACTION_LABEL)
		{
		}
		else
		{
			fprintf(stderr, "非法指令! %s\n", instruct.c_str());
			return -1;
		}
	}
	m_mem.GetScalarVar("$OUTDATA", rtnval);
	return 0;
}


FUNC_PTR CScript::GetFuncPtr(const string &funcname)
{
	static map<string, FUNC_PTR> s_mapFuncs;
	if (s_mapFuncs.size() == 0)
	{
		InitFuncMap( s_mapFuncs );
	}
	map<string, FUNC_PTR>::iterator it = s_mapFuncs.find(funcname);
	if (it == s_mapFuncs.end())
	{
		return NULL;
	}
	return it->second;
}
int CScript::AddExternFuncPtr(const string &funcname, EXTERN_FUNC_PTR fptr)
{
	if (fptr == NULL || funcname.length() == 0)
	{
		return -1;
	}
	ms_mapExternFuncs.insert( std::pair<string, EXTERN_FUNC_PTR> (funcname, fptr) );
	return 0;
}
EXTERN_FUNC_PTR CScript::GetExternFuncPtr(const string &funcname)
{
	map<string, EXTERN_FUNC_PTR>::const_iterator it = ms_mapExternFuncs.find(funcname);	
	if (it == ms_mapExternFuncs.end() )
	{
		return NULL;
	}
	return it->second;
}
int CScript::RunFunction(const string & funcname, deque<CVar*> &arglist, CVar & vvv)
{
	FUNC_PTR fptr = CScript::GetFuncPtr(funcname);
	if (fptr != NULL)
	{
		if ( (this->*fptr)(funcname, arglist, vvv) != 0)
		{
			return -1;
		}
		return 0;
	}

	EXTERN_FUNC_PTR pExternFunc = CScript::GetExternFuncPtr(funcname);
	if (pExternFunc != NULL)
	{
		if ( (*pExternFunc)(funcname, arglist, vvv, this->m_mem) != 0)
		{
			return -1;
		}
		return 0;
	}

	if (lnb::RunScript(funcname, arglist, vvv) == 0)
	{
		return 0;
	}
	return -1;

}

//语言内置的函数
int CScript::InnerFuncs(const string & funcname, deque<CVar*> &arglist, CVar & vvv)
{
	if (funcname == "lnb_print") //输出函数
	{
		deque<CVar*>::const_iterator it ;
		for (it = arglist.begin(); it != arglist.end(); ++it)
		{
			printf("%s", (*it)->ToString().c_str());
		}
		return 0;
	}
	else if (funcname == "lnb_toint")  //将一个参数转化为整数型
	{
		if (arglist.empty())
		{
			fprintf(stderr, "ERROR: int() need an argument!\n");
			return -1;
		}
		deque<CVar*>::const_iterator it2 = arglist.begin();
		CVar * it = *it2;

		vvv.Type() = CVar::T_INT;
		if ( it->Type() == CVar::T_STR)
		{
			vvv.IntVal() = atoi((*it).StrVal().c_str());
		}
		else if ( it->Type() == CVar::T_INT)
		{
			vvv.IntVal() = it->IntVal();
		}
		else if ( it->Type() == CVar::T_FLOAT)
		{
			vvv.IntVal() = it->FloatVal();
		}
		return 0;
	}
	else if (funcname == "lnb_tofloat") //将参数转化为浮点型
	{
		vvv.Type() = CVar::T_FLOAT;
		if (arglist.empty())
		{
			fprintf(stderr, "ERROR: float() need an argument!\n");
			return -1;
		}
		deque<CVar*>::const_iterator it2 = arglist.begin();
		CVar * it = *it2;
		if ( it->Type() == CVar::T_STR)
		{
			vvv.FloatVal() = atof((*it).StrVal().c_str());
		}
		else if ( it->Type() == CVar::T_INT)
		{
			vvv.FloatVal() = it->IntVal();
		}
		else if ( it->Type() == CVar::T_FLOAT)
		{
			vvv.FloatVal() = it->FloatVal();
		}
		return 0;
	}
	else if (funcname == "lnb_tostr")  //将参数转化为字符串型
	{
		vvv.Type() = CVar::T_STR;
		if (arglist.empty())
		{
			fprintf(stderr, "ERROR: str() need an argument!\n");
			return -1;
		}
		deque<CVar*>::const_iterator it2 = arglist.begin();
		CVar * it = *it2;
		if ( it->Type() == CVar::T_STR)
		{
			vvv.StrVal() = it->StrVal();
		}
		else if ( it->Type() == CVar::T_INT)
		{
			char tmpbuf[100];
			sprintf(tmpbuf, "%lld", it->IntVal());
			vvv.StrVal() = tmpbuf;
		}
		else if ( it->Type() == CVar::T_FLOAT)
		{
			char tmpbuf[100];
			sprintf(tmpbuf, "%f", it->FloatVal());
			vvv.StrVal() = tmpbuf;
		}
		return 0;
	}
	else if ("lnb_ltrim" == funcname)
	{
		if (arglist.empty())
		{
			fprintf(stderr, "ERROR: ltrim() need an argument!\n");
			return -1;
		}
		deque<CVar*>::const_iterator it2 = arglist.begin();
		CVar * it = *it2;
		if (it->Type() == CVar::T_STR)
		{
			int iStart = 0;
			while (iStart < it->StrVal().length())
			{
				char c = it->StrVal().c_str()[iStart];
				if (c == ' ' || c == '\t')
				{
					++iStart;
				}
				else
				{
					break;
				}
			}
			
			if (iStart >= it->StrVal().length())
			{
				it->StrVal() = "";
			}
			else
			{
			it->StrVal() = it->StrVal().substr(iStart);
			}
		}
		else
		{
			fprintf(stderr, "ltrim() need an argument with string type.");
			return -1;
		}
		vvv = *it;
		return 0;
	}
	else if ("lnb_rtrim" == funcname)
	{
		if (arglist.size() != 1 ||
			arglist[0]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "ERROR: rtrim(string s) need an argument!\n");
			return -1;
		}
		int iLen = arglist[0]->StrVal().length();
		while (iLen > 0)
		{
			char c = arglist[0]->StrVal().c_str()[iLen-1];
			if (c == ' ' || c == '\t')
			{
				--iLen;
			}
			else
			{
				break;
			}
		}
		
		arglist[0]->StrVal() = arglist[0]->StrVal().substr(0, iLen);
		vvv = *arglist[0];
		return 0;
	}
	return -1;
}
void  CScript::atexit()
{
	perlc_end();
	filefunc_end();
	mysqlfunc_end();
}
