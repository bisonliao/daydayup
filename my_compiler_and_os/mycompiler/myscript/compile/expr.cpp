#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "token.h"
#include <vector>
#include <stack>
#include <list>
#include "common.h"
using namespace std;


static int RunOperator(stack<CToken> & oprnt_stk, CToken & topword, 
	list<string> & codelist);
static int FunctionAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist);

/*
*分析一个表达式，成功返回表达式的长度，失败返回－1
*/
int ExpressionAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist)
{

	codelist.clear();

	char code[2048];

	int nbrk = 0; //左右括号的计数

	if (start < 0 || start >= words.size())
	{
		return -1;
	}
	//如果一开始就不是操作符或者操作数，那么表达式计算失败
	if (  !(words[start].IsExprOperator() || words[start].IsExprOperant() || 
			words[start].m_token == ID || words[start].m_token == LBRK )  )
	{
		fprintf(stderr, "%s %d:表达式不合法, lineno=%d\n",
						__FILE__,
						__LINE__,
					words[start].m_yylval.lineno);
		return -1;	
	}
	int i ;
	bool hasNoOperator = true; //整个表达式不含有操作符号
	int ExprLen; //分析成功后，使用的单词的个数，也就是表达式的长度

	stack<CToken> opt_stk, oprnt_stk; //操作数栈和操作符栈
	for ( i = start; i < words.size() && 
					(words[i].IsExprOperator() || words[i].IsExprOperant() ||
					 words[i].m_token == LBRK || words[i].m_token == RBRK ||
					 words[i].m_token == ID);
					 ++i)
	{
		CToken word = words[i];
#if 0
		if (word.IsExprOperant())
		{
			oprnt_stk.push(word);
		}
#else
		if (word.m_token == CONST_FLOAT || 
			word.m_token == CONST_INT ||
			word.m_token == CONST_STRING) /*直接入栈*/
		{
			oprnt_stk.push(word);
		}
		else if (word.m_token == VAR)
		{
			if ( (i+1) < words.size() &&
				words[i+1].m_token == LOFFSET) /*一个数组元素开始了*/
			{
				/*数组元素的下标是一个表达式*/
				list<string> arraycodelist;
				int nSubscriptLen =  ArrayElementAnalyze(words, i, arraycodelist);
				if (nSubscriptLen < 4)
				{
					return -1;
				}
				codelist.insert(codelist.end(), 
					arraycodelist.begin(), arraycodelist.end()); 
				//产生一个中间变量，压栈
				CToken tmptoken = GenTmpVar();
				tmptoken.m_yylval.lineno = word.m_yylval.lineno;
				oprnt_stk.push(tmptoken);
				//保存为中间变量
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);

				i += (nSubscriptLen - 1);
			}
			else /*就是一普通变量*/
			{
				oprnt_stk.push(word); /*直接入栈*/
			}
		}
#endif
		else if (word.m_token == ID) //函数
		{
			if (words[i+1].m_token != LBRK)
			{
				fprintf(stderr, "%s %d:表达式不合法, %s后面希望出现 ( . lineno=%d\n",
					__FILE__,
					__LINE__,
					word.ToString().c_str(),
					word.m_yylval.lineno);
				return -1;
			}
			else 
			{
				//函数的分析
				list<string> codes;
				int nFunLen = FunctionAnalyze(words, i, codes);
				if (nFunLen < 3)
				{
					return -1;
				}
				codelist.insert(codelist.end(), codes.begin(), codes.end());
				//产生一个中间变量，压栈
				CToken tmptoken = GenTmpVar();
				tmptoken.m_yylval.lineno = word.m_yylval.lineno;
				oprnt_stk.push(tmptoken);
				//保存为中间变量
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);

				i += (nFunLen - 1);
			}
		}
		else if (word.m_token == LBRK)
		{
			++nbrk;
			//直接入栈
			opt_stk.push(word);
		}
		else if (word.m_token == RBRK)
		{
			if (nbrk == 0) //表达式走到了尽头
			{
				break;
			}
			--nbrk;

			while (1)
			{
				if (oprnt_stk.empty() || opt_stk.empty())
				{
					fprintf(stderr, "%s %d:表达式中左右括号不匹配! lineno=%d\n",
						__FILE__,
						__LINE__,
						words[start].m_yylval.lineno);
					return -1;
				}
				CToken topword = opt_stk.top();
				opt_stk.pop();
				if (topword.m_token == LBRK)
				{
					CToken ttt = oprnt_stk.top();
					sprintf(code, "PUSH %s\n", ttt.ToString().c_str());
					codelist.push_back(code);
					break;
				}
				list<string> codes;
				if (RunOperator(oprnt_stk, topword, codes))
				{
					return -1;
				}
				codelist.insert(codelist.end(), codes.begin(), codes.end());

				//产生一个中间变量，压栈
				CToken tmptoken = GenTmpVar();
				tmptoken.m_yylval.lineno = topword.m_yylval.lineno;
				oprnt_stk.push(tmptoken);
				//保存为中间变量
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);
				
			}
		}
		else //如果是操作符
		{
			hasNoOperator = false;
			if (opt_stk.empty() ||
				word.GetPriorityOUT() > opt_stk.top().GetPriorityIN())
			{
				//如果符号栈为空，或者但前符号优先级高于栈定符号，那么压栈
				opt_stk.push(word);
			}
			else
			{
				while ( !(opt_stk.empty()) &&
					word.GetPriorityOUT() <= opt_stk.top().GetPriorityIN())
				{
					//弹出一个操作符号
					CToken topword = opt_stk.top();
					opt_stk.pop();
					if (topword.GetToken() == LBRK)
					{
						fprintf(stderr, "%s %d:表达式中左右括号不匹配! lineno=%d\n",
							__FILE__,
							__LINE__,
							words[start].m_yylval.lineno);
						return -1;
					}
					list<string> codes;
					if (RunOperator(oprnt_stk, topword, codes))
					{
						return -1;
					}
					codelist.insert(codelist.end(), codes.begin(), codes.end());

					//产生一个中间变量，压栈
					CToken tmptoken = GenTmpVar();
					tmptoken.m_yylval.lineno = topword.m_yylval.lineno;
					oprnt_stk.push(tmptoken);
					//保存为中间变量
					sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
					codelist.push_back(code);
				}
				if (word.GetToken() != RBRK)
				{
					opt_stk.push(word);
				}
			}
		}
	}
	ExprLen = i - start;
	if (hasNoOperator) //如果整个表达式没有含有操作符
	{
		if (oprnt_stk.size() != 1) //如果操作数个数大于1，错误
		{
			fprintf(stderr, "%s %d:表达式不合法! lineno=%d\n", 
				__FILE__,
				__LINE__,
				oprnt_stk.top().m_yylval.lineno);
			return -1;
		}
		else
		{
			//整个表达式就只有一个操作数，保存到运算栈里
			CToken onlyOprnt = oprnt_stk.top();
			sprintf(code, "PUSH %s\n", onlyOprnt.ToString().c_str());
			codelist.push_back(code);
		}
	}
	
	while (!opt_stk.empty())
	{
			//弹出一个操作符号
			CToken topword = opt_stk.top();
			opt_stk.pop();
			if (topword.GetToken() == LBRK || topword.GetToken() == RBRK)
			{
				fprintf(stderr, "%s %d:左右括符不匹配! lineno=%d\n", 
					__FILE__,
					__LINE__,
					topword.m_yylval.lineno);
				return -1;
			}
			list<string> codes;
			if (RunOperator(oprnt_stk, topword, codes))
			{
				return -1;
			}
			codelist.insert(codelist.end(), codes.begin(), codes.end());
			//产生一个中间变量，压栈
			CToken tmptoken = GenTmpVar();
			tmptoken.m_yylval.lineno = topword.m_yylval.lineno;
			oprnt_stk.push(tmptoken);
			if (opt_stk.size() != 0) //最后的结果不保存为中间变量，就放到运算栈里
			{
				//保存为中间变量
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);
			}
	}
	if (oprnt_stk.size() != 1)
	{
		fprintf(stderr, "%s %d:表达式非法! lineno=%d\n", 
			__FILE__,
			__LINE__,
					words[start].m_yylval.lineno);
		return -1;
		
	}
	return ExprLen;
}
/*
*操作符topword对操作数栈里的元素进行一次运算，把结果保存到操作数栈
*成功返回0， 失败返回－1
*/
static int RunOperator(stack<CToken> & oprnt_stk, CToken & topword, 
	list<string> & codelist)
{
	//弹出操作数
	codelist.clear();

	char code[2048];

	int OprntNum = topword.GetOprntNumNeed();
	if (oprnt_stk.size() < OprntNum)
	{
		fprintf(stderr, "%s %d:操作数个数不够! lineno=%d\n", 
					__FILE__,
					__LINE__,
					topword.m_yylval.lineno);
		return -1;
	}
	if (OprntNum == 1)
	{
		CToken oprnt;
		oprnt =  oprnt_stk.top();
		oprnt_stk.pop();
		//运算
		sprintf(code, "PUSH %s\n", oprnt.ToString().c_str());
		codelist.push_back(code);
		sprintf(code, "%s\n", topword.ToString().c_str());
		codelist.push_back(code);

	}
	else if (OprntNum == 2)
	{
		CToken loprnt, roprnt;
		roprnt =  oprnt_stk.top();
		oprnt_stk.pop();
		loprnt =  oprnt_stk.top();
		oprnt_stk.pop();
		//运算
		sprintf(code, "PUSH %s\n", loprnt.ToString().c_str());
		codelist.push_back(code);
		sprintf(code, "PUSH %s\n", roprnt.ToString().c_str());
		codelist.push_back(code);
		sprintf(code, "%s\n", topword.ToString().c_str());
		codelist.push_back(code);
	}
	else
	{
		fprintf(stderr, "%s %d:%s不是一个有效的操作符! lineno=%d\n",
			__FILE__,
			__LINE__,
				topword.ToString().c_str(),
				topword.m_yylval.lineno);
		return -1;
	}

	return 0;
}

CToken GenTmpVar(bool CleanAll /*= false*/)
{
	static int i = 0;

	if (CleanAll)
	{
		i = 0;
	}
	char buf[100];
	sprintf(buf, "v%d", i++);
	YYLVAL yylval;
	yylval.id_val = buf;
	CToken ttt(VAR, yylval);
	return ttt;
}
/*
*分析一个函数，成功返回表达式的长度，失败返回－1
*/
static int FunctionAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist)
{
	codelist.clear();
	char code[2048];

	if (start+1 >= words.size())
	{
		return -1;
	}
	if ( !(words[start].m_token == ID && words[start+1].m_token == LBRK) )
	{
		return -1;
	}
	int i = 0; //参数的长度和
	int nArgs = 0; //参数的个数
	list<CToken> args;
	if (words[start+2].m_token != RBRK) //含有参数
	{
		for (; ; )
		{
			if ( (start+2+i)>=words.size())
			{
				fprintf(stderr, "函数语法错误! lineno=%d\n",
					words[start].m_yylval.lineno);
				return -1;
			}
			list<string> codes;
			int nExprLen = ExpressionAnalyze(words, start+2+i, codes);
			if (nExprLen <= 0)
			{
				fprintf(stderr, "函数参数不合法! lineno=%d\n",
					words[start].m_yylval.lineno);
				return -1;
			}
			i += nExprLen;
			++ nArgs;
			codelist.insert(codelist.end(), codes.begin(), codes.end());

			CToken arg = GenTmpVar();
			args.push_back(arg);
			sprintf(code, "SAV %s\n", arg.ToString().c_str());
			codelist.push_back(code);

			if (words[start+2+i].m_token == RBRK) //参数列表结束
			{
				break;
			}
			else if (words[start+2+i].m_token != COMMA)
			{
				fprintf(stderr, "函数参数不合法! 需要用逗号分割参数! lineno=%d\n",
					words[start].m_yylval.lineno);
				return -1;
			}
			++i;
		}
	}
	list<CToken>::const_iterator it;
	for (it = args.begin(); it != args.end(); ++it)
	{
		sprintf(code, "PUSH %s\n", (*it).ToString().c_str());
		codelist.push_back(code);
	}
	sprintf(code, "PUSH ^%d\n", nArgs);
	codelist.push_back(code);
	sprintf(code, "CALL %s\n", words[start].ToString().c_str());
	codelist.push_back(code);

	return 2 + i + 1;
}
/*
*分析一个数组元素，成功返回数组元素的长度，失败返回－1
* isLeft 是说数组元素是作为左值否
*/
int ArrayElementAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist, bool isLeft)
{
	codelist.clear();
	char code[2048];

	if (start+3 >= words.size())
	{
		return -1;
	}
	if ( !(words[start].m_token == VAR && words[start+1].m_token == LOFFSET) )
	{
		return -1;
	}

	/*分析下标*/
	list<string> codes;
	int nExprLen = ExpressionAnalyze(words, start+2, codes);
	if (nExprLen <= 0)
	{
		fprintf(stderr, "数组元素下标不合法! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	codelist.insert(codelist.end(), codes.begin(), codes.end());

	if ( (start+1+nExprLen+1) >= words.size()
	   || words[start+1+nExprLen+1].m_token != ROFFSET) /*不等于右中括号*/
	{
		fprintf(stderr, "数组元素下标不合法, 需要']'! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}

	/*
	CToken arg = GenTmpVar();
	sprintf(code, "SAV %s\n", arg.ToString().c_str());
	codelist.push_back(code);

	sprintf(code, "PUSH %s\n", arg.ToString().c_str());
	codelist.push_back(code);
	*/

	memset(code, 0, sizeof(code));
	if (!isLeft)
	{
		snprintf(code, sizeof(code)-1, "PUSARY %s\n", words[start].ToString().c_str());
	}
	codelist.push_back(code);

	//   $v    [   subscript ] 
	return 1 + 1 + nExprLen + 1;
}
