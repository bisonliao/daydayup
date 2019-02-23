#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "token.h"
#include <vector>
#include <stack>
#include <list>
#include <string>
#include "common.h"
using namespace std;

static int AssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);
static int ParseStatement(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist);
static int IfAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> &codelist);
static int WhileAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist);
static int BreakAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist);
static int ContinueAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist);
static int ReturnAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist);
static int RunExprAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);
static int ArrayAssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);
static int ArrayStartAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);

typedef struct {
	int nBegin;
	int nEnd;
} LabelPair;

static stack<LabelPair> gs_lpstk; //记录循环语句跳转标签的栈，continue/break用到
static int gs_nModelEndLabel; //每个模块结束位置的标签

/*
*语法分析，成功返回0， 失败返回－1
*/
int parse(const vector<CToken> & words)
{
	while ( ! gs_lpstk.empty())
	{
		gs_lpstk.pop();
	}
	gs_nModelEndLabel = GetLabel();

	int len = words.size();
	if (len < 3)
	{
		return -1;
	}
	if (words[0].m_token != FUNCTION || 
		words[1].m_token!=ID ||
		words[2].m_token!=BEGIN_SCRIPT ||
		words[len-1].m_token!=END_SCRIPT)
	{
		fprintf(stderr, "函数/流程开始语法错误，或者没有用end结尾!\n");
		return -1;
	}
	printf("!!!BEGIN %s\n", words[1].m_yylval.id_val.c_str());
	int start = 3;
	int rc;
	while (1)
	{
		list<string>  codelist;
		rc = ParseStatement(words, start, codelist);
		if (rc < 0)
		{
			return -1;
		}

		list<string>::const_iterator it;
		for (it = codelist.begin(); it != codelist.end(); ++it)
		{
			printf("%s", (*it).c_str());
		}
		if (rc == 0)
		{
			break;
		}

		start += rc;
	}
	return 0;
}

/*
* 语法分析一条语句，成功则返回语句的长度,否则返回－1, 如果遇到END单词，返回0
*/
static int ParseStatement(const vector<CToken> & words, unsigned int start, 
	list<string> & codelist)
{
	GenTmpVar(true);

	codelist.clear();
	int nStatementLen = -1;
	if ( words[start].m_token == END_SCRIPT)
	{
		char code[2048];
		sprintf(code, "LABEL l%d\n", gs_nModelEndLabel);
		codelist.push_back(code);
		codelist.push_back("!!!END\n");
		return 0;
	}
	//简单变量赋值语句
	else if ( words[start].m_token == VAR &&
		words[start+1].m_token == '=')
	{
		nStatementLen = AssignAnalyze(words, start, codelist);
	}
	//if 语句
	else if ( words[start].m_token == IF)
	{
		nStatementLen = IfAnalyze(words, start, codelist);
	}
	// while 语句
	else if ( words[start].m_token == WHILE)
	{
		nStatementLen = WhileAnalyze(words, start, codelist);	
	}
	// break
	else if ( words[start].m_token == BREAK)
	{
		nStatementLen = BreakAnalyze(words, start, codelist);	
	}
	// continue
	else if ( words[start].m_token == CONTINUE)
	{
		nStatementLen = ContinueAnalyze(words, start, codelist);	
	}
	// return
	else if ( words[start].m_token == RETURN)
	{
		nStatementLen = ReturnAnalyze(words, start, codelist);	
	}
	//数组变量开头，可能是赋值，也可能是表达式语句
	else if ( words[start].m_token == VAR &&
		words[start+1].m_token == LOFFSET)
	{
		nStatementLen = ArrayStartAnalyze(words, start, codelist);	
	}
	else
	{
		//默认作为 expr ; 来分析
		nStatementLen = RunExprAnalyze(words, start, codelist);	
	}


	if (nStatementLen > 0)
	{
		//清除一下运算栈，防止一些无用的变量把栈撑死
		codelist.push_back("CLEAR\n");
	}
	return nStatementLen;
}
/*
*分析一个 数组变量开头的 语句，成功返回语句的长度，失败返回－1
*数组变量开头，可能是赋值，也可能是表达式语句
*/
static int ArrayStartAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	/*根据[]的匹配来跳过数组变量*/
	int nMatch = 0;
	int i;
	for (i = start+1; i < words.size(); ++i)
	{
		if (words[i].m_token == LOFFSET)
		{
			++nMatch;
		}
		else if (words[i].m_token == ROFFSET)
		{
			--nMatch;
		}

		if (nMatch == 0)
		{
			break;
		}
	}
	if (nMatch != 0)
	{
		fprintf(stderr, "数组变量语法错误! lineno=%d\n", 
			words[start].m_yylval.lineno);
		return -1;
	}
	if ( (i+1) >= words.size())
	{
		fprintf(stderr, "不可预料的结尾! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	if (words[i+1].m_token == '=') /*对数组的赋值语句*/
	{
		return ArrayAssignAnalyze(words, start, codelist);
	}
	else /*表达式语句*/
	{
		return RunExprAnalyze( words, start, codelist);
	}
}
/*
*分析一个 左边为数组变量的 语句，成功返回语句的长度，失败返回－1
*/
static int ArrayAssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	/*根据[]的匹配来跳过数组变量*/
	int nMatch = 0;
	int i;
	for (i = start+1; i < words.size(); ++i)
	{
		if (words[i].m_token == LOFFSET)
		{
			++nMatch;
		}
		else if (words[i].m_token == ROFFSET)
		{
			--nMatch;
		}

		if (nMatch == 0)
		{
			break;
		}
	}
	if (nMatch != 0)
	{
		fprintf(stderr, "数组变量语法错误! lineno=%d\n", 
			words[start].m_yylval.lineno);
		return -1;
	}
	int nPos = i+1; // '='的位置
	if ( (i+3) >= words.size()) // a[...] = ... ;
	{
		fprintf(stderr, "不可预料的结尾! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	list<string> lcodes, rcodes; // =左右两侧的代码
	int nlLen, nrLen; // = 左右两侧单词的个数
	nlLen = ArrayElementAnalyze(words, start, lcodes, true);
	nrLen = ExpressionAnalyze(words, nPos+1, rcodes);
	if (nlLen < 4 || nrLen < 1)
	{
		return -1;
	}
	if (words[start+nlLen+1+nrLen].m_token != ';')
	{
		fprintf(stderr, "语句末尾需要一个';'! lineno=%d\n",
			words[start+nlLen+1+nrLen].m_yylval.lineno);
		return -1;
	}
	codelist.insert(codelist.end(), rcodes.begin(), rcodes.end());
	codelist.insert(codelist.end(), lcodes.begin(), lcodes.end());

	char code[2048];
	memset(code, 0, sizeof(code));
	snprintf(code, sizeof(code)-1, "SAVARY %s\n", words[start].ToString().c_str());
	codelist.push_back(code);

	//  a[...]     =  .....    ;
	return nlLen + 1 + nrLen + 1;
}
/*
*分析一个 expr; 语句，成功返回语句的长度，失败返回－1
*/
static int RunExprAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	int nExprLen = ExpressionAnalyze(words, start, codelist);
	if (nExprLen <= 0)
	{
		return -1;
	}
	if ( (start+nExprLen) >= words.size() ||
		words[start+nExprLen].m_token != ';')
	{
		fprintf(stderr, "语句末尾需要一个分号! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}

	return nExprLen+1;
}
/*
*分析一个赋值语句，成功返回语句的长度，失败返回－1
*/
static int AssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	codelist.clear();
	// VAR = expr ;
	list<string> codes;
	int rc = ExpressionAnalyze(words, start+2, codes);
	if (rc < 0)
	{
		return -1;
	}
	codelist.insert(codelist.end(), codes.begin(), codes.end());
	if ((start+2+rc) >= words.size() ||
		words[start+2+rc].m_token != ';')
	{
		fprintf(stderr, "赋值语句末尾需要一个分号结束! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	char code[2048];
	sprintf(code, "SAV %s\n", words[start].ToString().c_str());
	codelist.push_back(code);
	return 3 + rc;
}
/*
*分析一个IF语句，成功返回语句的长度，失败返回－1
*/
static int IfAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	vector<string> codevector;

	int labelTRUE = GetLabel();
	int labelFALSE = GetLabel();
	int labelEND = GetLabel();

	// IF expr  THEN stmts ENDIF
	// IF expr  THEN stmts ELSE stmts ENDIF
	list<string> codes;
	char code[2048];

	// 条件表达式的分析
	int nExprLen = ExpressionAnalyze(words, start+1, codes);
	if (nExprLen < 0)
	{
		return -1;
	}
	codevector.insert(codevector.end(), codes.begin(), codes.end());
	if ( (start+1+nExprLen) >= words.size() ||
		words[start+1+nExprLen].m_token != THEN)
	{
		fprintf(stderr, "IF条件语句需要使用THEN! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	sprintf(code, "GOTOFALSE l%d\n", labelEND);
	codevector.push_back(code);
	//记住这个位置，后面根据ELSE/ENDIF来修改
	int pos1 = codevector.size() - 1;

	int i; // then后面的语句的长度
	for (i = 0;;)
	{
		//                IF   expr THEN
		int basep = start+1+nExprLen+1;

		if ( (basep+i) >= words.size() )
		{
			fprintf(stderr, "IF条件语句不完整! lineno=%d\n",
				words[start].m_yylval.lineno);
			return -1;	
		}
		if ( words[basep+i].m_token == ENDIF )
		{
			break;
		}

		if (words[basep+i].m_token == ELSE)
		{
			sprintf(code, "GOTO l%d\n", labelEND);
			codevector.push_back(code);
			break;
		}
		int sttlen = ParseStatement(words, basep+i, codes);
		if (sttlen == 0)
		{
			fprintf(stderr, "IF条件语句不完整! lineno=%d\n",
				words[start].m_yylval.lineno);
			return -1;	
		}
		if (sttlen < 0)
		{
			return -1;
		}
		codevector.insert(codevector.end(), codes.begin(), codes.end());

		i += sttlen;
	}
	//              IF expr   THEN stt
	if ( words[start+1+nExprLen+1+i].m_token == ENDIF )
	{
		sprintf(code, "LABEL l%d\n", labelEND);
		codevector.push_back(code);
		codelist.assign(codevector.begin(), codevector.end());

		//    IF  expr   THEN stt ENDIF
		return 1+nExprLen+1+  i  +1 ;

	}

	// ELSE
	int j = 0; // ELSE后面的statements的总长度
	if ( words[start+1+nExprLen+1+i].m_token == ELSE )
	{
		sprintf(code, "LABEL l%d\n", labelFALSE);
		codevector.push_back(code);

		sprintf(code, "GOTOFALSE l%d\n", labelFALSE);
		codevector.at(pos1) = code;
		for (;;)
		{
			//                  IF   expr   THEN  stmts ELSE
			int basep =  start + 1 + nExprLen + 1 + i + 1;
	
			if ( (basep+j) >= words.size() )
			{
				fprintf(stderr, "IF条件语句不完整! lineno=%d\n",
					words[start].m_yylval.lineno);
				return -1;	
			}
			if ( words[basep+j].m_token == ENDIF )
			{
				break;
			}
			int sttlen = ParseStatement(words, basep+j, codes);
			if (sttlen == 0)
			{
				fprintf(stderr, "IF条件语句不完整! lineno=%d\n",
					words[start].m_yylval.lineno);
				return -1;	
			}
			if (sttlen < 0)
			{
				return -1;
			}
			codevector.insert(codevector.end(), codes.begin(), codes.end());
	
			j += sttlen;

		}
	}
	//               IF expr  THEN stts ELSE stts
	if ( words[start+1+nExprLen+1+  i  +1  +j].m_token == ENDIF )
	{
			sprintf(code, "LABEL l%d\n", labelEND);
			codevector.push_back(code);
			codelist.assign(codevector.begin(), codevector.end());

			//    IF  expr   THEN stt ELSE stt ENDIF
			return 1+nExprLen+1+  i  +1  +j + 1;
	}
	fprintf(stderr, "IF语句不完整! lineno=%d\n", 
				words[start].m_yylval.lineno);
	return -1;
}
/*
*分析一个WHILE语句，成功返回语句的长度，失败返回－1
*/
static int WhileAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	int labelBEGIN = GetLabel();
	int labelEND = GetLabel();

	// WHILE expr DO stmts ENDWHILE
	list<string> codes;
	char code[2048];


	//开始的标签
	sprintf(code, "LABEL l%d\n", labelBEGIN);
	codelist.push_back(code);

	// 条件表达式的分析
	int nExprLen = ExpressionAnalyze(words, start+1, codes);
	if (nExprLen < 0)
	{
		return -1;
	}
	codelist.insert(codelist.end(), codes.begin(), codes.end());
	if ( (start+1+nExprLen) >= words.size() ||
		words[start+1+nExprLen].m_token != DO)
	{
		fprintf(stderr, "WHILE循环语句需要使用DO! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	sprintf(code, "GOTOFALSE l%d\n", labelEND);
	codelist.push_back(code);

	LabelPair lp;
	lp.nBegin = labelBEGIN;
	lp.nEnd = labelEND;
	gs_lpstk.push(lp);

	int i; // DO后面的语句的长度
	for (i = 0;;)
	{
		//              WHILE expr  DO
		int basep = start+1+nExprLen+1;

		if ( (basep+i) >= words.size() )
		{
			fprintf(stderr, "WHILE循环语句不完整! lineno=%d\n",
				words[start].m_yylval.lineno);
			return -1;	
		}
		if ( words[basep+i].m_token == ENDWHILE )
		{
			if (gs_lpstk.empty())
			{
				fprintf(stderr, "WHILE和ENDWHILE不匹配! lineno=%d\n",
					words[start].m_yylval.lineno);
				return -1;	
			}
			gs_lpstk.pop();

			sprintf(code, "GOTO l%d\n", labelBEGIN);
			codelist.push_back(code);
			break;
		}

		int sttlen = ParseStatement(words, basep+i, codes);
		if (sttlen == 0)
		{
			fprintf(stderr, "WHILE循环语句不完整! lineno=%d\n",
				words[start].m_yylval.lineno);
			return -1;	
		}
		if (sttlen < 0)
		{
			return -1;
		}
		codelist.insert(codelist.end(), codes.begin(), codes.end());

		i += sttlen;
	}
	//               WHILE expr DO stts 
	if ( words[start+1+nExprLen+1+  i  ].m_token == ENDWHILE )
	{
			sprintf(code, "LABEL l%d\n", labelEND);
			codelist.push_back(code);
			return 1+nExprLen+1+ i +1;
	}
	fprintf(stderr, "WHILE循环语句不完整! lineno=%d\n",
				words[start].m_yylval.lineno);
	return -1;
}
/*
*分析一个break句，成功返回语句的长度，失败返回－1
*/
static int BreakAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	char code[2048];

	if (words[start+1].m_token != ';')
	{
		fprintf(stderr, "BREAK后面需要一个分号! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	if (gs_lpstk.empty())
	{
		fprintf(stderr, "BREAK不在循环语句内! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	LabelPair lp;
	lp = gs_lpstk.top();
	sprintf(code, "GOTO l%d\n", lp.nEnd);
	codelist.push_back(code);
	return 2;
}
/*
*分析一个continue句，成功返回语句的长度，失败返回－1
*/
static int ContinueAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	char code[2048];

	if (words[start+1].m_token != ';')
	{
		fprintf(stderr, "CONTINUE后面需要一个分号! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	if (gs_lpstk.empty())
	{
		fprintf(stderr, "CONTINUE不在循环语句内! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}
	LabelPair lp;
	lp = gs_lpstk.top();
	sprintf(code, "GOTO l%d\n", lp.nBegin);
	codelist.push_back(code);
	return 2;
}
/*
*分析一个return句，成功返回语句的长度，失败返回－1
*/
static int ReturnAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	char code[2048];
	list<string> codes;

	int nExprLen = 0;
	if (words[start+1].m_token != ';')
	{
		nExprLen = ExpressionAnalyze(words, start+1, codes);
		if (nExprLen <= 0)
		{
			return -1;
		}
	}
	if (nExprLen > 0)
	{
		codelist.insert(codelist.end(), codes.begin(), codes.end());
		codelist.push_back("SAV $OUTDATA\n");
	}
	if (words[start+1+nExprLen].m_token != ';')
	{
		fprintf(stderr, "RETURN语句末尾需要一个分号! lineno=%d\n",
			words[start].m_yylval.lineno);
		return -1;
	}

	sprintf(code, "GOTO l%d\n", gs_nModelEndLabel);
	codelist.push_back(code);
	return 1+nExprLen+1;
}

int GetLabel()
{
	static int LabelIndex = 0;
	return  LabelIndex++;
}
