#ifndef __COMPILE_H_INCLUDED__
#define __COMPILE_H_INCLUDED__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <deque>
#include <list>
using namespace std;

#include "token.h"

using lnb::CToken;
using lnb::YYLVAL;


namespace lnb {
/*
*分析一个表达式，成功返回表达式的长度，失败返回－1
*/
int ExpressionAnalyze(const vector<CToken> & words, unsigned int start,
   list<string> &codelist);
//int ExpressionAnalyze(const vector<CToken> & words, unsigned int start);

CToken GenTmpVar(bool CleanAll = false);
int GetLabel();
int ArrayElementAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist, bool isLeft = false);
int MapElementAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist, bool isLeft = false);

int yylex(YYLVAL & yylval);
int WordAnalyze(vector<CToken> & words, const char * filename);
int parse(const vector<CToken> & words, deque<string> & PCodeList);
int compile(const string & sSourceFile,  deque<string> & PCodeList);
};

#endif

