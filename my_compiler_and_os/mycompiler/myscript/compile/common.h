#ifndef _COMMON_H_INCLUDED_
#define _COMMON_H_INCLUDED_

#include "common.h"

/*
*分析一个表达式，成功返回表达式的长度，失败返回－1
*/
int ExpressionAnalyze(const vector<CToken> & words, unsigned int start,
   list<string> &codelist);
int parse(const vector<CToken> & words);
CToken GenTmpVar(bool CleanAll = false);
int GetLabel();
int ArrayElementAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist, bool isLeft = false);



#endif

