#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "token.h"
#include <vector>
using namespace std;

int yylex(YYLVAL & yylval);
int WordAnalyze(vector<CToken> & words, const char * filename);
int parse(const vector<CToken> & words);
int ExpressionAnalyze(const vector<CToken> & words, unsigned int start);


int main(int argc, char ** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "usage: %s source_file\n", argv[0]);
		return -1;
	}
	vector<CToken> words;
	words.reserve(1024);
	int rc;
	//词法分析
	rc = WordAnalyze(words, argv[1]);
	if (rc < 0)
	{
		return  -1;
	}
	if (rc == 0)
	{
		return 0;
	}
	//语法分析多个流程/函数
	#if 1
	int i = 0; 
	while (i < words.size())
	{
		vector<CToken> vword;
		vword.clear();
		while (i < words.size())
		{
			vword.push_back(words[i]);
			++i;
			if (words[i-1].m_token == END_SCRIPT)
			{
				break;
			}
		}
		if (parse(vword) != 0)
		{
			fprintf(stderr, "语法分析失败!\n");
			return -1;
		}
	}
	#else
	if (parse(words) != 0)
	{
		fprintf(stderr, "语法分析失败!\n");
		return -1;
	}
	#endif
	return 0;
}

/*
*对文件filename进行词法分析，失败返回－1，或者也可能直接退出程序
*成功返回分析获得的单词的个数
*/
int WordAnalyze(vector<CToken> & words, const char * filename)
{
	int retcode;
	extern FILE * yyin;
	if ( (yyin = fopen(filename, "rb")) == NULL)
	{
		fprintf(stderr, "打开文件[%s]失败!\n", filename);
		return -1;
	}
	YYLVAL yylval;
	for (;;)
	{
		retcode = yylex(yylval);
		if (retcode == 0)
		{
			break;
		}
		CToken ttt(retcode, yylval);
		words.push_back(ttt);
	}
	fclose(yyin);
	return words.size();
}
