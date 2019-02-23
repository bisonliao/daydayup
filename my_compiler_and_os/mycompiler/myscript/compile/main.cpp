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
	//�ʷ�����
	rc = WordAnalyze(words, argv[1]);
	if (rc < 0)
	{
		return  -1;
	}
	if (rc == 0)
	{
		return 0;
	}
	//�﷨�����������/����
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
			fprintf(stderr, "�﷨����ʧ��!\n");
			return -1;
		}
	}
	#else
	if (parse(words) != 0)
	{
		fprintf(stderr, "�﷨����ʧ��!\n");
		return -1;
	}
	#endif
	return 0;
}

/*
*���ļ�filename���дʷ�������ʧ�ܷ��أ�1������Ҳ����ֱ���˳�����
*�ɹ����ط�����õĵ��ʵĸ���
*/
int WordAnalyze(vector<CToken> & words, const char * filename)
{
	int retcode;
	extern FILE * yyin;
	if ( (yyin = fopen(filename, "rb")) == NULL)
	{
		fprintf(stderr, "���ļ�[%s]ʧ��!\n", filename);
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
