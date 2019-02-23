#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "token.h"
#include <vector>
#include <stack>
#include <list>
#include "compile.h"
using namespace std;


static int RunOperator(stack<CToken> & oprnt_stk, CToken & topword, 
	list<string> & codelist);
static int FunctionAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist);

/*
*����һ�����ʽ���ɹ����ر��ʽ�ĳ��ȣ�ʧ�ܷ��أ�1
*/
int lnb::ExpressionAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist)
{

	codelist.clear();

	char code[2048];

	int nbrk = 0; //�������ŵļ���

	if (start < 0 || start >= words.size())
	{
		return -1;
	}
	//���һ��ʼ�Ͳ��ǲ��������߲���������ô���ʽ����ʧ��
	if (  !(words[start].IsExprOperator() || words[start].IsExprOperant() || 
			words[start].m_token == ID || words[start].m_token == LBRK )  )
	{
		fprintf(stderr, "%s %d:���ʽ���Ϸ�.\n",
					words[start].m_yylval.srcfile.c_str(),
					words[start].m_yylval.lineno);
		return -1;	
	}
	int i ;
	bool hasNoOperator = true; //�������ʽ�����в�������
	int ExprLen; //�����ɹ���ʹ�õĵ��ʵĸ�����Ҳ���Ǳ��ʽ�ĳ���



	stack<CToken> opt_stk, oprnt_stk; //������ջ�Ͳ�����ջ
	for ( i = start; i < words.size() && 
					(words[i].IsExprOperator() || words[i].IsExprOperant() ||
					 words[i].m_token == LBRK || words[i].m_token == RBRK ||
					 words[i].m_token == ID);
					 ++i)
	{
		CToken word = words[i];

		if (word.m_token == SUB)
		{
			if (i == start) //��һ������
			{
				word.m_token = MINUS;
			}
			else if ( i > start && words[i-1].m_token == LBRK ) //��һ�������� (
			{
				word.m_token = MINUS;
			}
		}

		if (word.m_token == CONST_FLOAT || 
			word.m_token == CONST_INT ||
			word.m_token == CONST_STRING) /*ֱ����ջ*/
		{
			oprnt_stk.push(word);

			sprintf(code, "PUSH %s\n", word.ToString().c_str());
			codelist.push_back(code);

		}
		else if (word.m_token == VAR)
		{
			if ( (i+1) < words.size() &&
				words[i+1].m_token == LOFFSET) /*һ������Ԫ�ؿ�ʼ��*/
			{
				/*����Ԫ�ص��±���һ�����ʽ*/
				list<string> arraycodelist;
				int nSubscriptLen =  ArrayElementAnalyze(words, i, arraycodelist);
				if (nSubscriptLen < 4)
				{
					return -1;
				}
				codelist.insert(codelist.end(), 
					arraycodelist.begin(), arraycodelist.end()); 
				//����һ���м������ѹջ
				CToken tmptoken = lnb::GenTmpVar();
				tmptoken.m_yylval.lineno = word.m_yylval.lineno;
				tmptoken.m_yylval.srcfile = word.m_yylval.srcfile;
				oprnt_stk.push(tmptoken);
					/*
				//����Ϊ�м����
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);
				*/


				i += (nSubscriptLen - 1);
			}
			else if ( (i+1) < words.size() &&
				words[i+1].m_token == LMAP) /*һ��mapԪ�ؿ�ʼ��*/
			{
				/*mapԪ�ص��±���һ�����ʽ*/
				list<string> mapcodelist;
				int nSubscriptLen =  MapElementAnalyze(words, i, mapcodelist);
				if (nSubscriptLen < 4)
				{
					return -1;
				}
				codelist.insert(codelist.end(), 
					mapcodelist.begin(), mapcodelist.end()); 
				//����һ���м������ѹջ
				CToken tmptoken = lnb::GenTmpVar();
				tmptoken.m_yylval.lineno = word.m_yylval.lineno;
				tmptoken.m_yylval.srcfile = word.m_yylval.srcfile;
				oprnt_stk.push(tmptoken);


				i += (nSubscriptLen - 1);
			}
			else /*����һ��ͨ����*/
			{
				oprnt_stk.push(word); /*ֱ����ջ*/

				sprintf(code, "PUSH %s\n", word.ToString().c_str());
				codelist.push_back(code);
			}
		}
		else if (word.m_token == ID) //����
		{
			if (words[i+1].m_token != LBRK)
			{
				fprintf(stderr, "%s %d:���ʽ���Ϸ�, %s����ϣ������ '(' .\n",
					word.m_yylval.srcfile.c_str(),
					word.m_yylval.lineno,
					word.ToString().c_str());
				return -1;
			}
			else 
			{
				//�����ķ���
				list<string> codes;
				int nFunLen = FunctionAnalyze(words, i, codes);
				if (nFunLen < 3)
				{
					return -1;
				}
				codelist.insert(codelist.end(), codes.begin(), codes.end());
				//����һ���м������ѹջ
				CToken tmptoken = lnb::GenTmpVar();
				tmptoken.m_yylval.lineno = word.m_yylval.lineno;
				tmptoken.m_yylval.srcfile = word.m_yylval.srcfile;
				oprnt_stk.push(tmptoken);
				/*
				//����Ϊ�м����
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);
				*/

				i += (nFunLen - 1);
			}
		}
		else if (word.m_token == LBRK)
		{
			++nbrk;
			//ֱ����ջ
			opt_stk.push(word);
		}
		else if (word.m_token == RBRK)
		{
			if (nbrk == 0) //���ʽ�ߵ��˾�ͷ
			{
				break;
			}
			--nbrk;

			int iNumOptInBrk = 0;
			while (1)
			{
				if (oprnt_stk.empty() || opt_stk.empty())
				{
					fprintf(stderr, "%s %d:���ʽ���������Ų�ƥ��! \n",
						words[start].m_yylval.srcfile.c_str(),
						words[start].m_yylval.lineno);
					return -1;
				}
				CToken topword = opt_stk.top();
				opt_stk.pop();
				++iNumOptInBrk;
				if (topword.m_token == LBRK)
				{
					CToken ttt = oprnt_stk.top();
					/*
					if (ttt.ToString()[0] != 'v' && iNumOptInBrk > 1)
					{
					sprintf(code, "PUSH %s\n", ttt.ToString().c_str());
					codelist.push_back(code);
					}
					*/
					break;
				}
				list<string> codes;
				if (RunOperator(oprnt_stk, topword, codes))
				{
					return -1;
				}
				codelist.insert(codelist.end(), codes.begin(), codes.end());

				//����һ���м������ѹջ
				CToken tmptoken = lnb::GenTmpVar();
				tmptoken.m_yylval.lineno = topword.m_yylval.lineno;
				tmptoken.m_yylval.srcfile = topword.m_yylval.srcfile;
				oprnt_stk.push(tmptoken);
				/*
				//����Ϊ�м����
				sprintf(code, "SAV %s\n", tmptoken.ToString().c_str());
				codelist.push_back(code);
				*/
				
			}
		}
		else //����ǲ�����
		{

			hasNoOperator = false;
			if (opt_stk.empty() ||
				word.GetPriorityOUT() > opt_stk.top().GetPriorityIN())
			{
				//�������ջΪ�գ����ߵ�ǰ�������ȼ�����ջ�����ţ���ôѹջ
				opt_stk.push(word);
			}
			else
			{
				while ( !(opt_stk.empty()) &&
					word.GetPriorityOUT() <= opt_stk.top().GetPriorityIN())
				{
					//����һ����������
					CToken topword = opt_stk.top();
					opt_stk.pop();
					if (topword.GetToken() == LBRK)
					{
						fprintf(stderr, "%s %d:���ʽ���������Ų�ƥ��! \n",
							words[start].m_yylval.srcfile.c_str(),
							words[start].m_yylval.lineno);
						return -1;
					}
					list<string> codes;
					if (RunOperator(oprnt_stk, topword, codes))
					{
						return -1;
					}
					codelist.insert(codelist.end(), codes.begin(), codes.end());

					//����һ���м������ѹջ
					CToken tmptoken = lnb::GenTmpVar();
					tmptoken.m_yylval.srcfile = topword.m_yylval.srcfile;
					tmptoken.m_yylval.lineno = topword.m_yylval.lineno;
					oprnt_stk.push(tmptoken);
				}
				if (word.GetToken() != RBRK)
				{
					opt_stk.push(word);
				}
			}
		}
	}
	ExprLen = i - start;
	if (hasNoOperator) //����������ʽû�к��в�����
	{
		if (oprnt_stk.size() != 1) //�����������������1������
		{
			fprintf(stderr, "%s %d:���ʽ���Ϸ�! \n",
				oprnt_stk.top().m_yylval.srcfile.c_str(),
				oprnt_stk.top().m_yylval.lineno);
			return -1;
		}
	}
	
	while (!opt_stk.empty())
	{
			//����һ����������
			CToken topword = opt_stk.top();
			opt_stk.pop();
			if (topword.GetToken() == LBRK || topword.GetToken() == RBRK)
			{
				fprintf(stderr, "%s %d:����������ƥ��! \n",
					topword.m_yylval.srcfile.c_str(),
					topword.m_yylval.lineno);
				return -1;
			}
			list<string> codes;
			if (RunOperator(oprnt_stk, topword, codes))
			{
				return -1;
			}
			codelist.insert(codelist.end(), codes.begin(), codes.end());
			//����һ���м������ѹջ
			CToken tmptoken = lnb::GenTmpVar();
			tmptoken.m_yylval.lineno = topword.m_yylval.lineno;
			tmptoken.m_yylval.srcfile = topword.m_yylval.srcfile;
			oprnt_stk.push(tmptoken);
	}
	if (oprnt_stk.size() != 1)
	{
		fprintf(stderr, "%s %d:���ʽ�Ƿ�! \n",
					words[start].m_yylval.srcfile.c_str(),
					words[start].m_yylval.lineno);
		return -1;
		
	}
	return ExprLen;
}
/*
*������topword�Բ�����ջ���Ԫ�ؽ���һ�����㣬�ѽ�����浽������ջ
*�ɹ�����0�� ʧ�ܷ��أ�1
*/
static int RunOperator(stack<CToken> & oprnt_stk, CToken & topword, 
	list<string> & codelist)
{
	//����������
	codelist.clear();

	char code[2048];

	int OprntNum = topword.GetOprntNumNeed();
	if (oprnt_stk.size() < OprntNum)
	{
		fprintf(stderr, "%s %d:��������������! (%d, %d) \n",
					topword.m_yylval.srcfile.c_str(),
					topword.m_yylval.lineno,
					oprnt_stk.size(),
					OprntNum);
		return -1;
	}
	if (OprntNum == 1)
	{
		CToken oprnt;
		oprnt =  oprnt_stk.top();
		oprnt_stk.pop();
		//����
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
		//����
		sprintf(code, "%s\n", topword.ToString().c_str());
		codelist.push_back(code);
	}
	else
	{
		fprintf(stderr, "%s %d:%s����һ����Ч�Ĳ�����! \n",
				topword.m_yylval.srcfile.c_str(),
				topword.m_yylval.lineno,
				topword.ToString().c_str());
		return -1;
	}

	return 0;
}

CToken lnb::GenTmpVar(bool CleanAll /*= false*/)
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
*����һ���������ɹ����ر��ʽ�ĳ��ȣ�ʧ�ܷ��أ�1
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
	int i = 0; //�����ĳ��Ⱥ�
	int nArgs = 0; //�����ĸ���
	list<CToken> args;
	if (words[start+2].m_token != RBRK) //���в���
	{
		for (; ; )
		{
			if ( (start+2+i)>=words.size())
			{
				fprintf(stderr, "%s %d:�����﷨����! \n",
					words[start].m_yylval.srcfile.c_str(),
					words[start].m_yylval.lineno);
				return -1;
			}
			list<string> codes;
			int nExprLen = ExpressionAnalyze(words, start+2+i, codes);
			if (nExprLen <= 0)
			{
				fprintf(stderr, "%s %d:�����������Ϸ�! \n",
					words[start].m_yylval.srcfile.c_str(),
					words[start].m_yylval.lineno);
				return -1;
			}
			i += nExprLen;
			++ nArgs;
			codelist.insert(codelist.end(), codes.begin(), codes.end());

			CToken arg = lnb::GenTmpVar();
			args.push_back(arg);
			/*
			sprintf(code, "SAV %s\n", arg.ToString().c_str());
			codelist.push_back(code);
			*/

			if (words[start+2+i].m_token == RBRK) //�����б����
			{
				break;
			}
			else if (words[start+2+i].m_token != COMMA)
			{
				fprintf(stderr, "%s %d:�����������Ϸ�! ��Ҫ�ö��ŷָ����! \n",
					words[start].m_yylval.srcfile.c_str(),
					words[start].m_yylval.lineno);
				return -1;
			}
			++i;
		}
	}
	list<CToken>::const_iterator it;
	for (it = args.begin(); it != args.end(); ++it)
	{
	/*
		if ((*it).ToString()[0] != 'v')
		{
		sprintf(code, "PUSH %s\n", (*it).ToString().c_str());
		codelist.push_back(code);
		}
	*/
	}
	sprintf(code, "PUSH ^%d\n", nArgs);
	codelist.push_back(code);
	sprintf(code, "CALL %s\n", words[start].ToString().c_str());
	codelist.push_back(code);

	return 2 + i + 1;
}
/*
*����һ������Ԫ�أ��ɹ���������Ԫ�صĳ��ȣ�ʧ�ܷ��أ�1
* isLeft ��˵����Ԫ������Ϊ��ֵ��
*/
int lnb::ArrayElementAnalyze(const vector<CToken> & words, unsigned int start, 
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

	/*�����±�*/
	list<string> codes;
	int nExprLen = ExpressionAnalyze(words, start+2, codes);
	if (nExprLen <= 0)
	{
		fprintf(stderr, "%s %d:����Ԫ���±겻�Ϸ�! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	codelist.insert(codelist.end(), codes.begin(), codes.end());

	if ( (start+1+nExprLen+1) >= words.size()
	   || words[start+1+nExprLen+1].m_token != ROFFSET) /*��������������*/
	{
		fprintf(stderr, "%s %d:����Ԫ���±겻�Ϸ�, ��Ҫ']'! \n",
			words[start].m_yylval.srcfile.c_str(),
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
		codelist.push_back(code);
	}

	//   $v    [   subscript ] 
	return 1 + 1 + nExprLen + 1;
}
/*
*����һ��mapԪ�أ��ɹ�����mapԪ�صĳ��ȣ�ʧ�ܷ��أ�1
* isLeft ��˵mapԪ������Ϊ��ֵ��
*/
int lnb::MapElementAnalyze(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist, bool isLeft)
{
	codelist.clear();
	char code[2048];

	if (start+3 >= words.size())
	{
		return -1;
	}
	if ( !(words[start].m_token == VAR && words[start+1].m_token == LMAP) )
	{
		return -1;
	}

	/*�����±�*/
	list<string> codes;
	int nExprLen = ExpressionAnalyze(words, start+2, codes);
	if (nExprLen <= 0)
	{
		fprintf(stderr, "%s %d:mapԪ���±겻�Ϸ�! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	codelist.insert(codelist.end(), codes.begin(), codes.end());

	if ( (start+1+nExprLen+1) >= words.size()
	   || words[start+1+nExprLen+1].m_token != RMAP) /*�������һ�����*/
	{
		fprintf(stderr, "%s %d:mapԪ���±겻�Ϸ�, ��Ҫ'}'! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}


	memset(code, 0, sizeof(code));
	if (!isLeft)
	{
		snprintf(code, sizeof(code)-1, "PUSMAP %s\n", words[start].ToString().c_str());
		codelist.push_back(code);
	}

	//   $v    {   subscript } 
	return 1 + 1 + nExprLen + 1;
}
