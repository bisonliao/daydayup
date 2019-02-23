#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "token.h"
#include <vector>
#include <deque>
#include <stack>
#include <list>
#include <string>
#include "compile.h"
using namespace std;
using namespace lnb;

static int AssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);
static int ParseStatement(const vector<CToken> & words, unsigned int start, 
	list<string> &codelist);
static int IfAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> &codelist);
static int IfAnalyzeEx(const vector<CToken> & words, unsigned int start,
	list<string> &codelist);
static int WhileAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist);
static int ForAnalyze(const vector<CToken> & words, unsigned int start,
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
static int MapAssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);
static int MapStartAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist);


namespace lnb {
typedef struct {
	int nBegin;
	int nEnd;
} LabelPair;

};

static stack<LabelPair> gs_lpstk; //��¼ѭ�������ת��ǩ��ջ��continue/break�õ�
static int gs_nModelEndLabel; //ÿ��ģ�����λ�õı�ǩ

/*
*�﷨�������ɹ�����0�� ʧ�ܷ��أ�1
*/
int lnb::parse(const vector<CToken> & words, deque<string> & PCodeList)
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
		fprintf(stderr, "����/���̿�ʼ�﷨���󣬻���û����end��β!\n");
		return -1;
	}
	{
		char szCodeBuf[100];
		snprintf(szCodeBuf, sizeof(szCodeBuf), "!!!BEGIN %s", words[1].m_yylval.id_val.c_str());
		//printf("!!!BEGIN %s\n", words[1].m_yylval.id_val.c_str());
		PCodeList.push_back(szCodeBuf);
	}

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
		//	printf("%s", (*it).c_str());
			int iLen = it->length();
			if (it->data()[iLen-1] == '\n')
			{
				PCodeList.push_back(it->substr(0, iLen-1) );
			}
			else
			{
				PCodeList.push_back(*it);
			}
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
* �﷨����һ����䣬�ɹ��򷵻����ĳ���,���򷵻أ�1, �������END���ʣ�����0
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
	//�򵥱�����ֵ���
	else if ( words[start].m_token == VAR &&
		words[start+1].m_token == '=')
	{
		nStatementLen = AssignAnalyze(words, start, codelist);
	}
	//if ���
	else if ( words[start].m_token == IF)
	{
		nStatementLen = IfAnalyzeEx(words, start, codelist);
	}
	// while ���
	else if ( words[start].m_token == WHILE)
	{
		nStatementLen = WhileAnalyze(words, start, codelist);	
	}
	// for ���
	else if ( words[start].m_token == FOR)
	{
		nStatementLen = ForAnalyze(words, start, codelist);	
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
	//���������ͷ�������Ǹ�ֵ��Ҳ�����Ǳ��ʽ���
	else if ( words[start].m_token == VAR &&
		words[start+1].m_token == LOFFSET)
	{
		nStatementLen = ArrayStartAnalyze(words, start, codelist);	
	}
	//Map������ͷ�������Ǹ�ֵ��Ҳ�����Ǳ��ʽ���
	else if ( words[start].m_token == VAR &&
		words[start+1].m_token == LMAP)
	{
		nStatementLen = MapStartAnalyze(words, start, codelist);	
	}
	else
	{
		//Ĭ����Ϊ expr ; ������
		nStatementLen = RunExprAnalyze(words, start, codelist);	
	}


	/*
	if (nStatementLen > 0)
	{
		//���һ������ջ����ֹһЩ���õı�����ջ����
		codelist.push_back("CLEAR\n");
	}
	*/
	return nStatementLen;
}
/*
*����һ�� ���������ͷ�� ��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
*���������ͷ�������Ǹ�ֵ��Ҳ�����Ǳ��ʽ���
*/
static int ArrayStartAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	/*����[]��ƥ���������������*/
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
		fprintf(stderr, "%s %d:��������﷨����! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	if ( (i+1) >= words.size())
	{
		fprintf(stderr, "%s %d:����Ԥ�ϵĽ�β! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	if (words[i+1].m_token == '=') /*������ĸ�ֵ���*/
	{
		return ArrayAssignAnalyze(words, start, codelist);
	}
	else /*���ʽ���*/
	{
		return RunExprAnalyze( words, start, codelist);
	}
}
/*
*����һ�� Map������ͷ�� ��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
*map������ͷ�������Ǹ�ֵ��Ҳ�����Ǳ��ʽ���
*/
static int MapStartAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	/*����{}��ƥ��������map����*/
	int nMatch = 0;
	int i;
	for (i = start+1; i < words.size(); ++i)
	{
		if (words[i].m_token == LMAP)
		{
			++nMatch;
		}
		else if (words[i].m_token == RMAP)
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
		fprintf(stderr, "%s %d:map�����﷨����! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	if ( (i+1) >= words.size())
	{
		fprintf(stderr, "%s %d:����Ԥ�ϵĽ�β! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	if (words[i+1].m_token == '=') /*������ĸ�ֵ���*/
	{
		return MapAssignAnalyze(words, start, codelist);
	}
	else /*���ʽ���*/
	{
		return RunExprAnalyze( words, start, codelist);
	}
}
/*
*����һ�� ���Ϊ��������� ��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
*/
static int ArrayAssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	/*����[]��ƥ���������������*/
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
		fprintf(stderr, "%s %d:��������﷨����! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	int nPos = i+1; // '='��λ��
	if ( (i+3) >= words.size()) // a[...] = ... ;
	{
		fprintf(stderr, "%s %d:����Ԥ�ϵĽ�β! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	list<string> lcodes, rcodes; // =��������Ĵ���
	int nlLen, nrLen; // = �������൥�ʵĸ���
	nlLen = ArrayElementAnalyze(words, start, lcodes, true);
	nrLen = ExpressionAnalyze(words, nPos+1, rcodes);
	if (nlLen < 4 || nrLen < 1)
	{
		return -1;
	}
	if (words[start+nlLen+1+nrLen].m_token != ';')
	{
		fprintf(stderr, "%s %d: ���ĩβ��Ҫһ��';'! \n",
			words[start+nlLen+1+nrLen].m_yylval.srcfile.c_str(),
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
*����һ�� ���Ϊmap������ ��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
*/
static int MapAssignAnalyze(const vector<CToken> & words, unsigned int start,
	list<string>& codelist)
{
	/*����{}��ƥ���������������*/
	int nMatch = 0;
	int i;
	for (i = start+1; i < words.size(); ++i)
	{
		if (words[i].m_token == LMAP)
		{
			++nMatch;
		}
		else if (words[i].m_token == RMAP)
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
		fprintf(stderr, "%s %d:map�����﷨����! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	int nPos = i+1; // '='��λ��
	if ( (i+3) >= words.size()) // a{...} = ... ;
	{
		fprintf(stderr, "%s %d:����Ԥ�ϵĽ�β! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	list<string> lcodes, rcodes; // =��������Ĵ���
	int nlLen, nrLen; // = �������൥�ʵĸ���
	nlLen = MapElementAnalyze(words, start, lcodes, true);
	nrLen = ExpressionAnalyze(words, nPos+1, rcodes);
	if (nlLen < 4 || nrLen < 1)
	{
		return -1;
	}
	if (words[start+nlLen+1+nrLen].m_token != ';')
	{
		fprintf(stderr, "%s %d: ���ĩβ��Ҫһ��';'! \n",
			words[start+nlLen+1+nrLen].m_yylval.srcfile.c_str(),
			words[start+nlLen+1+nrLen].m_yylval.lineno);
		return -1;
	}
	codelist.insert(codelist.end(), rcodes.begin(), rcodes.end());
	codelist.insert(codelist.end(), lcodes.begin(), lcodes.end());


	char code[2048];
	memset(code, 0, sizeof(code));
	snprintf(code, sizeof(code)-1, "SAVMAP %s\n", words[start].ToString().c_str());
	codelist.push_back(code);


	//  a{...}     =  .....    ;
	return nlLen + 1 + nrLen + 1;
}
/*
*����һ�� expr; ��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
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
		fprintf(stderr, "%s %d:���ĩβ��Ҫһ���ֺ�! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	//������һ�� expr;��䣬��������ջ������һ����������Ҫ���������ռ�ÿռ�
	codelist.push_back("CLEAR\n");

	return nExprLen+1;
}
/*
*����һ����ֵ��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
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
		fprintf(stderr, "%s %d:��ֵ���ĩβ��Ҫһ���ֺŽ���! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	char code[2048];
	sprintf(code, "SAV %s\n", words[start].ToString().c_str());
	codelist.push_back(code);
	return 3 + rc;
}
/*
*����һ��IF��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
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

	// �������ʽ�ķ���
	int nExprLen = ExpressionAnalyze(words, start+1, codes);
	if (nExprLen < 0)
	{
		return -1;
	}
	codevector.insert(codevector.end(), codes.begin(), codes.end());
	if ( (start+1+nExprLen) >= words.size() ||
		words[start+1+nExprLen].m_token != THEN)
	{
		fprintf(stderr, "%s %d: IF���������Ҫʹ��THEN! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	sprintf(code, "GOTOFALSE l%d\n", labelEND);
	codevector.push_back(code);
	//��ס���λ�ã��������ELSE/ENDIF���޸�
	int pos1 = codevector.size() - 1;

	int i; // then��������ĳ���
	for (i = 0;;)
	{
		//                IF   expr THEN
		int basep = start+1+nExprLen+1;

		if ( (basep+i) >= words.size() )
		{
			fprintf(stderr, "%s %d:IF������䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
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
			fprintf(stderr, "%s %d:IF������䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
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
	int j = 0; // ELSE�����statements���ܳ���
	if ( words[start+1+nExprLen+1+i].m_token == ELSE )
	{
		sprintf(code, "LABEL l%d\n", labelFALSE);
		codevector.push_back(code);

		sprintf(code, "GOTOFALSE l%d\n", labelFALSE);
		//codevector.at(pos1) = code;
		codevector[pos1] = code;
		for (;;)
		{
			//                  IF   expr   THEN  stmts ELSE
			int basep =  start + 1 + nExprLen + 1 + i + 1;
	
			if ( (basep+j) >= words.size() )
			{
				fprintf(stderr, "%s %d: IF������䲻����! \n",
					words[start].m_yylval.srcfile.c_str(),
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
				fprintf(stderr, "%s %d: IF������䲻����! \n",
					words[start].m_yylval.srcfile.c_str(),
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
	fprintf(stderr, "%s %d:IF��䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
				words[start].m_yylval.lineno);
	return -1;
}
static int IfAnalyzeEx(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	int offset = 0;

	vector<string> codevector;

	list<string> codes;
	char code[2048];

	int labelEND = GetLabel();
if_lable:
	int labelTRUE = GetLabel();
	int labelFALSE = GetLabel();

	// IF expr  THEN stmts ENDIF
	// IF expr  THEN stmts ELSE stmts ENDIF
	// IF expr  THEN stmts ELSIF stmts ...

	// �������ʽ�ķ���
	offset += 1; 	//���� IF
	int nExprLen = ExpressionAnalyze(words, start+offset, codes);
	if (nExprLen < 0)
	{
		return -1;
	}
	codevector.insert(codevector.end(), codes.begin(), codes.end());
	offset += nExprLen;	//���� �������ʽ
	if ( (start+offset) >= words.size() ||
		words[start+offset].m_token != THEN)
	{
		fprintf(stderr, "%s %d: IF���������Ҫʹ��THEN! \n",
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	sprintf(code, "GOTOFALSE l%d\n", labelEND);
	codevector.push_back(code);
	//��ס���λ�ã��������ELSE/ENDIF���޸�
	int pos1 = codevector.size() - 1;

	offset += 1; //���� THEN

	int i; // then��������ĳ���
	int basep = start+offset;
	for (i = 0;;)
	{
		//                IF   expr THEN

		if ( (basep+i) >= words.size() )
		{
			fprintf(stderr, "%s %d:IF������䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
				words[start].m_yylval.lineno);
			return -1;	
		}
		if ( words[basep+i].m_token == ENDIF )
		{
			break;
		}

		if (words[basep+i].m_token == ELSE  || words[basep+i].m_token == ELSIF)
		{
			sprintf(code, "GOTO l%d\n", labelEND);
			codevector.push_back(code);
			break;
		}
		int sttlen = ParseStatement(words, basep+i, codes);
		if (sttlen == 0)
		{
			fprintf(stderr, "%s %d: IF������䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
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
	offset += i; //���� stmt

	//              IF expr   THEN stt
	if ( words[start+offset].m_token == ENDIF )
	{
		sprintf(code, "LABEL l%d\n", labelEND);
		codevector.push_back(code);
		codelist.assign(codevector.begin(), codevector.end());

		offset += 1; // ENDIF
		return offset;
	}


	sprintf(code, "LABEL l%d\n", labelFALSE);
	codevector.push_back(code);

	sprintf(code, "GOTOFALSE l%d\n", labelFALSE);
	codevector[pos1] = code;

	if ( words[start+offset].m_token != ELSE  &&
	    words[start+offset].m_token != ELSIF)
	{
		fprintf(stderr, "%s %d:IF������䲻����! \n",
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;	
	}
	if (words[start+offset].m_token == ELSIF)
	{
		goto if_lable;
	}

	offset +=1; // ����ELSE
	// ELSE


	int j = 0; // ELSE�����statements���ܳ���
	basep =  start+ offset;
	for (;;)
	{
		//                  IF   expr   THEN  stmts ELSE

		if ( (basep+j) >= words.size() )
		{
			fprintf(stderr, "%s %d:IF������䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
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
			fprintf(stderr, "%s %d:IF������䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
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
	offset += j;
	//               IF expr  THEN stts ELSE stts
	if ( words[start+offset].m_token == ENDIF )
	{
			sprintf(code, "LABEL l%d\n", labelEND);
			codevector.push_back(code);
			codelist.assign(codevector.begin(), codevector.end());

			offset += 1;

			//    IF  expr   THEN stt ELSE stt ENDIF
			return offset;
	}
	fprintf(stderr, "%s %d:IF��䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
				words[start].m_yylval.lineno);
	return -1;
}
/*
*����һ��WHILE��䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
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


	//��ʼ�ı�ǩ
	sprintf(code, "LABEL l%d\n", labelBEGIN);
	codelist.push_back(code);

	// �������ʽ�ķ���
	int nExprLen = ExpressionAnalyze(words, start+1, codes);
	if (nExprLen < 0)
	{
		return -1;
	}
	codelist.insert(codelist.end(), codes.begin(), codes.end());
	if ( (start+1+nExprLen) >= words.size() ||
		words[start+1+nExprLen].m_token != DO)
	{
		fprintf(stderr, "%s %d: WHILEѭ�������Ҫʹ��DO! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	sprintf(code, "GOTOFALSE l%d\n", labelEND);
	codelist.push_back(code);

	LabelPair lp;
	lp.nBegin = labelBEGIN;
	lp.nEnd = labelEND;
	gs_lpstk.push(lp);

	int i; // DO��������ĳ���
	for (i = 0;;)
	{
		//              WHILE expr  DO
		int basep = start+1+nExprLen+1;

		if ( (basep+i) >= words.size() )
		{
			fprintf(stderr, "%s %d: WHILEѭ����䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
				words[start].m_yylval.lineno);
			return -1;	
		}
		if ( words[basep+i].m_token == ENDWHILE )
		{
			if (gs_lpstk.empty())
			{
				fprintf(stderr, "%s %d: WHILE��ENDWHILE��ƥ��! \n",
					words[start].m_yylval.srcfile.c_str(),
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
			fprintf(stderr, "%s %d:WHILEѭ����䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
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
	fprintf(stderr, "%s %d:WHILEѭ����䲻����! \n",
				words[start].m_yylval.srcfile.c_str(),
				words[start].m_yylval.lineno);
	return -1;
}
static int ForAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();
	int offset = 0;

	int labelBEGIN = GetLabel();
	int labelEND = GetLabel();
	int labelBetwennCD = GetLabel();

	// FOR ( a; b; c;) DO d ENDFOR;
	list<string> codeA, codeB, codeC, codeD;
	char code[2048];


	offset += 1; //���� FOR 

	if (start + offset >= words.size())
	{
		fprintf(stderr, "%s %d: FOR ��䲻����! \n", 
					words[start+offset].m_yylval.srcfile.c_str(),
					words[start+offset].m_yylval.lineno);
		return -1;
	}
	if (words[start+offset].m_token != LBRK)
	{
		fprintf(stderr, "%s %d: FOR ��� expect '(' \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	offset += 1; //���� LBRK
	//  A �ķ���
	int nAlen = AssignAnalyze(words, start+offset, codeA);
	if (nAlen < 0)
	{
		return -1;
	}
	offset += nAlen;
	if (start + offset >= words.size())
	{
		fprintf(stderr, "%s %d:FOR ��䲻����! \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}

	//  B �ķ���
	int nBlen = ExpressionAnalyze(words, start+offset, codeB);
	if (nBlen < 0)
	{
		return -1;
	}
	offset += nBlen;
	if (start + offset >= words.size())
	{
		fprintf(stderr, "%s %d: FOR ��䲻����! \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	if (words[offset+start].m_token != ';')
	{
		fprintf(stderr, "%s %d : FOR ���expect ';'! \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	offset += 1; // ����';'

	//  C �ķ���
	int nClen = AssignAnalyze(words, start+offset, codeC);
	if (nClen < 0)
	{
		return -1;
	}
	offset += nClen;
	if (start + offset + 1>= words.size())
	{
		fprintf(stderr, "%s %d: FOR ��䲻����! \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	if (words[offset+start].m_token != RBRK)
	{
		fprintf(stderr, "%s %d: FOR ���expect ')'! \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	offset += 1; // ����')'
	if (words[offset+start].m_token != DO)
	{
		fprintf(stderr, "%s %d: FOR ���expect ')'! \n", 
			words[start+offset].m_yylval.srcfile.c_str(),
			words[start+offset].m_yylval.lineno);
		return -1;
	}
	offset += 1; // ���� DO

	codelist.insert(codelist.end(), codeA.begin(), codeA.end());

	//��ʼ�ı�ǩ
	sprintf(code, "LABEL l%d\n", labelBEGIN);
	codelist.push_back(code);

	codelist.insert(codelist.end(), codeB.begin(), codeB.end());

	sprintf(code, "GOTOFALSE l%d\n", labelEND);
	codelist.push_back(code);

	LabelPair lp;
	lp.nBegin = labelBetwennCD;
	lp.nEnd = labelEND;
	gs_lpstk.push(lp);

	int i; // DO��������ĳ���
	int basep = start + offset;
	for (i = 0;;)
	{

		if ( (basep+i) >= words.size() )
		{
			fprintf(stderr, "%s %d: FORѭ����䲻����! \n",
				words[start+offset].m_yylval.srcfile.c_str(),
				words[start+offset].m_yylval.lineno);
			return -1;	
		}
		if ( words[basep+i].m_token == ENDFOR )
		{
			if (gs_lpstk.empty())
			{
				fprintf(stderr, "%s %d:FOR��ENDFOR��ƥ��! \n", 
					words[start].m_yylval.srcfile.c_str(),
					words[start].m_yylval.lineno);
				return -1;	
			}
			gs_lpstk.pop();

			sprintf(code, "LABEL l%d\n", labelBetwennCD);
			codelist.push_back(code);

			codelist.insert(codelist.end(), codeC.begin(), codeC.end());

			sprintf(code, "GOTO l%d\n", labelBEGIN);
			codelist.push_back(code);
			break;
		}

		int sttlen = ParseStatement(words, basep+i, codeD);
		if (sttlen == 0)
		{
			fprintf(stderr, "%s %d:FORѭ����䲻����! \n",
				words[start+offset].m_yylval.srcfile.c_str(),
				words[start+offset].m_yylval.lineno);
			return -1;	
		}
		if (sttlen < 0)
		{
			return -1;
		}
		codelist.insert(codelist.end(), codeD.begin(), codeD.end());

		i += sttlen;
	}
	offset += i;
	if ( words[start+offset  ].m_token == ENDFOR )
	{
			sprintf(code, "LABEL l%d\n", labelEND);
			codelist.push_back(code);

			++offset; // ENDFOR
			return offset;
	}
	fprintf(stderr, "%s %d:FORѭ����䲻����! \n",
				words[start+offset].m_yylval.srcfile.c_str(),
				words[start+offset].m_yylval.lineno);
	return -1;
}
/*
*����һ��break�䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
*/
static int BreakAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	char code[2048];

	if (words[start+1].m_token != ';')
	{
		fprintf(stderr, "%s %d:BREAK������Ҫһ���ֺ�! \n", 
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	if (gs_lpstk.empty())
	{
		fprintf(stderr, "%s %d:BREAK����ѭ�������! \n",
			words[start].m_yylval.srcfile.c_str(),
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
*����һ��continue�䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
*/
static int ContinueAnalyze(const vector<CToken> & words, unsigned int start,
	list<string> & codelist)
{
	codelist.clear();

	char code[2048];

	if (words[start+1].m_token != ';')
	{
		fprintf(stderr, "%s %d:CONTINUE������Ҫһ���ֺ�! \n", 
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}
	if (gs_lpstk.empty())
	{
		fprintf(stderr, "%s %d: CONTINUE����ѭ�������! \n",
			words[start].m_yylval.srcfile.c_str(),
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
*����һ��return�䣬�ɹ��������ĳ��ȣ�ʧ�ܷ��أ�1
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
		fprintf(stderr, "%s %d: RETURN���ĩβ��Ҫһ���ֺ�! \n",
			words[start].m_yylval.srcfile.c_str(),
			words[start].m_yylval.lineno);
		return -1;
	}

	sprintf(code, "GOTO l%d\n", gs_nModelEndLabel);
	codelist.push_back(code);
	return 1+nExprLen+1;
}

int lnb::GetLabel()
{
	static int LabelIndex = 0;
	return  LabelIndex++;
}
