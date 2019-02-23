// AnalyseTable.cpp: implementation of the CAnalyseTable class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "AnalyseTable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CAnalyseTable::CAnalyseTable()
{
	this->m_nActionCount = 0;
	this->m_nGotoCount = 0;
	this->m_action = new ACTION[ACTION_MAX];
	if (m_action == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	this->m_goto = new GOTO[GOTO_MAX];
	if (m_goto == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
}

CAnalyseTable::~CAnalyseTable()
{

	if (m_action != NULL)
	{
		delete[] m_action;
	}
	if (m_goto != NULL)
	{
		delete[] m_goto;
	}

}

void CAnalyseTable::AddAction(const ACTION &ele)
{
	if (m_nActionCount >= ACTION_MAX)
	{
		fprintf(stderr, "[%s][%d]ACTION数组开辟的空间不够!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	m_action[m_nActionCount++] = ele;
	printf("Add a action %d\n", m_nActionCount);
}

void CAnalyseTable::AddGoto(const GOTO &ele)
{
	if (m_nGotoCount >= GOTO_MAX)
	{
		fprintf(stderr, "[%s][%d]GOTO数组开辟的空间不够!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	m_goto[m_nGotoCount++] = ele;
	printf("Add a goto %d\n", m_nGotoCount);
}

int CAnalyseTable::GetAction(ACTION &ele)
{
	for (int i = 0; i < m_nActionCount; i++)
	{
		if (m_action[i].state == ele.state &&
			m_action[i].terminator == ele.terminator)
		{
			strcpy(ele.action, m_action[i].action);
			return 0;
		}
	}
	return -1;
}

int CAnalyseTable::GetGoto(GOTO &ele)
{
	for (int i = 0; i < m_nGotoCount; i++)
	{
		if (m_goto[i].state == ele.state &&
			m_goto[i].nonterminator == ele.nonterminator)
		{
			ele.gotostate = m_goto[i].gotostate;
			return 0;
		}
	}
	return -1;
}

void CAnalyseTable::clear()
{	
	this->m_nActionCount = 0;
	this->m_nGotoCount = 0;
}

CAnalyseTable::CAnalyseTable(const CAnalyseTable &another)
{
	this->m_nActionCount = another.m_nActionCount;
	this->m_nGotoCount = another.m_nGotoCount;
	this->m_action = new ACTION[ACTION_MAX];
	if (m_action == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	this->m_goto = new GOTO[GOTO_MAX];
	if (m_goto == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	int i;
	for (i = 0; i < this->m_nActionCount; ++i)
	{
		m_action[i] = another.m_action[i];
	}
	for (i = 0; i < this->m_nGotoCount; ++i)
	{
		m_goto[i] = another.m_goto[i];
	}

}

const CAnalyseTable & CAnalyseTable::operator =(const CAnalyseTable &another)
{
	
	
	
	this->m_nActionCount = another.m_nActionCount;
	this->m_nGotoCount = another.m_nGotoCount;
	

	int i;
	for (i = 0; i < this->m_nActionCount; ++i)
	{
		m_action[i] = another.m_action[i];
	}
	for (i = 0; i < this->m_nGotoCount; ++i)
	{
		m_goto[i] = another.m_goto[i];
	}

	return *this;
}

int CAnalyseTable::WriteToFile(const char *filename)
{
	assert(NULL != filename);

	FILE * fp = NULL;
	if ( (fp = fopen(filename, "wb")) == NULL)
	{
		return -1;
	}
	int i;
	for (i = 0; i < m_nActionCount; i++)
	{
		fprintf(fp, "ACTION    %-10d%-100s%-100s\n",
				m_action[i].state,
				m_action[i].terminator.ToString().c_str(),
				m_action[i].action);
	}
	for (i = 0; i < m_nGotoCount; i++)
	{
		fprintf(fp, "GOTO      %-10d%-100s%-10d\n",
				m_goto[i].state,
				m_goto[i].nonterminator.ToString().c_str(),
				m_goto[i].gotostate);
	}
	fclose(fp);
	return 0;
}
/*去除字符串右侧的空格和tab键*/
void rtrim(char *p)
{
       int len;

       if (NULL == p)
       {
               return;
       }

       len = strlen(p);
       len--;
       while ((p[len] == ' ' || p[len] == '\t')&&
               len >= 0)
       {
               len--;
       }
       p[len + 1] = 0;
}
int CAnalyseTable::ReadFrmFile(const char *filename)
{
	assert(NULL != filename);

	FILE * fp = NULL;
	if ( (fp = fopen(filename, "rb")) == NULL)
	{
		return -1;
	}

	m_nActionCount = 0;
	m_nGotoCount = 0;
	char buf[1024];
	while ( (fgets(buf, sizeof(buf), fp)) != NULL)
	{
		if (buf[strlen(buf) - 1] == '\n')
		{
			buf[strlen(buf) - 1] = 0;
		}

		if ( memcmp(buf, "ACTION", 6) == 0)
		{
			if (strlen(buf) != 220)
			{
				fprintf(stderr, "[%s][%d]分析表文件格式不正确!\n",
					__FILE__,
					__LINE__);
				exit(-1);
			}
			if (m_nActionCount >= ACTION_MAX)
			{
				fprintf(stderr, "[%s][%d]分析表数组开辟的空间不够!\n",
					__FILE__,
					__LINE__);
				exit(-1);
			}

			char tmp[200];
			
			memset(tmp, 0, sizeof(tmp));
			memcpy(tmp, buf + 10, 10);
			rtrim(tmp);
			m_action[m_nActionCount].state = atoi(tmp);

			memset(tmp, 0, sizeof(tmp));
			memcpy(tmp, buf + 20, 100);
			rtrim(tmp);
			m_action[m_nActionCount].terminator = CTerminator(tmp);

			memset(tmp, 0, sizeof(tmp));
			memcpy(tmp, buf + 120, 100);
			rtrim(tmp);
			strcpy(m_action[m_nActionCount].action, tmp);	
			m_nActionCount++;
		}
		else
		{
			if (strlen(buf) != 130)
			{
				fprintf(stderr, "[%s][%d]分析表文件格式不正确!\n",
					__FILE__,
					__LINE__);
				exit(-1);
			}
			char tmp[200];
			if (m_nGotoCount >= GOTO_MAX)
			{
				fprintf(stderr, "[%s][%d]分析表数组开辟的空间不够!\n",
					__FILE__,
					__LINE__);
				exit(-1);
			}
			memset(tmp, 0, sizeof(tmp));
			memcpy(tmp, buf + 10, 10);
			rtrim(tmp);
			m_goto[m_nGotoCount].state = atoi(tmp);

			memset(tmp, 0, sizeof(tmp));
			memcpy(tmp, buf + 20, 100);
			rtrim(tmp);
			m_goto[m_nGotoCount].nonterminator = CNonTerminator(tmp);

			memset(tmp, 0, sizeof(tmp));
			memcpy(tmp, buf + 120, 10);
			rtrim(tmp);
			m_goto[m_nGotoCount].gotostate = atoi(tmp);
			m_nGotoCount++;
		}
	}
	fclose(fp);
	return 0;
}
