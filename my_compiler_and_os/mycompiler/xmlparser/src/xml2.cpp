#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream.h>
#include <string.h>

#undef yyFlexLexer 
#define yyFlexLexer zzFlexLexer
#include "lex2/FlexLexer.h"
#undef yyFlexLexer

#include "xml.h"
#include "xmllex2.h"

#if defined(_WIN32_)
	#define snprintf _snprintf
#endif

#include <stack>

int Xml::ReadFrmBuffer(const char * buffer, unsigned short buffersize,char *errmsg, unsigned short errmsgsize)
{
	this->Clear();
	m_RootPtr = NULL;
	XmlNode* CurPtr = NULL;
	XmlNode* newptr = NULL;


	zzFlexLexer lexer;
	lexer.g_lineno = 1;
	lexer.g_inputbuf = NULL;
	lexer.__InitReadFrmBuffer(buffer, buffersize);

	int retcode;
	int nLevel = 0; /*��Σ����ڵ�Ϊ��1��*/

	stack<XmlNode*> stk;
	stk.push(NULL);

    while ( (retcode = lexer.yylex()) != 0)
	{
		/*ȡ�õ�������*/
		const YYVAL_TYPE *yylval = lexer.__GetYYVal2();


		switch(retcode)
		{
		/*
			case XML_LEX_HEADER:
					if (nLevel != 0)
					{
						snprintf(errmsg, errmsgsize, "�ļ�ͷ<?xml...?>���ֵ�λ�ò���ȷ!");
						return -1;
					}
					break;
		*/
			case XML_LEX_BEGIN:
					++nLevel;
					/*�½�һ�ڵ�*/
					newptr = new XmlNode();
					if (newptr == NULL)
					{
						snprintf(errmsg, errmsgsize, "�ڴ����ʧ��!");
						return -1;
					}
					if (nLevel == 1)
					{
						m_RootPtr = newptr;
					}
					newptr->SetNodeName(yylval->name);	
					newptr->SetNodeProperty(yylval->property);
					newptr->SetNodeValue("");	
					newptr->SetParent(CurPtr);

					if (CurPtr != NULL)
					{
						CurPtr->SetNodeValue("");	/*�������ӽڵ�Ľڵ㣬��ֵΪ��*/
						CurPtr->AddChild(newptr);
					}
					stk.push(newptr);
					/*���µ�ǰ�ڵ�ָ��*/
					CurPtr = newptr;

					break;
			case XML_LEX_EMPNODE:
					++nLevel;
					/*�½�һ�ڵ�*/
					newptr = new XmlNode();
					if (newptr == NULL)
					{
						snprintf(errmsg, errmsgsize, "�ڴ����ʧ��!");
						return -1;
					}
					if (nLevel == 1)
					{
						m_RootPtr = newptr;
					}
					newptr->SetNodeName(yylval->name);	
					newptr->SetNodeProperty(yylval->property);
					newptr->SetNodeValue("");	
					newptr->SetParent(CurPtr);

					if (CurPtr != NULL)
					{
						CurPtr->SetNodeValue("");	/*�������ӽڵ�Ľڵ㣬��ֵΪ��*/
						CurPtr->AddChild(newptr);
					}

					nLevel--;
					if (nLevel == 0)
					{
						return 0;
					}
					break;
			case XML_LEX_END:
					if ( !(CurPtr->GetNodeName()==AnsiString(yylval->name)) )
					{
						snprintf(errmsg, errmsgsize, "�ֶεĿ�ʼ���źͽ������Ų�ƥ��!");
						return -1;
					}

					--nLevel;
					if (nLevel < 0 || stk.size() == 0)
					{
						snprintf(errmsg, errmsgsize, "�ֶεĿ�ʼ���źͽ������Ų�ƥ��!");
						return -1;
					}
					/*���µ�ǰ�ڵ�ָ��*/
					stk.pop();
					CurPtr = stk.top();

					if (nLevel == 0)
					{
						return 0;
					}

					break;
			case XML_LEX_VALUE:
					if (CurPtr != NULL)
					{
						CurPtr->SetNodeValue(yylval->value);
					}
					else
					{
						snprintf(errmsg, errmsgsize, "�ļ���ʽ����ȷ!");
						return -1;
					}
					break;
			default:
					snprintf(errmsg, errmsgsize, "�ļ���ʽ����ȷ!");
					return -1;
					break;
		}
	}
	if (nLevel != 0)
	{
		snprintf(errmsg, errmsgsize, "�ļ���ʽ����ȷ!");
		return -1;
	}
	return 0;
}
