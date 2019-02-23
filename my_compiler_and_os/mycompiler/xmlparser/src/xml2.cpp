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
	int nLevel = 0; /*层次，根节点为第1层*/

	stack<XmlNode*> stk;
	stk.push(NULL);

    while ( (retcode = lexer.yylex()) != 0)
	{
		/*取得单词属性*/
		const YYVAL_TYPE *yylval = lexer.__GetYYVal2();


		switch(retcode)
		{
		/*
			case XML_LEX_HEADER:
					if (nLevel != 0)
					{
						snprintf(errmsg, errmsgsize, "文件头<?xml...?>出现的位置不正确!");
						return -1;
					}
					break;
		*/
			case XML_LEX_BEGIN:
					++nLevel;
					/*新建一节点*/
					newptr = new XmlNode();
					if (newptr == NULL)
					{
						snprintf(errmsg, errmsgsize, "内存分配失败!");
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
						CurPtr->SetNodeValue("");	/*对于有子节点的节点，其值为空*/
						CurPtr->AddChild(newptr);
					}
					stk.push(newptr);
					/*更新当前节点指针*/
					CurPtr = newptr;

					break;
			case XML_LEX_EMPNODE:
					++nLevel;
					/*新建一节点*/
					newptr = new XmlNode();
					if (newptr == NULL)
					{
						snprintf(errmsg, errmsgsize, "内存分配失败!");
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
						CurPtr->SetNodeValue("");	/*对于有子节点的节点，其值为空*/
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
						snprintf(errmsg, errmsgsize, "字段的开始符号和结束符号不匹配!");
						return -1;
					}

					--nLevel;
					if (nLevel < 0 || stk.size() == 0)
					{
						snprintf(errmsg, errmsgsize, "字段的开始符号和结束符号不匹配!");
						return -1;
					}
					/*更新当前节点指针*/
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
						snprintf(errmsg, errmsgsize, "文件格式不正确!");
						return -1;
					}
					break;
			default:
					snprintf(errmsg, errmsgsize, "文件格式不正确!");
					return -1;
					break;
		}
	}
	if (nLevel != 0)
	{
		snprintf(errmsg, errmsgsize, "文件格式不正确!");
		return -1;
	}
	return 0;
}
