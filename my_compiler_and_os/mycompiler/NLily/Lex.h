// Lex.h: interface for the CLex class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_LEX_H__D9F5A4B8_F8E9_4B85_BDC6_5B5359F3FD94__INCLUDED_)
#define AFX_LEX_H__D9F5A4B8_F8E9_4B85_BDC6_5B5359F3FD94__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Terminator_Index.h"

/*
*´Ê·¨·ÖÎö
*/
class CLex  
{
private:
	static void lexerror(const char * msg);
	enum{TEXT_MAX = 1024};
	CLex();
	virtual ~CLex();
	static FILE * m_yyin;
	static unsigned int m_lineno;
	static char m_yytext[TEXT_MAX];
	static int m_yylength;
	static unsigned int m_position;
public:
	static void rollback();
	static int getlineno();
	static const char * getyytext();
	static int init_lex(const char* filename);
	static int lex();


};

#endif // !defined(AFX_LEX_H__D9F5A4B8_F8E9_4B85_BDC6_5B5359F3FD94__INCLUDED_)
