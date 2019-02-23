%{
	static int yywrap();
	#include <stdio.h>
	#include <string.h>
	#include <stdlib.h>
	#include "xmllex.h"


%}


%s ELE_BEGIN ELE_VALUE
%option c++

%%
<INITIAL>[ \t]						{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ�������ո�\n");
									#endif
									}
\r\n								{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ���������з�\n");
									#endif
									}
\n									{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ���������з�\n");
									#endif
									}
<*>((\<!)|(\<\?))([^\>]|\r|\n)*\>	{/*����ע�ͺ��ĵ�˵��*/ 
									#if defined(_DEBUG)
									printf("�ʷ�ƥ������ע�ͺ��ĵ�˵��[%s]\n", yytext);
									#endif
									}
<INITIAL,ELE_VALUE>\<[a-zA-Z_][a-zA-Z0-9_]*			{	
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��'<id', ��ʼһ����ʼ����\n");
									#endif
										BEGIN(ELE_BEGIN);
										yylval.property = "";
										yylval.value = "";
										yylval.name = yytext + 1;
									}
<ELE_BEGIN>\/\> 					{
									/* <Password/> */
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��'<id/>'\n");
									#endif
										BEGIN(ELE_VALUE);
										return  XML_LEX_EMPNODE;
									}
<ELE_BEGIN>\>						{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��'>', �ر�һ����ʼ����\n");
									#endif
									BEGIN(ELE_VALUE);return XML_LEX_BEGIN;
									}
<ELE_BEGIN>[^\>]+[^\>\/]					{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��ڵ�����[%s]\n", yytext);
									#endif
									yylval.property = yytext; yylval.property.trim();
									}

<ELE_VALUE>\<\/[a-zA-Z_][a-zA-Z0-9_]*\>				{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��ڵ�رշ���\n");
									#endif
										AnsiString ss = yytext + 2;
										yylval.name = ss.substring(0, ss.length()-1);
										return XML_LEX_END;
									}
<ELE_VALUE>[^\<]+					{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��ڵ�ֵ[%s]\n", yytext);
									#endif
									yylval.value = yytext;
									return XML_LEX_VALUE;
									}

<*>.								{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��Ĭ��[%s]\n", yytext);
									#endif
									return yytext[0];
									}
%%

static int yywrap()
{
	return 1;
}

const YYVAL_TYPE* yyFlexLexer::__GetYYVal()
{
	return &yylval;
}
unsigned int yyFlexLexer::__GetLineNo()
{
	return g_lineno;
}
