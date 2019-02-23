%{
	static int yywrap();
	#include <stdio.h>
	#include <string.h>
	#include <stdlib.h>
	#include "xmllex2.h"

   #undef YY_INPUT
   #define YY_INPUT(b, r, ms) (r = my_yyinput(b,ms))
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
<INITIAL,ELE_VALUE>\<[a-zA-Z_][a-zA-Z0-9_]* {
									#if defined(_DEBUG)
									printf("�ʷ�ƥ��'<id'\n");
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
									printf("�ʷ�ƥ��'>', �رտ�ʼ����\n");
									#endif
									BEGIN(ELE_VALUE);
									return XML_LEX_BEGIN;
									}
<ELE_BEGIN>[^\>]+[^\>\/]					{
									#if defined(_DEBUG)
									printf("�ʷ�ƥ������[%s]\n", yytext);
									#endif
										yylval.property = yytext; yylval.property.trim();
									}

<ELE_VALUE>\<\/[a-zA-Z_][a-zA-Z0-9_]*\>	{
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
									printf("�ʷ�ƥ��Ĭ��ֵ[%s]\n", yytext);
									#endif
									return yytext[0];
									}
%%

static int yywrap()
{
	return 1;
}

const YYVAL_TYPE* yyFlexLexer::__GetYYVal2()
{
	return &yylval;
}
unsigned int yyFlexLexer::__GetLineNo2()
{
	return g_lineno;
}
int yyFlexLexer::my_yyinput(char *buf, int max_size)
{
    int n = ( g_inputlimit - g_inputptr);
    if (n > max_size)
    {
        n = max_size;
    }
    if (n > 0)
    {
        memcpy(buf, g_inputbuf + g_inputptr, n);
        g_inputptr += n;
    }
    return n;
}
int yyFlexLexer::__InitReadFrmBuffer(const char *buf, int buflen)
{
    if (buflen <= 0)
    {
        return -1;
    }
	g_inputstring = buf;
	g_inputbuf = g_inputstring.c_str();
    g_inputlimit = buflen;
    g_inputptr = 0;
    return 0;
}
