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
									printf("词法匹配跳过空格\n");
									#endif
									}
\r\n								{
									#if defined(_DEBUG)
									printf("词法匹配跳过换行符\n");
									#endif
									}
\n									{
									#if defined(_DEBUG)
									printf("词法匹配跳过换行符\n");
									#endif
									}
<*>((\<!)|(\<\?))([^\>]|\r|\n)*\>	{/*跳过注释和文档说明*/ 
									#if defined(_DEBUG)
									printf("词法匹配跳过注释和文档说明[%s]\n", yytext);
									#endif
									}
<INITIAL,ELE_VALUE>\<[a-zA-Z_][a-zA-Z0-9_]*			{	
									#if defined(_DEBUG)
									printf("词法匹配'<id', 开始一个开始符号\n");
									#endif
										BEGIN(ELE_BEGIN);
										yylval.property = "";
										yylval.value = "";
										yylval.name = yytext + 1;
									}
<ELE_BEGIN>\/\> 					{
									/* <Password/> */
									#if defined(_DEBUG)
									printf("词法匹配'<id/>'\n");
									#endif
										BEGIN(ELE_VALUE);
										return  XML_LEX_EMPNODE;
									}
<ELE_BEGIN>\>						{
									#if defined(_DEBUG)
									printf("词法匹配'>', 关闭一个开始符号\n");
									#endif
									BEGIN(ELE_VALUE);return XML_LEX_BEGIN;
									}
<ELE_BEGIN>[^\>]+[^\>\/]					{
									#if defined(_DEBUG)
									printf("词法匹配节点属性[%s]\n", yytext);
									#endif
									yylval.property = yytext; yylval.property.trim();
									}

<ELE_VALUE>\<\/[a-zA-Z_][a-zA-Z0-9_]*\>				{
									#if defined(_DEBUG)
									printf("词法匹配节点关闭符号\n");
									#endif
										AnsiString ss = yytext + 2;
										yylval.name = ss.substring(0, ss.length()-1);
										return XML_LEX_END;
									}
<ELE_VALUE>[^\<]+					{
									#if defined(_DEBUG)
									printf("词法匹配节点值[%s]\n", yytext);
									#endif
									yylval.value = yytext;
									return XML_LEX_VALUE;
									}

<*>.								{
									#if defined(_DEBUG)
									printf("词法匹配默认[%s]\n", yytext);
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
