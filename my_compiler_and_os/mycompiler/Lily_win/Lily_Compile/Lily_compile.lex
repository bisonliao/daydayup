%{
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "common.h"
#include "compile.yacc.h"

static char g_qstring_buf[QSTRING_BUF_MAX];
void lex_error(const char *msg);
int yylineno = 1;
char yyfilename[255];
%}

%s	s_qstring s_line

%%
<*>\n		{yylineno++;REJECT;}
<INITIAL>[ \t]+	{}
<INITIAL>\n		{}
	/*关键字*/
<INITIAL>if	{return IF;}
<INITIAL>then	{return THEN;}
<INITIAL>else	{return ELSE;}
<INITIAL>endif	{return ENDIF;}
<INITIAL>for	{return FOR;}
<INITIAL>do	{return DO;}
<INITIAL>endfor	{return ENDFOR;}
<INITIAL>while	{return WHILE;}
<INITIAL>endwhile	{return ENDWHILE;}
<INITIAL>begin	{return BEGIN_FLOW;}
<INITIAL>end	{return END_FLOW;}
<INITIAL>run	{return RUN;}
<INITIAL>int	{return INTEGER;}
<INITIAL>string	{return STRING;}
<INITIAL>float	{return FLOAT;}
<INITIAL>memblock	{return MEMBLOCK;}
<INITIAL>return {return RETURN;}
<INITIAL>continue	{return CONTINUE;}
<INITIAL>break	{return BREAK;}
<INITIAL>repeat	{return REPEAT;}
<INITIAL>until	{return UNTIL;}
<INITIAL>switch	{return SWITCH;}
<INITIAL>endswitch	{return ENDSWITCH;}
<INITIAL>case	{return CASE;}


	/*表示符和常量*/
<INITIAL>[a-zA-Z][a-zA-Z0-9_]*	{
			if (yyleng > (ID_MAX - 1))
			{
				lex_error("标识符太长!");	
			}
			strcpy(yylval.id_val, yytext);
			return ID;
		}
<INITIAL>(0)|([1-9][0-9]*)	{
				 yylval.const_int_val = atoi(yytext);
				 return CONST_INTEGER;
			}
<INITIAL>'.'				{
				 yylval.const_int_val = yytext[1];
				 return CONST_INTEGER;
			}
<INITIAL>(([1-9][0-9]*)|0)\.[0-9]+	{
				strcpy(yylval.const_float_val, yytext);
				return CONST_FLOAT;
			}

	/*常量字符串*/
<INITIAL>\"	{
		 BEGIN s_qstring; 
	         memset(g_qstring_buf, 0, QSTRING_BUF_MAX);
		 strcpy(g_qstring_buf, "\"");
		}
<s_qstring>\\\" {
		 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
		 {
		 	lex_error("程序中常量字符串长度太大!");
		 }
		 strcat(g_qstring_buf, "\"");
		}	
<s_qstring>\\[ \t]*\n	{}
<s_qstring>\n	{lex_error("常量字符串首尾双引号不匹配!");}
<s_qstring>\"	{
		 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
		 {
		 	lex_error("程序中常量字符串长度太大!");
		 }
		 strcat(g_qstring_buf, yytext);
		 BEGIN INITIAL;
		 strcpy(yylval.const_string_val, g_qstring_buf);
		 return CONST_STRING;
		}
<s_qstring>[^\"\n]{1}	{
			 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
			 {
			 	lex_error("程序中常量字符串长度太大!");
			 }
			 strcat(g_qstring_buf, yytext);
		}
	/*一般符号*/
<INITIAL>"("	{return yytext[0];}
<INITIAL>")"	{return yytext[0];}
<INITIAL>"%"	{return yytext[0];}
<INITIAL>"*"	{return yytext[0];}
<INITIAL>"/"	{return yytext[0];}
<INITIAL>"+"	{return yytext[0];}
<INITIAL>"-"	{return yytext[0];}
<INITIAL>";"	{return yytext[0];}
<INITIAL>":"	{return yytext[0];}
<INITIAL>"="	{return yytext[0];}
<INITIAL>","	{return yytext[0];}
<INITIAL>"["	{return yytext[0];}
<INITIAL>"]"	{return yytext[0];}

<INITIAL>"=="	{return EQ;}
<INITIAL>"<="	{return LE;}
<INITIAL>"<"	{return LT;}
<INITIAL>">="	{return GE;}
<INITIAL>">"	{return GT;}
<INITIAL>"!="	{return NE;}
<INITIAL>"&&"	{return AND;}
<INITIAL>"||"	{return OR;}
<INITIAL>"!"	{return NOT;}

	/*	#line lineno filename 指令的解释*/
<INITIAL>"#line"	{BEGIN s_line;}
<s_line>[ \t]+		{}
<s_line>[0-9].*\n	{
		sscanf(yytext, "%d %s", &yylineno, yyfilename);
		BEGIN INITIAL;
	}
		
	/*其他*/
<*>.		{lex_error("unexpected character!");}

%%
void lex_error(const char *msg)
{
	fprintf(stderr, "%s, filename=[%s], lineno=[%d]\n",
		msg,
		yyfilename,
		yylineno);
	fflush(stderr);
	fflush(stdout);
	exit(-1);
}
const char * getyytext()
{
	return yytext;
}
/*
int main(int argc, char**argv)
{
	int retcode;
	if (argc < 2)
	{
		return -1;
	}
	if ( (yyin = fopen(argv[1], "rb")) == NULL)
	{
		fprintf(stderr, "打开文件[%s]失败!\n", argv[1]);
	}
	yyout = stdout;
	while ( (retcode = yylex()) != 0)
	{
		printf("[%d]\n", retcode);
		if (retcode == CONST_STRING)
		{
			printf("[%s]\n", g_qstring_buf);
		}
	}
	return 0;
}
*/
