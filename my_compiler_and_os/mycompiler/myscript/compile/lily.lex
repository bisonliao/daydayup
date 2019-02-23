%{
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "token.h"
#include <stdio.h>

#define YY_DECL int yylex(YYLVAL & yylval)

#define QSTRING_BUF_MAX 1024
#define ID_MAX 32


static char g_qstring_buf[QSTRING_BUF_MAX];
void lex_error(const char *msg, ...);
int yylineno = 1;
%}

%s	s_qstring s_line

%%
<*>\n		{yylineno++;REJECT;}
<INITIAL>[ \t]+	{}
<INITIAL>#.*$   {/*注释*/}
<INITIAL>\n		{}
<INITIAL>\r\n		{}
	/*关键字*/
<INITIAL>if	{yylval.lineno=yylineno; return IF;}
<INITIAL>then	{yylval.lineno=yylineno; return THEN;}
<INITIAL>else	{yylval.lineno=yylineno; return ELSE;}
<INITIAL>endif	{yylval.lineno=yylineno; return ENDIF;}
<INITIAL>while	{yylval.lineno=yylineno; return WHILE;}
<INITIAL>do	{yylval.lineno=yylineno; return DO;}
<INITIAL>endwhile	{yylval.lineno=yylineno; return ENDWHILE;}
<INITIAL>begin	{yylval.lineno=yylineno; return BEGIN_SCRIPT;}
<INITIAL>end	{yylval.lineno=yylineno; return END_SCRIPT;}
<INITIAL>return {yylval.lineno=yylineno; return RETURN;}
<INITIAL>continue	{yylval.lineno=yylineno; return CONTINUE;}
<INITIAL>break	{yylval.lineno=yylineno; return BREAK;}

	/*流程/函数*/
<INITIAL>function	{
			yylval.lineno=yylineno; return FUNCTION;
		}

	/*表示符、变量和常量*/
<INITIAL>[a-zA-Z][a-zA-Z0-9_]*	{
			if (yyleng > (ID_MAX - 1))
			{
				lex_error("标识符太长!");	
			}
			yylval.id_val = yytext;
			yylval.lineno=yylineno; 
			return ID;
		}
<INITIAL>\$[a-zA-Z0-9_]+	{
			if (yyleng > (ID_MAX - 1))
			{
				lex_error("变量名太长!");	
			}
			yylval.id_val = yytext;
			yylval.lineno=yylineno; 
			return VAR;
		}
<INITIAL>(0+)|([1-9][0-9]*)	{
				 yylval.int_val = atoi(yytext);
				 yylval.lineno=yylineno; 
				 return CONST_INT;
			}
<INITIAL>[0-9]+\.[0-9]+	{
				 yylval.float_val = atof(yytext);
				 yylval.lineno=yylineno; 
				 return CONST_FLOAT;
			}
<INITIAL>(([1-9][0-9]*)|0)\.[0-9]+	{
				yylval.float_val = atof(yytext);
				yylval.lineno=yylineno; 
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
<s_qstring>\\[ \t]*\r\n	{}
<s_qstring>\n	{lex_error("常量字符串首尾双引号不匹配!");}
<s_qstring>\"	{
		 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
		 {
		 	lex_error("程序中常量字符串长度太大!");
		 }
		 strcat(g_qstring_buf, yytext);
		 BEGIN INITIAL;
		 yylval.string_val =  g_qstring_buf;
		 yylval.lineno=yylineno; return CONST_STRING;
		}
<s_qstring>[^\"\n]{1}	{
			 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
			 {
			 	lex_error("程序中常量字符串长度太大!");
			 }
			 strcat(g_qstring_buf, yytext);
		}
	/*一般符号*/
<INITIAL>"("	{yylval.lineno=yylineno; return LBRK;}
<INITIAL>")"	{yylval.lineno=yylineno; return RBRK;}
<INITIAL>"*"	{yylval.lineno=yylineno; return MUL;}
<INITIAL>"/"	{yylval.lineno=yylineno; return DIV;}
<INITIAL>"+"	{yylval.lineno=yylineno; return ADD;}
<INITIAL>"-"	{yylval.lineno=yylineno; return SUB;}
<INITIAL>";"	{yylval.lineno=yylineno; return yytext[0];}
<INITIAL>"="	{yylval.lineno=yylineno; return yytext[0];}
<INITIAL>">"	{yylval.lineno=yylineno; return GT;}
<INITIAL>">="	{yylval.lineno=yylineno; return GE;}
<INITIAL>"<="	{yylval.lineno=yylineno; return LE;}
<INITIAL>"<"	{yylval.lineno=yylineno; return LT;}
<INITIAL>"=="	{yylval.lineno=yylineno; return EQ;}
<INITIAL>"!="	{yylval.lineno=yylineno; return NE;}
<INITIAL>"!"	{yylval.lineno=yylineno; return NOT;}
<INITIAL>","	{yylval.lineno=yylineno; return COMMA;}
<INITIAL>"["	{yylval.lineno=yylineno; return LOFFSET;}
<INITIAL>"]"	{yylval.lineno=yylineno; return ROFFSET;}


		
	/*其他*/
<*>.		{lex_error("unexpected character=[%c][%d]", yytext[0], yytext[0]);}

%%
void lex_error(const char *msg, ...)
{
	fprintf(stderr, "lineno=[%d]:\t",  yylineno);
	va_list vvv;
	va_start(vvv, msg);
	vfprintf(stderr, msg, vvv);
	fprintf(stderr, "\n");
	va_end(vvv);
	fflush(stderr);
	exit(-1);
}
const char * getyytext()
{
	return yytext;
}
int yywrap()
{
	return 1;
}
