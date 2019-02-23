%{
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "token.h"
#include <stdio.h>

using lnb::YYLVAL;

namespace lnb {
	int yylex(YYLVAL & yylval);
};



#define YY_DECL int lnb::yylex(YYLVAL & yylval)

#define QSTRING_BUF_MAX 1024
#define ID_MAX 32


static char g_qstring_buf[QSTRING_BUF_MAX];
void lex_error(const char *msg, ...);
static int yylineno = 1;
static string yysrcfile = "";
%}

%s	s_qstring s_line s_comment

%%
<INITIAL>^@line=[0-9]+:.*$ 	{
				char tmpbuf[258];
				tmpbuf[0] = '\0';
				if (2 != sscanf(yytext+6, "%d:%s", &yylineno, tmpbuf))
				{
					lex_error("@line����ָ��Ƿ�!%s\n", yytext);
				}
				--yylineno;
				yysrcfile = tmpbuf;
			}
<*>\n		{ yylineno++; REJECT; }
<INITIAL>[ \t]+	{}
<INITIAL>#.*$   {/*ע��*/}
<INITIAL>\/\/.*$   {/*ע��*/}
<INITIAL>\/\*	   { BEGIN(s_comment); }
<INITIAL>\n		{}
<INITIAL>\r\n		{}
	/*�ؼ���*/
<INITIAL>if	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return IF;}
<INITIAL>then	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return THEN;}
<INITIAL>else	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ELSE;}
<INITIAL>elsif	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ELSIF;}
<INITIAL>endif	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ENDIF;}
<INITIAL>while	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return WHILE;}
<INITIAL>do	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return DO;}
<INITIAL>endwhile	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ENDWHILE;}
<INITIAL>begin	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return BEGIN_SCRIPT;}
<INITIAL>end	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return END_SCRIPT;}
<INITIAL>return {yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return RETURN;}
<INITIAL>continue	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return CONTINUE;}
<INITIAL>break	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return BREAK;}
<INITIAL>for	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return FOR;}
<INITIAL>endfor	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ENDFOR;}

	/*���ע��*/
<s_comment>.|\r|\n {/*����*/}
<s_comment>"\*\/"		  { BEGIN(INITIAL); }

	/*����/����*/
<INITIAL>function	{
			yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return FUNCTION;
		}

	/*��ʾ���������ͳ���*/
<INITIAL>[a-zA-Z][a-zA-Z0-9_]*	{
			if (yyleng > (ID_MAX - 1))
			{
				lex_error("��ʶ��̫��!");	
			}
			yylval.id_val = yytext;
			yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; 
			return ID;
		}
	/*ȫ�ֱ���*/
<INITIAL>\$\$[a-zA-Z0-9_]+	{
			if (yyleng > (ID_MAX - 1))
			{
				lex_error("������̫��!");	
			}
			yylval.id_val = yytext;
			yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; 
			return VAR;
		}
	/*�ֲ�����*/
<INITIAL>\$[a-zA-Z0-9_]+	{
			if (yyleng > (ID_MAX - 1))
			{
				lex_error("������̫��!");	
			}
			yylval.id_val = yytext;
			yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; 
			return VAR;
		}
<INITIAL>(0+)|([1-9][0-9]*)	{
				 yylval.int_val = atoll(yytext);
				 yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; 
				 return CONST_INT;
			}
<INITIAL>[0-9]+\.[0-9]+	{
				 yylval.float_val = atof(yytext);
				 yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; 
				 return CONST_FLOAT;
			}
<INITIAL>(([1-9][0-9]*)|0)\.[0-9]+	{
				yylval.float_val = atof(yytext);
				yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; 
				return CONST_FLOAT;
			}

	/*�����ַ���*/
<INITIAL>\"	{
		 BEGIN s_qstring; 
	         memset(g_qstring_buf, 0, QSTRING_BUF_MAX);
		 strcpy(g_qstring_buf, "\"");
		}
<s_qstring>\\\" {
		 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
		 {
		 	lex_error("�����г����ַ�������̫��!");
		 }
		 strcat(g_qstring_buf, "\\\"");
		}	
<s_qstring>\\[ \t]*\n	{}
<s_qstring>\\[ \t]*\r\n	{}
<s_qstring>\n	{lex_error("�����ַ�����β˫���Ų�ƥ��!");}
<s_qstring>\"	{
		 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
		 {
		 	lex_error("�����г����ַ�������̫��!");
		 }
		 strcat(g_qstring_buf, yytext);
		 BEGIN INITIAL;
		 yylval.string_val =  g_qstring_buf;
		 yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return CONST_STRING;
		}
<s_qstring>[^\"\n]{1}	{
			 if ( (QSTRING_BUF_MAX - 1 - strlen(g_qstring_buf)) < yyleng)
			 {
			 	lex_error("�����г����ַ�������̫��!");
			 }
			 strcat(g_qstring_buf, yytext);
		}
	/*һ�����*/
<INITIAL>"("	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return LBRK;}
<INITIAL>")"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return RBRK;}
<INITIAL>"*"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return MUL;}
<INITIAL>"/"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return DIV;}
<INITIAL>"+"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ADD;}
<INITIAL>"-"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return SUB;}
<INITIAL>";"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return yytext[0];}
<INITIAL>"="	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return yytext[0];}
<INITIAL>">"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return GT;}
<INITIAL>">="	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return GE;}
<INITIAL>"<="	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return LE;}
<INITIAL>"<"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return LT;}
<INITIAL>"=="	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return EQ;}
<INITIAL>"!="	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return NE;}
<INITIAL>"!"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return NOT;}
<INITIAL>","	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return COMMA;}
<INITIAL>"["	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return LOFFSET;}
<INITIAL>"]"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return ROFFSET;}
<INITIAL>"{"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return LMAP;}
<INITIAL>"}"	{yylval.lineno=yylineno; yylval.srcfile=yysrcfile;; return RMAP;}


		
	/*����*/
<*>.		{lex_error("unexpected character=[%c][%d]", yytext[0], yytext[0]);}

%%
void lex_error(const char *msg, ...)
{
	fprintf(stderr, "%s %d:\t",  yysrcfile.c_str(), yylineno);
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
