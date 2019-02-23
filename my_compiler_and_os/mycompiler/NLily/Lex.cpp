// Lex.cpp: implementation of the CLex class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Lex.h"
#include <ctype.h>
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

FILE * CLex::m_yyin = NULL;
unsigned int CLex::m_lineno = 1;
char CLex::m_yytext[TEXT_MAX];
int CLex::m_yylength;
unsigned int CLex::m_position;

CLex::CLex()
{

}

CLex::~CLex()
{

}
int CLex::init_lex(const char* filename)
{
	m_position = 0;
	m_lineno = 1;
	memset(m_yytext, 0, sizeof(TEXT_MAX));
	m_yylength = 0;
	if (filename == NULL)
	{
		m_yyin = 0;
		return 0;
	}
	if ( (m_yyin = fopen(filename, "rb")) == NULL)
	{
		return -1;
	}
	return 0;
}
int CLex::lex()
{
	int state = 0;
	int chr;

	while (1)
	{
		chr = fgetc(m_yyin);
		switch (state)
		{
		case 0:
			//记录下记号开始的位置，利于回退
			m_position = ftell(m_yyin);
			memset(m_yytext, 0, TEXT_MAX);
			m_yylength = 0;

			if (chr == ' ' || chr == '\t' || chr == '\r')
			{
				break;
			}
			else if (chr == '\n')
			{
				m_lineno++;
				break;
			}
			else if (isdigit(chr))
			{
				state = 1;
				m_yytext[m_yylength++] = chr;
				break;
			}
			else if (isalpha(chr))
			{
				state = 2;
				m_yytext[m_yylength++] = chr;
				break;
			}	
			else if (chr == '"')
			{
				state = 3;
				break;
			}
			else if (chr == EOF)
			{
				return 0;
			}
			else if (chr == '(')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_L_BRACKET;
			}
			else if (chr == ')')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_R_BRACKET;
			}
			else if (chr == '%')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_MOD;
			}
			else if (chr == '*')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_MUL;
			}
			else if (chr == '/')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_DIV;
			}
			else if (chr == '+')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_ADD;
			}
			else if (chr == '-')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_SUB;
			}
			else if (chr == ';')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_SEMI;
			}
			else if (chr == ',')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_COMMA;
			}
			else if (chr == ':')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_COLON;
			}
			else if (chr == '=')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '=')
				{
					m_yytext[m_yylength++] = chr;
					m_yytext[m_yylength++] = nextchr;
					return IDX_EQ;
				}
				else 
				{
					fseek(m_yyin, -1, SEEK_CUR);
					m_yytext[m_yylength++] = chr;
					return IDX_ASSIGN;
				}
			}
			else if (chr == '[')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_L_SQ_BRACKET;
			}
			else if (chr == ']')
			{
				m_yytext[m_yylength++] = chr;
				return IDX_R_SQ_BRACKET;
			}
			else if (chr == '!')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '=')
				{
					m_yytext[m_yylength++] = chr;
					m_yytext[m_yylength++] = nextchr;
					return IDX_NE;
				}
				else 
				{
					fseek(m_yyin, -1, SEEK_CUR);
					m_yytext[m_yylength++] = chr;
					return IDX_NOT;
				}				
			}
			else if (chr == '<')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '=')
				{
					m_yytext[m_yylength++] = chr;
					m_yytext[m_yylength++] = nextchr;
					return IDX_LE;
				}
				else 
				{
					fseek(m_yyin, -1, SEEK_CUR);
					m_yytext[m_yylength++] = chr;
					return IDX_LT;
				}	
			}
			else if (chr == '>')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '=')
				{
					m_yytext[m_yylength++] = chr;
					m_yytext[m_yylength++] = nextchr;
					return IDX_GE;
				}
				else 
				{
					fseek(m_yyin, -1, SEEK_CUR);
					m_yytext[m_yylength++] = chr;
					return IDX_GT;
				}	
			}
			else if (chr == '&')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '&')
				{
					m_yytext[m_yylength++] = chr;
					m_yytext[m_yylength++] = nextchr;
					return IDX_AND;
				}
				else 
				{
					fseek(m_yyin, -1, SEEK_CUR);
					m_yytext[m_yylength++] = chr;
					lexerror("");
				}
			}	
			else if (chr == '|')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '|')
				{
					m_yytext[m_yylength++] = chr;
					m_yytext[m_yylength++] = nextchr;
					return IDX_OR;
				}
				else 
				{
					fseek(m_yyin, -1, SEEK_CUR);
					m_yytext[m_yylength++] = chr;
					lexerror("");
				}
			}
			else 
			{
				lexerror("");
			}
			break;
		case 1:
			if (isdigit(chr))
			{
				m_yytext[m_yylength++] = chr;
				break;
			}
			else if (chr == '.')
			{
				m_yytext[m_yylength++] = chr;
				state = 4;
				break;
			}
			else
			{
				fseek(m_yyin, -1, SEEK_CUR);
				state = 0;
				return IDX_CONST_INTEGER;
			}
			break;
		case 2:
			if (isdigit(chr) || isalpha(chr) || chr == '_')
			{
				m_yytext[m_yylength++] = chr;
				break;
			}
			else 
			{
				fseek(m_yyin, -1, SEEK_CUR);
				state = 0;
				if (strcmp(m_yytext, "if") == 0)
				{
					return IDX_IF;
				}
				else if (strcmp(m_yytext, "else") == 0)
				{
					return IDX_ELSE;
				}
				else if (strcmp(m_yytext, "then") == 0)
				{
					return IDX_THEN;
				}
				else if (strcmp(m_yytext, "endif") == 0)
				{
					return IDX_ENDIF;
				}
				else if (strcmp(m_yytext, "for") == 0)
				{
					return IDX_FOR;
				}
				else if (strcmp(m_yytext, "do") == 0)
				{
					return IDX_DO;
				}
				else if (strcmp(m_yytext, "endfor") == 0)
				{
					return IDX_ENDFOR;
				}
				else if (strcmp(m_yytext, "while") == 0)
				{
					return IDX_WHILE;
				}
				else if (strcmp(m_yytext, "endwhile") == 0)
				{
					return IDX_ENDWHILE;
				}
				else if (strcmp(m_yytext, "begin") == 0)
				{
					return IDX_BEGIN_FLOW;
				}
				else if (strcmp(m_yytext, "end") == 0)
				{
					return IDX_END_FLOW;
				}
				else if (strcmp(m_yytext, "run") == 0)
				{
					return IDX_RUN;
				}
				else if (strcmp(m_yytext, "int") == 0)
				{
					return IDX_INTEGER;
				}
				else if (strcmp(m_yytext, "string") == 0)
				{
					return IDX_STRING;
				}
				else if (strcmp(m_yytext, "float") == 0)
				{
					return IDX_FLOAT;
				}
				else if (strcmp(m_yytext, "memblock") == 0)
				{
					return IDX_MEMBLOCK;
				}
				else if (strcmp(m_yytext, "return") == 0)
				{
					return IDX_RETURN;
				}
				else if (strcmp(m_yytext, "continue") == 0)
				{
					return IDX_CONTINUE;
				}
				else if (strcmp(m_yytext, "break") == 0)
				{
					return IDX_BREAK;
				}
				else if (strcmp(m_yytext, "repeat") == 0)
				{
					return IDX_REPEAT;
				}
				else if (strcmp(m_yytext, "until") == 0)
				{
					return IDX_UNTIL;
				}
				else
				{
					return IDX_ID;
				}
				return IDX_ID;				
			}
			break;
		case 3:
			if (chr == '\\')
			{
				int nextchr = fgetc(m_yyin);
				if (nextchr == '"')
				{
					m_yytext[m_yylength++] = '"';
					break;
				}
				else if (nextchr == '\\')
				{
					m_yytext[m_yylength++] = '\\';
					break;
				}
				else 
				{		
					lexerror("转义符使用错误.");
				}
			}
			else if (chr == '"')
			{
				state = 0;
				return IDX_CONST_STRING;
			}
			else if (chr != '"' && chr != '\n')
			{
				m_yytext[m_yylength++] = chr;
				break;
			}
			else 
			{
				lexerror("");
			}
			break;
		case 4:
			if (isdigit(chr))
			{
				m_yytext[m_yylength++] = chr;
				state = 5;
				break;
			}
			else
			{
				lexerror("");
			}
			break;
		case 5:
			if (isdigit(chr))
			{
				m_yytext[m_yylength++] = chr;
				break;	
			}
			else 
			{
				fseek(m_yyin, -1, SEEK_CUR);
				state = 0;
				return IDX_CONST_FLOAT;				
			}
			break;
		}
	}
}

void CLex::lexerror(const char *msg)
{
	assert(msg != NULL);
	fprintf(stderr, "词法分析错误! %s lineno=[%d] text=[%s]\n",
				msg,
				m_lineno,
				m_yytext);
	exit(-1);
}

const char * CLex::getyytext()
{
	return m_yytext;
}

int CLex::getlineno()
{
	return m_lineno;
}

void CLex::rollback()
{
	fseek(m_yyin, m_position, SEEK_SET);
}
