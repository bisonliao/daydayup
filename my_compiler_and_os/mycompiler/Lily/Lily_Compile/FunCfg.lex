%{
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream.h>
#include <fstream.h>
#include "lily_fc.h"
#include "expr_type.h"

#define FUN_NAME	260	/*函数名称*/
#define FUN_ARGNUM	261	/*函数参数个数*/
#define FUN_NO		262	/*函数编号*/
#define FUN_TYPE	263	/*函数类型*/

#define ID_MAX 256	/*函数名称的缓冲区最大长度*/
#define FUNC_MAX 100	/*函数配置的最大个数*/

typedef union
{
	char fun_name[ID_MAX];
	int	fun_argnum;
	int 	fun_no;
	int	fun_type;
}	VALUE;
static VALUE g_value;

typedef struct
{
	char fun_name[ID_MAX];
	int	fun_argnum;
	int 	fun_no;
	int	fun_type;
} FUNC;

static FUNC g_function[FUNC_MAX];
static int g_funcount;

%}
%option C++

%s s_ARGNUM s_NO s_COMMENT s_TYPE

%%
<*>[ \t\n]+	{}
<INITIAL>#.*$	{}
<INITIAL>^[ \t]*$	{/*空行*/}

<INITIAL>[a-zA-Z][a-zA-Z0-9]*	{
			if (yyleng >= ID_MAX)
			{
				return -1;
			}
			strcpy(g_value.fun_name, yytext);
			BEGIN s_ARGNUM;
			return FUN_NAME;
		}
<s_ARGNUM>[-+]?((0)|([1-9][0-9]*))	{
			g_value.fun_argnum = atoi(yytext);
			BEGIN s_NO;
			return FUN_ARGNUM;
		}
<s_NO>[-+]?((0)|([1-9][0-9]*))      {
			g_value.fun_no = atoi(yytext);
			BEGIN s_TYPE;
			return FUN_NO;
		}
<s_TYPE>int	{
			g_value.fun_type = TYPE_INTEGER;
			BEGIN INITIAL;
			return FUN_TYPE;
		}
<s_TYPE>float	{
			g_value.fun_type = TYPE_FLOAT;
			BEGIN INITIAL;
			return FUN_TYPE;
		}
<s_TYPE>string	{
			g_value.fun_type = TYPE_STRING;
			BEGIN INITIAL;
			return FUN_TYPE;
		}
<s_TYPE>memblock	{
			g_value.fun_type = TYPE_MEMBLOCK;
			BEGIN INITIAL;
			return FUN_TYPE;
		}
<*>.		{return -1;}
%%
/*
*从文件fname中读取函数配置
*成功返回0，失败返回-1
*/
int fcInit(const char * fname)
{
	ifstream fs;
	fs.open(fname);
	fcFlexLexer lex(&fs);
	int i = 0;
	int ret;
	while ( i < FUNC_MAX)
	{
		ret = lex.yylex();
		if (ret == 0)
		{
			break;
		}
		if (ret != FUN_NAME)
		{
			return -1;
		}
		strcpy(g_function[i].fun_name, g_value.fun_name);

		ret = lex.yylex();
		if (ret != FUN_ARGNUM)
		{
			return -2;
		}
		g_function[i].fun_argnum = g_value.fun_argnum;

		ret = lex.yylex();
		if (ret != FUN_NO)
		{
			return -3;
		}
		g_function[i].fun_no = g_value.fun_no;

		ret = lex.yylex();
		if (ret != FUN_TYPE)
		{
			return -4;
		}
		g_function[i].fun_type = g_value.fun_type;
	
		i++;
	}
	g_funcount = i;
	return 0;
}
/*
*根据函数名称FuncName得到函数的参数个数和编号
*成功返回0， 失败返回-1
*/
int fcGetFunc(const char* FuncName, int & argnum, int &number, int &type)
{
	int i;
	for (i = 0; i < g_funcount; i++)
	{
		if (strcmp(g_function[i].fun_name, FuncName) == 0)
		{
			argnum = g_function[i].fun_argnum;
			number = g_function[i].fun_no;
			type = g_function[i].fun_type;
			return 0;
		}
	}
	return -1;
}
/*
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		return -1;
	}	
	int ret;
	
	if ( (ret = fcInit(argv[1])) < 0)
	{
		fprintf(stderr, "fcInit failed! ret=[%d]\n", ret);
		return -1;
	}
	int i;
	for (i = 0; i < g_funcount; i++)
	{
		printf("[%s][%d][%d][%d]\n", g_function[i].fun_name,
				g_function[i].fun_argnum,
				g_function[i].fun_no,
				g_function[i].fun_type);
	}
	return 0;
}
*/
int fcwrap()
{
	return 1;
}
