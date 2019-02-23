%{
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "util.h"
#include "prep.h"
#include <unistd.h>

using namespace lnb;

#define MAX_INCLUDE_DEPTH 20
#define YY_DECL static int yylex()


static void GetRidOfQuote(const char * src, char* dest);
static int search_file(const char * basename, char *fullname);

static YY_BUFFER_STATE include_stack[MAX_INCLUDE_DEPTH];

/*正在处理的文件的名字*/
static char filename[256];

/*被包含将被打开的文件的名字*/
static char file_to_open[256];

/*保存文件名字的栈*/
static char filename_stack[MAX_INCLUDE_DEPTH][256];

/*保存行编号的栈*/
static int yylineno_stack[MAX_INCLUDE_DEPTH];
static int include_stack_ptr = 0;
static int yylineno = 1;
static const char * g_search_dir = NULL;
%}

%x incl s_comment

%%
<*>\n		{
		yylineno++; 
		REJECT;}
<*>\r		{}
@include[ \t]+	{BEGIN(incl);}

<incl>\"[^ \t\n\"]+\"	{
		/*取得新的文件名*/
		GetRidOfQuote(yytext, file_to_open);
		//trim(file_to_open);
		/*有几种路径无须搜索文件*/
		if (file_to_open[0] == '/' ||
	    		memcmp(file_to_open, "../", 3) == 0 ||	
			memcmp(file_to_open, "./", 2) == 0)
		{
		}
		else
		{
			/*搜索文件，得到全路径*/
			char fullfilename[256];
			if (search_file(file_to_open, fullfilename) == 0)
			{
				strcpy(file_to_open, fullfilename);	
			}
			else
			{
				fprintf(stderr, "找不到文件[%s]!\n", file_to_open);
				exit(-1);
			}
		}
	}

<incl>\n    { /* open the include file */
	if ( include_stack_ptr >= MAX_INCLUDE_DEPTH )
	{
		fprintf(stderr, "文件包含的层次太深!" );
		exit(-1);
	}
	/*保存当前文件的信息和输入缓冲区*/
	include_stack[include_stack_ptr] = YY_CURRENT_BUFFER;
	yylineno_stack[include_stack_ptr] = yylineno;
	strcpy(filename_stack[include_stack_ptr], filename);
	include_stack_ptr++;

	strcpy(filename, file_to_open);
	yyin = fopen( filename, "rb" );
	if ( ! yyin )
	{
		fprintf(stderr, "打开文件[%s]失败!\n", filename);
		exit (-1);
	}
	yy_switch_to_buffer(yy_create_buffer( yyin, YY_BUF_SIZE));
	fprintf(yyout, "@line=1:%s\n", filename);
	yylineno = 1;
	BEGIN(INITIAL);
	}
<incl>.             {}
	/*常量字符串*/
<INITIAL>\"			{ECHO; BEGIN(s_comment);}
<s_comment>\\\"		{ECHO;}
<s_comment>[^\"]	{ECHO;}
<s_comment>\"		{ECHO; BEGIN(INITIAL);}

<INITIAL>.|\n       {ECHO;}
<INITIAL>"__FILE__" {
			string sFileName;
			StringEscape(filename, sFileName);
			fprintf(yyout, "\"%s\"", sFileName.c_str());
		}
<INITIAL>"__LINE__" {fprintf(yyout, "%d", yylineno);}

<<EOF>>     {
		if ( --include_stack_ptr < 0 )
		{
			return 0;
		}
		else
		{
			yy_delete_buffer( YY_CURRENT_BUFFER );
			yy_switch_to_buffer( include_stack[include_stack_ptr] );
			yylineno = yylineno_stack[include_stack_ptr];
			strcpy(filename, filename_stack[include_stack_ptr]);
			fprintf(yyout, "@line=%d:%s\n", yylineno, filename);
		}
   	}
%%
int lnb::precompile(const char * input_file, const char * output_file, const char * search_dir)
{
	g_search_dir = search_dir;
   if (strlen(input_file) > 0)
   {
   	if ((yyin = fopen(input_file, "rb") ) == NULL)
   	{
		fprintf(stderr, "打开输入文件[%s]失败!\n", input_file);
		return -1;	
   	}
   	strncpy(filename, input_file, sizeof(filename)-1);
	filename[ sizeof(filename)-1 ] = '\0';
   }
   if (strlen(output_file) > 0)
   {
   	if ((yyout = fopen(output_file, "wb") ) == NULL)
   	{
		fprintf(stderr, "打开输出文件[%s]失败!\n", output_file);
		return -1;	
   	}
   }
   /*词法分析*/
   fprintf(yyout, "@line=1:%s\n", input_file);
   if (yylex() != 0)
   {
   		return -1;
   }
   fclose(yyout);
   fclose(yyin);
   return 0;
}
static void GetRidOfQuote(const char * src, char* dest)
{
   int len = strlen(src);
   if (len >= 2)
   {
	strcpy(dest, src + 1);
	dest[len - 2] = 0;
   }
   else
   {
	strcpy(dest, "");
   }
}
/*
*在g_search_dir中搜索文件名为basename的文件，如果找到，返回0
*并且将文件全路径保存在fullname中，否则，返回-1
*/
static int search_file(const char * basename, char *fullname)
{
	char dir[256];
	char file[256];
	int start, end;
	const int len = strlen(g_search_dir);
	start = 0;
	end = 0;
	while (1)
	{	
		if (end >= len)
		{
			break;
		}
		if (g_search_dir[end] != ':')
		{
			end++;
			continue;
		}
		memset(dir, 0, sizeof(dir));
		memcpy(dir, g_search_dir + start, (end - start));
		//trim(dir);
		if (dir[strlen(dir) - 1] == '/')
		{
			dir[strlen(dir) - 1] = 0;
		}
		snprintf(file, sizeof(file), 
			"%s/%s", dir, basename);
		if (access(file, R_OK | F_OK) == 0)
		{
			strcpy(fullname, file);
			return 0;
		}
		/*继续搜索*/
		end++;
		start = end;
		
	}
	return -1;
}
int yywrap()
{
    return 1;   
}
