%{
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "tool.h"

#define MAX_INCLUDE_DEPTH 20

static void GetRidOfQuote(const char * src, char* dest);
static int search_file(const char * basename, char *fullname);

YY_BUFFER_STATE include_stack[MAX_INCLUDE_DEPTH];

/*���ڴ�����ļ�������*/
char filename[256];

/*�����������򿪵��ļ�������*/
char file_to_open[256];

/*�����ļ����ֵ�ջ*/
char filename_stack[MAX_INCLUDE_DEPTH][256];

/*�����б�ŵ�ջ*/
int yylineno_stack[MAX_INCLUDE_DEPTH];
int include_stack_ptr = 0;
int yylineno = 1;
%}

%x incl

%%
<*>\n		{yylineno++; REJECT;}
<*>\r		{}
#include[ \t]+	{BEGIN(incl);}

<incl>\"[^ \t\n\"]+\"	{
		/*ȡ���µ��ļ���*/
		GetRidOfQuote(yytext, file_to_open);
		trim(file_to_open);
		/*�м���·�����������ļ�*/
		if (file_to_open[0] == '/' ||
	    		memcmp(file_to_open, "../", 3) == 0 ||	
			memcmp(file_to_open, "./", 2) == 0)
		{
		}
		else
		{
			/*�����ļ����õ�ȫ·��*/
			char fullfilename[256];
			if (search_file(file_to_open, fullfilename) == 0)
			{
				strcpy(file_to_open, fullfilename);	
			}
			else
			{
				fprintf(stderr, "�Ҳ����ļ�[%s]!\n", file_to_open);
				exit(-1);
			}
		}
	}

<incl>\n    { /* open the include file */
	if ( include_stack_ptr >= MAX_INCLUDE_DEPTH )
	{
		fprintf(stderr, "�ļ������Ĳ��̫��!" );
		exit(-1);
	}
	/*���浱ǰ�ļ�����Ϣ�����뻺����*/
	include_stack[include_stack_ptr] = YY_CURRENT_BUFFER;
	yylineno_stack[include_stack_ptr] = yylineno;
	strcpy(filename_stack[include_stack_ptr], filename);
	include_stack_ptr++;

	strcpy(filename, file_to_open);
	yyin = fopen( filename, "rb" );
	if ( ! yyin )
	{
		fprintf(stderr, "���ļ�[%s]ʧ��!\n", filename);
		exit (-1);
	}
	yy_switch_to_buffer(yy_create_buffer( yyin, YY_BUF_SIZE));
	fprintf(yyout, "#line 1 %s\n", filename);
	yylineno = 1;
	BEGIN(INITIAL);
	}
<incl>.             {}
<INITIAL>\-\-.*$	{/*ȥ��ע��*/}
<INITIAL>.|\n       {ECHO;}
<INITIAL>"__FILE__" {fprintf(yyout, "\"%s\"", filename);}
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
			fprintf(yyout, "#line %d %s\n", yylineno, filename);
		}
   	}
%%
static char search_dir[2048];	/*�����ļ�������Ŀ¼,ð�ŷָ�*/
int main(int argc, char** argv)
{
   /*��ȡ����*/
   int opt;
   char input_file[255];
   char output_file[255];
   memset(input_file, 0, sizeof(input_file));
   memset(output_file, 0, sizeof(output_file));
   memset(search_dir, 0, sizeof(search_dir));
   strcpy(search_dir, ".:");
   /*
   *-I �������ļ���������Ŀ¼�������ж��
   *-o ����ļ��� ֻ����һ��������ж���������һ��Ϊ׼
   *-f ������ļ��� ֻ����һ��������ж���������һ��Ϊ׼
   */
   while (1)
   {
   	opt = getopt(argc, argv, "I:o:f:");
	if (opt == -1)
	{
		break;
	}
	switch (opt)
	{
		case 'I':
			strncat(search_dir, optarg, 
				sizeof(search_dir) - 1 - strlen(search_dir));
			strncat(search_dir, ":",
				sizeof(search_dir) - 1 - strlen(search_dir));
			break;
		case 'o':
			strncpy(output_file, optarg, sizeof(output_file));
			break;
		case 'f':
			strncpy(input_file, optarg, sizeof(input_file));
			break;
		default:
			return -1;
			break;
	}
   }
   /*�����������*/
   yyin = stdin;
   yyout = stdout;
   strcpy(filename, "standard input file");
   if (strlen(input_file) > 0)
   {
   	if ((yyin = fopen(input_file, "rb") ) == NULL)
   	{
		fprintf(stderr, "�������ļ�[%s]ʧ��!\n", input_file);
		return -1;	
   	}
   	strcpy(filename, input_file);
   }
   if (strlen(output_file) > 0)
   {
   	if ((yyout = fopen(output_file, "wb") ) == NULL)
   	{
		fprintf(stderr, "������ļ�[%s]ʧ��!\n", output_file);
		return -1;	
   	}
   }
   /*�ʷ�����*/
   fprintf(yyout, "#line 1 %s\n", filename);
   yylex();
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
*��search_dir�������ļ���Ϊbasename���ļ�������ҵ�������0
*���ҽ��ļ�ȫ·��������fullname�У����򣬷���-1
*/
static int search_file(const char * basename, char *fullname)
{
	char dir[256];
	char file[256];
	int start, end;
	const int len = strlen(search_dir);
	start = 0;
	end = 0;
	while (1)
	{	
		if (end >= len)
		{
			break;
		}
		if (search_dir[end] != ':')
		{
			end++;
			continue;
		}
		memset(dir, 0, sizeof(dir));
		memcpy(dir, search_dir + start, (end - start));
		trim(dir);
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
		/*��������*/
		end++;
		start = end;
		
	}
	return -1;
}
