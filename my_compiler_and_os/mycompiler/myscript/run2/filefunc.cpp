#include <stdlib.h>
#include <stdio.h>
#include <deque>
#include <string.h>
#include "var.h"
#include "util.h"
#include <string>
#include "script.h"

using namespace std;
using namespace lnb;

#define MAX_FILE_NUM 100


static FILE * g_pstFILE[ MAX_FILE_NUM ] = {NULL};
static unsigned char g_aucIOBuf[1024 * 1024];

void CScript::filefunc_end()
{
	for (int i = 0; i < MAX_FILE_NUM; ++i)
	{
		if (g_pstFILE[i] != NULL)
		{
			fclose(g_pstFILE[i]);
			g_pstFILE[i] = NULL;
		}
	}
}

int CScript::filefunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret)
{
	static bool bInitStdIo = false;
	if (!bInitStdIo)
	{
		g_pstFILE[0] = stdin;
		g_pstFILE[1] = stdout;
		g_pstFILE[2] = stderr;

		bInitStdIo = true;
	}
	if ("lnb_fopen" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 2 || arglist[0]->Type() != CVar::T_STR || arglist[1]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\n\tint fopen(string filename, string mode)\n", funcname.c_str());
			return -1;
		}

		int iIndex = -1, i;
		for (i = 0; i < MAX_FILE_NUM; ++i)
		{
			if (g_pstFILE[i] == NULL)
			{
				iIndex = i;
				break;
			}
		}
		if (-1 == iIndex)
		{
			fprintf(stderr, "执行函数%s失败， 打开的文件太多\n", funcname.c_str());
			return 0;
		}
		FILE * fp = fopen(arglist[0]->StrVal().c_str(), arglist[1]->StrVal().c_str());
		if (NULL != fp)
		{
			g_pstFILE[iIndex] = fp;
			ret.IntVal()  = iIndex;
		}
		return 0;
	}
	else if ("lnb_popen" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 2 || arglist[0]->Type() != CVar::T_STR || arglist[1]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\n\tint popen(string filename, string mode)\n", funcname.c_str());
			return -1;
		}

		int iIndex = -1, i;
		for (i = 0; i < MAX_FILE_NUM; ++i)
		{
			if (g_pstFILE[i] == NULL)
			{
				iIndex = i;
				break;
			}
		}
		if (-1 == iIndex)
		{
			fprintf(stderr, "执行函数%s失败， 打开的文件太多\n", funcname.c_str());
			return 0;
		}
		FILE * fp = popen(arglist[0]->StrVal().c_str(), arglist[1]->StrVal().c_str());
		if (NULL != fp)
		{
			g_pstFILE[iIndex] = fp;
			ret.IntVal()  = iIndex;
		}
		return 0;
	}
	else if ( "lnb_fclose" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 1 || arglist[0]->Type() != CVar::T_INT)
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\n\tint fclose(int fptr)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_FILE_NUM ||
			g_pstFILE[ arglist[0]->IntVal() ] == NULL)
		{
			return 0;
		}
		ret.IntVal() = fclose(g_pstFILE[ arglist[0]->IntVal() ]);
		g_pstFILE[ arglist[0]->IntVal() ] = NULL;
		return 0;
	}
	else if ( "lnb_pclose" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		//语法检查
		if (arglist.size() != 1 || arglist[0]->Type() != CVar::T_INT)
		{
			fprintf(stderr, "执行函数%s失败， 参数不正确!\n\tint pclose(int fptr)\n", funcname.c_str());
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal() >= MAX_FILE_NUM ||
			g_pstFILE[ arglist[0]->IntVal() ] == NULL)
		{
			return 0;
		}
		ret.IntVal() = pclose(g_pstFILE[ arglist[0]->IntVal() ]);
		g_pstFILE[ arglist[0]->IntVal() ] = NULL;
		return 0;
	}
	else if ( "lnb_fread" == funcname)
	{
		//size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() != 4 ||
		    arglist[1]->Type() != CVar::T_INT ||
		    arglist[2]->Type() != CVar::T_INT ||
		    arglist[3]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "Invalid argument for function 'int fread(string, int, int, int)'\n");
			return -1;
		}
		if (arglist[1]->IntVal() < 0 || arglist[2]->IntVal() < 0)
		{
			fprintf(stderr, "fread() arg#2, arg#3 can not be negitive.\n");
			return 0;
		}
		int iByteToRead = arglist[1]->IntVal() * arglist[2]->IntVal();
		if (iByteToRead > sizeof(g_aucIOBuf))
		{
			fprintf(stderr, "fread() arg#2 * arg#4 too large!\n");
			return 0;
		}
		if (arglist[3]->IntVal() < 0 || arglist[3]->IntVal()  >=  MAX_FILE_NUM)
		{
			fprintf(stderr, "Invalid file handle for fread().\n");
			return 0;
		}
		if (g_pstFILE[ arglist[3]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for fread().\n");
			return 0;
		}
		ret.IntVal() = fread(g_aucIOBuf, arglist[1]->IntVal(), arglist[2]->IntVal(), 
				g_pstFILE[ arglist[3]->IntVal() ] );
		if (ret.IntVal() > 0)
		{
			arglist[0]->Type() = CVar::T_STR;
			arglist[0]->StrVal() = string( (const  char *)g_aucIOBuf, 
					(string::size_type)(arglist[1]->IntVal() * ret.IntVal()) );
		}
		return 0;
	}
	else if ( "lnb_fwrite" == funcname)
	{
		//size_t fwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() != 4 ||
		    arglist[0]->Type() !=  CVar::T_STR ||
		    arglist[1]->Type() != CVar::T_INT ||
		    arglist[2]->Type() != CVar::T_INT ||
		    arglist[3]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "Invalid argument for function 'int fwrite(string, int, int, int)'\n");
			return -1;
		}
		if (arglist[1]->IntVal() < 0 || arglist[2]->IntVal() < 0)
		{
			fprintf(stderr, "fwrite() arg#2, arg#3 can not be negitive.\n");
			return 0;
		}
		int iByteToWrite = arglist[1]->IntVal() * arglist[2]->IntVal();
		if (iByteToWrite > arglist[0]->StrVal().length())
		{
			fprintf(stderr, "fwrite() arg#2, arg#3 too large!\n");
			return 0;
		}
		if (arglist[3]->IntVal() < 0 || arglist[3]->IntVal()  >=  MAX_FILE_NUM)
		{
			fprintf(stderr, "Invalid file handle for fwrite().\n");
			return 0;
		}
		if (g_pstFILE[ arglist[3]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for fwrite().\n");
			return 0;
		}
		ret.IntVal() = fwrite(arglist[0]->StrVal().data(), arglist[1]->IntVal(), arglist[2]->IntVal(), 
				g_pstFILE[ arglist[3]->IntVal() ] );
		return 0;
	}
	else if ( "lnb_fgets" == funcname)
	{
		//int fgets(string buf, int size, int stream);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() != 3 ||
		    arglist[1]->Type() != CVar::T_INT ||
		    arglist[2]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "Invalid argument for function 'int fgets(string buf, int size, int stream)'\n");
			return -1;
		}
		if (arglist[1]->IntVal() < 0 || arglist[1]->IntVal() >= sizeof(g_aucIOBuf))
		{
			fprintf(stderr, "fgets() arg#2 invalid.\n");
			return 0;
		}
		if (arglist[2]->IntVal() < 0 || arglist[2]->IntVal()  >=  MAX_FILE_NUM)
		{
			fprintf(stderr, "Invalid file handle for fgets().\n");
			return 0;
		}
		if (g_pstFILE[ arglist[2]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for fgets().\n");
			return 0;
		}
		g_aucIOBuf[ arglist[1]->IntVal() ] = '\0';
		if (NULL != fgets( (char*)g_aucIOBuf, arglist[1]->IntVal(), g_pstFILE[ arglist[2]->IntVal() ] ) ) 
		{
			ret.IntVal() = 0;
		}
		arglist[0]->Type() = CVar::T_STR;
		arglist[0]->StrVal() = (const char *)g_aucIOBuf;

		return 0;
	}
	else if ( "lnb_fprintf" == funcname)
	{
		//int fprintf(FILE *stream, const char *format, ...);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() < 2 ||
		    arglist[0]->Type() != CVar::T_INT ||
		    arglist[1]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "Invalid argument for function 'int fprintf(int stream, string format, ...)'");
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal()  >=  MAX_FILE_NUM)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str());
			return 0;
		}
		if (g_pstFILE[ arglist[0]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str() );
			return 0;
		}
		string ss;
		deque<CVar*> arglist2;
		for (int i = 1; i < arglist.size(); ++i)
		{
			arglist2.push_back(arglist[i]);
		}
		//展开格式化到ss
		ret.IntVal() = CVar::FormatStr(arglist2,  ss);
		if (ret.IntVal() < 0)
		{
			return 0;
		}
		//写入文件
		if (fwrite(ss.data(), 1, ss.length(), g_pstFILE[ arglist[0]->IntVal() ]) != ss.length())
		{
			ret.IntVal() = -1;
			return 0;
		}
		return 0;
	}
	else if ( "lnb_printf" == funcname)
	{
		//int fprintf(FILE *stream, const char *format, ...);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() < 1 ||
		    arglist[0]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "Invalid argument for function 'int printf(string format, ...)'");
			return -1;
		}
		if (g_pstFILE[ arglist[0]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str() );
			return 0;
		}
		string ss;
		deque<CVar*> arglist2;
		for (int i = 0; i < arglist.size(); ++i)
		{
			arglist2.push_back(arglist[i]);
		}
		//展开格式化到ss
		ret.IntVal() = CVar::FormatStr(arglist2,  ss);
		if (ret.IntVal() < 0)
		{
			return 0;
		}
		//写入文件
		if (fwrite(ss.data(), 1, ss.length(), g_pstFILE[1]) != ss.length())
		{
			ret.IntVal() = -1;
			return 0;
		}
		return 0;
	}
	else if ( "lnb_fseek" == funcname)
	{
		//int fseek(FILE *stream, long offset, int whence);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() != 3 ||
		    arglist[0]->Type() != CVar::T_INT ||
		    arglist[1]->Type() != CVar::T_INT ||
		    arglist[2]->Type() != CVar::T_INT)
		{
			fprintf(stderr, "Invalid argument for function 'int fseek(FILE *stream, long offset, int whence)'\n");
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal()  >=  MAX_FILE_NUM)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str());
			return 0;
		}
		if (g_pstFILE[ arglist[0]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str() );
			return 0;
		}
		ret.IntVal() = fseek( g_pstFILE[ arglist[0]->IntVal() ], arglist[1]->IntVal(), arglist[2]->IntVal());
		return 0;
	}
	else if ( "lnb_ftell" == funcname)
	{
		// long ftell(FILE *stream);
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() != 1 ||
		    arglist[0]->Type() != CVar::T_INT )
		{
			fprintf(stderr, "Invalid argument for function 'int fseek(FILE *stream, long offset, int whence)'\n");
			return -1;
		}
		if (arglist[0]->IntVal() < 0 || arglist[0]->IntVal()  >=  MAX_FILE_NUM)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str());
			return 0;
		}
		if (g_pstFILE[ arglist[0]->IntVal() ] == NULL)
		{
			fprintf(stderr, "Invalid file handle for %s().\n", funcname.c_str() );
			return 0;
		}
		ret.IntVal() = ftell( g_pstFILE[ arglist[0]->IntVal() ]);
		return 0;
	}
	return -1;
}
