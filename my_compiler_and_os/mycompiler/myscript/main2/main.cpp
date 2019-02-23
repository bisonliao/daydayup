#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "run.h"
#include "compile.h"
#include "prep.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace lnb;

static char search_dir[10240] = {0};   /*包含文件的搜索目录,冒号分割*/

int main(int argc, char ** argv)
{
	string ScriptFile = "";
	bool bCompile = false;
	bool bRun = false;
	int ch;


	while((ch = getopt(argc,argv,"rcf:I:"))!= -1)
	{
		switch(ch)
		{
			case 'c':
				bCompile = true;
				break;
			case 'r':
				bRun = true;
				break;
			case 'f':
				ScriptFile =  optarg;
				break;
			case 'I':
				strncat(search_dir, optarg, sizeof(search_dir) - strlen(search_dir) -2);
				strcat(search_dir, ":");
			default:
				fprintf(stderr, "unkown option %c!\n", ch);
				return -1;
		}
	}
	int iNextArg = optind;
	deque<string> arrArgs;
	for (;iNextArg < argc; ++iNextArg)
	{
		arrArgs.push_back( argv[iNextArg] );
	}	

	if (ScriptFile.length() < 1 ||
		bCompile && bRun)
	{
		fprintf(stderr, "invalid argument!\n");
		return -1;
	}
	if (strlen(search_dir) < 1)
	{
		strcpy(search_dir, ".:");
	}
	bool IsTmpFile = false;
	if (!bRun) //不是直接运行中间代码文件，那么需要做预编译
	{
		char mediafile[256];
		mediafile[0] = '\0';
		tmpnam(mediafile);	
		if (strlen(mediafile) < 1)
		{
			fprintf(stderr, "产生临时文件失败!\n");
			return -1;
		}
		if (precompile(ScriptFile.c_str(), mediafile, search_dir) < 0)
		{
			return -1;
		}
		ScriptFile = mediafile;
		IsTmpFile = true;
	}
	

	deque<string>  PCodeList;
	int iRet;
	if (bCompile)
	{
		iRet = compile(ScriptFile, PCodeList) ;
		if ( iRet == 0)
		{
			deque<string>::const_iterator it;
			for (it = PCodeList.begin(); it != PCodeList.end(); ++it)
			{
				printf("%s\n", it->c_str());
			}
		}
		if (IsTmpFile)
		{
			unlink(ScriptFile.c_str());
		}
		return iRet;
	}
	if (bRun)
	{
		FILE * fp = NULL;
		if ( (fp = fopen(ScriptFile.c_str(), "rb")) == NULL)
		{
			perror("fopen:");
			if (IsTmpFile)
			{
				unlink(ScriptFile.c_str());
			}
			return -1;
		}
		char buf[3000];
		while (1)
		{
			if (fgets(buf, sizeof(buf)-1, fp) == NULL)
			{
				break;
			}
			int len = strlen(buf);
			if (len < 3)
			{
				fprintf(stderr, "非法指令:%s\n", buf);
				if (IsTmpFile)
				{
					unlink(ScriptFile.c_str());
				}
				return -1;
			}
			if (buf[len-1] == '\n')
			{
				buf[len-1] = '\0';
			}
			if (buf[len-2] == '\r')
			{
				buf[len-2] = '\0';
			}
			PCodeList.push_back(buf);
		}
		fclose(fp);
		fp = NULL;

		if (IsTmpFile)
		{
			unlink(ScriptFile.c_str());
		}
	
		return Run( PCodeList, arrArgs );
	}

	if (compile(ScriptFile, PCodeList) == 0)
	{
		if (IsTmpFile)
		{
			unlink(ScriptFile.c_str());
		}
		return Run(PCodeList, arrArgs);
	}

	return -1;
}
