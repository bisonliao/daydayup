#include "script.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <list>
#include "var.h"
using namespace std;

static map<string, CScript> gs_AllScript;

int RunScript(const string & scriptname, const list<CVar> &args, CVar & rtnval);

int main(int argc, char**argv)
{
	if (argc < 2)
	{
		return 0;
	}
	FILE * fp = NULL;
	if ( (fp = fopen(argv[1], "rb")) == NULL)
	{
		perror("fopen:");
		return -1;
	}
	while (1)
	{
		char buf[3000];
		CScript sss;
		string scriptname;
		vector<string> instructs;

		while (1)
		{
			memset(buf, 0, sizeof(buf));
			if (fgets(buf, sizeof(buf)-1, fp) == NULL)
			{
				break;
			}
			int len = strlen(buf);
			if (len < 3)
			{
				fprintf(stderr, "非法指令:%s\n", buf);
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
			instructs.push_back(buf);

			if (strncmp("!!!BEGIN ", buf, 9) == 0)
			{
				scriptname = buf + 9;	
			}
			if (strncmp("!!!END", buf, 6) == 0)
			{
				break;
			}
		}
		if (instructs.empty())
		{
			break;
		}
		if (sss.PushInstruct(instructs))
		{
			fprintf(stderr, "指令非法!!!\n");
			return -1;
		}
		gs_AllScript.insert(std::pair<string, CScript>(scriptname, sss));
	}
	fclose(fp);
	fp = NULL;

	list<CVar> args;
	CVar rtnval;
	return RunScript("main", args, rtnval);
}

int RunScript(const string & scriptname, const list<CVar> &args, CVar & rtnval)
{
	map<string,CScript>::iterator script_it = gs_AllScript.find(scriptname);
	if (script_it == gs_AllScript.end())
	{
		fprintf(stderr, "Error! 没有找到函数%s!\n", scriptname.c_str());
		return -1;
	}
	CScript sss = script_it->second;
	if (sss.Run(args, rtnval) < 0)
	{
		return -1;
	}
	return 0;
}
