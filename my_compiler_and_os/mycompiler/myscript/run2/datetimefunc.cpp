#include <stdlib.h>
#include <stdio.h>
#include <deque>
#include <string.h>
#include <time.h>
#include "var.h"
#include "util.h"
#include <string>
#include "script.h"
#include "perlc.h"

using namespace std;
using namespace lnb;




int CScript::datetimefunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret)
{
	if ("lnb_time" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		//语法检查
		if (arglist.size() != 0 && arglist.size() != 1)
		{
			fprintf(stderr, "Invalid argument for int time(t)\n");
			return -1;
		}
		time_t curt;
		curt = time(NULL);
		ret.IntVal() = curt;

		if (arglist.size() == 1)
		{
			arglist[0]->Type() = CVar::T_INT;
			arglist[0]->IntVal() = curt;
		}

		return 0;
	}
	else if ("lnb_localtime" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		//语法检查
		if (arglist.size() < 2 || arglist.size() > 4 || 
			arglist[0]->Type() != CVar::T_INT )
			
		{
			fprintf(stderr, "Invalid argument for int localtime(int t, string dt [, int yday, int wday] )\n");
			return -1;
		}
		time_t tt;
		if (arglist[0]->IntVal() == -1)
		{
			tt = time(NULL);
		}
		else
		{
			tt = arglist[0]->IntVal();
		}
		struct tm *ptm = localtime(&tt);
		if (ptm == NULL)
		{
			arglist[1]->StrVal() = "";
			return 0;
		}
		char buf[100];
		snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
                      ptm->tm_year+1900,
                      ptm->tm_mon+1,
                      ptm->tm_mday,
                      ptm->tm_hour,
                      ptm->tm_min,
                      ptm->tm_sec);
		arglist[1]->Type() = CVar::T_STR;
		arglist[1]->StrVal() = buf;
		if (arglist.size() >= 3)
		{
			arglist[2]->Type() = CVar::T_INT;
			arglist[2]->IntVal() = ptm->tm_yday;
		}
		if (arglist.size() >= 4)
		{
			arglist[3]->Type() = CVar::T_INT;
			arglist[3]->IntVal() = ptm->tm_wday;
		}
		ret.IntVal() = 0;
			
		return 0;
	}
	else if ("lnb_str2time" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		//语法检查
		if (arglist.size() != 1 ||
			arglist[0]->Type() != CVar::T_STR )
			
		{
			fprintf(stderr, "Invalid argument for int str2time(string dt)\n");
			return -1;
		}
		int iRet, year, month, day, hour, minute, second;
		struct tm tt;
		iRet = sscanf(arglist[0]->StrVal().c_str(), "%d-%d-%d %d:%d:%d",
				&tt.tm_year,
				&tt.tm_mon,
				&tt.tm_mday,
				&tt.tm_hour,
				&tt.tm_min,
				&tt.tm_sec);
		if (iRet != 6)
		{
			return 0;
		}
		tt.tm_year -= 1900;
		tt.tm_mon -= 1;
		
		ret.IntVal() = mktime(&tt);
			
		return 0;
	}
	return -1;
}
