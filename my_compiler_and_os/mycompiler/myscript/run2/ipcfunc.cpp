#include <stdlib.h>
#include <stdio.h>
#include <deque>
#include <string.h>
#include "var.h"
#include "util.h"
#include <string>
#include "script.h"
#include "perlc.h"

using namespace std;
using namespace lnb;




int CScript::ipcfunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret)
{
	if ("lnb_system" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		//Óï·¨¼ì²é
		if (arglist.size() != 1 ||
			arglist[0]->Type() != CVar::T_STR )
		{
			fprintf(stderr, "Invalid argument for int system(string str)\n");
			return -1;
		}
		ret.IntVal() = system(arglist[0]->StrVal().c_str());

		return 0;
	}
	return -1;
}
