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



static unsigned char g_aucIOBuf[1024 * 1024];

int CScript::stringfunc(const string & funcname,  deque<CVar*> &arglist, CVar & ret)
{
	if ("lnb_split" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		bool bIgnoreCase = false;
		//语法检查
		if (arglist.size() != 2 && arglist.size() != 3 ||
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_STR ||
			arglist.size() == 3 && arglist[2]->Type() != CVar::T_INT)
		{
			fprintf(stderr, "Invalid argument for int split(string str, string pattern, int IgnoreCase = 0)\n");
			return -1;
		}
		if (arglist.size() == 3 && arglist[2]->IntVal() != 0)
		{
			bIgnoreCase = true;
		}

		list<int> startpos, endpos;
		if (!re(arglist[0]->StrVal(),  arglist[1]->StrVal(), bIgnoreCase, startpos, endpos))
		{
			ret.IntVal() = 1;
			m_mem.SetArrayVar("$OUTDATA", *arglist[0], 0);
			return 0;
		}
		list<int>::const_iterator it1, it2;
		int iNonFmtStart = 0;
		ret.IntVal() = 0;
		for (it1 = startpos.begin(), it2 = endpos.begin();
			it1 != startpos.end(), it2 != endpos.end();
			++it1, ++it2)
		{
			//不是格式化的部分
			CVar part;
			int iNonFmtLen = *it1 - iNonFmtStart;
			part.Type() = CVar::T_STR;
			part.StrVal() = arglist[0]->StrVal().substr(iNonFmtStart, iNonFmtLen);

			m_mem.SetArrayVar("$OUTDATA", part, ret.IntVal()++ );

			
			iNonFmtStart = *it2;
		}
		if (iNonFmtStart < arglist[0]->StrVal().length())
		{
			CVar part;
			part.Type() = CVar::T_STR;
			part.StrVal() = arglist[0]->StrVal().substr(iNonFmtStart);

			m_mem.SetArrayVar("$OUTDATA", part, ret.IntVal()++ );
		}

		return 0;
	}
	else if ( "lnb_sprintf" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;
		if (arglist.size() < 2 ||
		    arglist[1]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "Invalid argument for function 'int sprintf(string str, string format, ...)'");
			return -1;
		}
		deque<CVar*> arglist2;
		for (int i = 1; i < arglist.size(); ++i)
		{
			arglist2.push_back(arglist[i]);
		}
		//展开格式化
		arglist[0]->Type() = CVar::T_STR;
		ret.IntVal() = CVar::FormatStr(arglist2,  arglist[0]->StrVal());
		return 0;
	}
	else if ("lnb_find" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		//find($string,$pattern, $substr, $ic, $startpos)

		bool bIgnoreCase = false;
		int iStartPos = 0;
		//语法检查
		if (arglist.size() < 3 || arglist.size() > 5 ||
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_STR ||
			arglist.size() >= 4 && arglist[3]->Type() != CVar::T_INT ||
			arglist.size() == 5 && arglist[4]->Type() != CVar::T_INT)
		{
			fprintf(stderr, "Invalid argument for int find($string,$pattern, $toSave, $ignoreCase, $startpos)\n");
			return -1;
		}
		if (arglist.size() >= 4 && arglist[3]->IntVal() != 0)
		{
			bIgnoreCase = true;
		}
		if (arglist.size() == 5 && arglist[4]->IntVal() > 0)
		{
			if (arglist[4]->IntVal() < arglist[0]->StrVal().length())
			{
				iStartPos = arglist[4]->IntVal();
			}
			else
			{
				return 0;
			}
		}

		list<int> startpos, endpos;
		string s = arglist[0]->StrVal().substr(iStartPos);
		if (!re(s,  arglist[1]->StrVal(), bIgnoreCase, startpos, endpos))
		{
			return 0;
		}
		list<int>::const_iterator it1, it2;
		for (it1 = startpos.begin(), it2 = endpos.begin();
			it1 != startpos.end(), it2 != endpos.end();
			++it1, ++it2)
		{
			ret.IntVal() = *it1 + iStartPos;
			arglist[2]->Type() = CVar::T_STR;
			arglist[2]->StrVal() = s.substr(*it1, *it2 - *it1);
		}

		return 0;
	}
	else if ("lnb_substr" == funcname)
	{
		ret.Type() = CVar::T_STR; 
		ret.StrVal() = "";

		bool bIgnoreCase = false;
		//语法检查
		if (arglist.size() != 2 && arglist.size() != 3 ||
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_INT ||
			arglist.size() == 3 && arglist[2]->Type() != CVar::T_INT)
		{
			fprintf(stderr, "Invalid argument for int substr(string str, int offset, int length)\n");
			return -1;
		}
		if (arglist[1]->IntVal() >= arglist[0]->StrVal().length())
		{
			return 0;
		}
		if (arglist.size() == 2)
		{
			ret.StrVal() = arglist[0]->StrVal().substr(arglist[1]->IntVal());
		}
		else if (arglist.size() == 3)
		{
			ret.StrVal() = arglist[0]->StrVal().substr(arglist[1]->IntVal(), arglist[2]->IntVal());
		}

		return 0;
	}
	else if ("lnb_substitute" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		string flags = "";
		//语法检查
		if (arglist.size() != 3 && arglist.size() != 4 ||
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_STR ||
			arglist[2]->Type() != CVar::T_STR ||
			arglist.size() == 4 && arglist[3]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "Invalid argument for int substitute(string str, string pattern, string to, string flags);\n");
			return -1;
		}
		if (arglist.size() == 4)
		{
			flags = arglist[3]->StrVal();
		}
		perlc_substitute(arglist[0]->StrVal(), arglist[1]->StrVal(), 
			arglist[2]->StrVal(), flags);

		ret.IntVal() = 0;

		return 0;
	}
	else if ("lnb_translate" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		if (arglist.size() != 3 ||
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_STR ||
			arglist[2]->Type() != CVar::T_STR )
			
		{
			fprintf(stderr, "Invalid argument for int translate(string str, string from, string to);\n");
			return -1;
		}
#if 0
		int iLen = arglist[0]->StrVal().length();
		for (int i = 0; i < iLen; ++i)
		{
			char ch = arglist[0]->StrVal()[i];
			string::size_type pos = arglist[1]->StrVal().find(ch);
			if (pos == string::npos)
			{
				continue;
			}
			if (pos >= 0 && pos < arglist[2]->StrVal().length())
			{
				arglist[0]->StrVal()[i] =  arglist[2]->StrVal()[pos];
			}
		}
#else
		perlc_translate(arglist[0]->StrVal(), arglist[1]->StrVal(), arglist[2]->StrVal());
#endif
		ret.IntVal() = 0;

		return 0;
	}
	else if ("lnb_match" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = 0;

		string flags = "";
		//语法检查
		if (arglist.size() != 2 && arglist.size() != 3 ||
			arglist[0]->Type() != CVar::T_STR || 
			arglist[1]->Type() != CVar::T_STR ||
			arglist.size() == 3 && arglist[2]->Type() != CVar::T_STR)
		{
			fprintf(stderr, "Invalid argument for int match(string str, string pattern,  string flags);\n");
			return -1;
		}
		if (arglist.size() == 3)
		{
			flags = arglist[2]->StrVal();
		}
		bool bMatched = false;
		deque<string>  lstBackTrace;
		perlc_match(arglist[0]->StrVal(), arglist[1]->StrVal(),  bMatched, lstBackTrace, flags);

		if (bMatched)
		{
			ret.IntVal() = 1;
		}
		int i;
		for (i = 0; i  < lstBackTrace.size(); ++i)
		{
			CVar part;
			part.Type() = CVar::T_STR;
			part.StrVal() = lstBackTrace[i];

			m_mem.SetArrayVar("$OUTDATA", part, i);
		}

		return 0;
	}
	else if ("lnb_length" == funcname)
	{
		ret.Type() = CVar::T_INT; 
		ret.IntVal() = -1;

		bool bIgnoreCase = false;
		//语法检查
		if (arglist.size() != 1 ||
			arglist[0]->Type() != CVar::T_STR )
		{
			fprintf(stderr, "Invalid argument for int length(string str\n");
			return -1;
		}
		ret.IntVal() = arglist[0]->StrVal().length();

		return 0;
	}
	return -1;
}
