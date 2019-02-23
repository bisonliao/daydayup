#ifndef __RUN__H__INCLUDED__
#define __RUN__H__INCLUDED__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <deque>
#include "script.h"
using namespace std;

namespace lnb{

int RunScript(const string & scriptname, const deque<CVar*> &args, CVar & rtnval);
int Run(const deque<string> & PCodeList, const deque<string> & arrArgs );
int ClearPCode();
int AddPCode(const deque<string> & PCodeList );
int AddExternFunction(const string & funcname, EXTERN_FUNC_PTR fptr);
}

#endif
