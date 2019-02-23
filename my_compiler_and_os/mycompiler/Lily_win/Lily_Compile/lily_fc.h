#ifndef __LILY_FC_H__
#define __LILY_FC_H__

extern "C" int fcInit(const char * fname);
extern "C" int fcGetFunc(const char* FuncName, int & argnum, int &number, int &type);

#endif
