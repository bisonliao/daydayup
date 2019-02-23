#ifndef _MEM_H_INCLUDED_
#define _MEM_H_INCLUDED_
/*
*ʵ�ְ����ִ�ȡ����
*�ڲ�����mapʵ��. map��ÿ��Ԫ����һ��deque, ͨ�������,dequeֻ��һ��Ԫ��,
*������Ҫ��ȡ�ı���,�������������һ������Ԫ��,��ôdeque�Ͱ�����һ������
*һ��������������2048��Ԫ��
*/

#include <map>
#include <string>
#include <deque>
using namespace std;
#include "var.h"

class CMem
{
public:
	enum { ARRAY_MAX_SIZE=2048 };
private:
	map<string, deque<CVar> > m_map;
public:
	void SetVar(const string &varname,
				CVar & var, unsigned short subscript=0);
	void GetVar(const string &varname,
				CVar & var, unsigned short subscript=0);
};

#endif
