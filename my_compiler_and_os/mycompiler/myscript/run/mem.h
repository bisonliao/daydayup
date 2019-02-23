#ifndef _MEM_H_INCLUDED_
#define _MEM_H_INCLUDED_
/*
*实现按名字存取变量
*内部基于map实现. map的每个元素是一个deque, 通常情况下,deque只有一个元素,
*就是需要存取的变量,但是如果变量是一个数组元素,那么deque就包含了一个数组
*一个数组最大可以有2048个元素
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
