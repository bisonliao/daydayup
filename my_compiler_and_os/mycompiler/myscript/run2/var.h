#ifndef _VAR_H_INCLUDED_
#define _VAR_H_INCLUDED_

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <deque>
using namespace std;

#include <new>

namespace lnb {

class CVar
{
public:
	//变量的类型可取的值  T_PTR为内部使用
	enum {T_INT, T_FLOAT, T_STR, T_PTR};


	CVar();
	CVar(const CVar& another);
	const CVar& operator=(const CVar& another);
	const string ToString() const;

	CVar & operator*();

	static int FormatStr(const deque<CVar*> &arglist, string & ss);

	inline string & StrVal() { return m_stringval; }
	inline const string & StrVal() const { return m_stringval; }

	inline long long & IntVal() { return m_intval;}
	inline const long long & IntVal() const { return m_intval;}

	inline double & FloatVal() { return m_floatval;}
	inline const double & FloatVal() const { return m_floatval;}

	inline int & Type() {return m_nType;}
	inline const int & Type() const {return m_nType;}

	inline CVar* & PtrVal() {return m_ptrval;}
	inline CVar* const  & PtrVal() const {return m_ptrval;}

	static void Add(const CVar & lop, const CVar & rop, CVar& result);
	static void Sub(const CVar & lop, const CVar & rop, CVar& result);
	static void Minus(const CVar & op, CVar& result);
	static void Mul(const CVar & lop, const CVar & rop, CVar& result);
	static void Div(const CVar & lop, const CVar & rop, CVar& result);
	static void Lt(const CVar & lop, const CVar & rop, CVar& result);
	static void Gt(const CVar & lop, const CVar & rop, CVar& result);
	static void Ge(const CVar & lop, const CVar & rop, CVar& result);
	static void Le(const CVar & lop, const CVar & rop, CVar& result);
	static void Eq(const CVar & lop, const CVar & rop, CVar& result);
	static void Ne(const CVar & lop, const CVar & rop, CVar& result);
	static void Not(const CVar & op,  CVar& result);

	bool less(const CVar & v) const
	{
		if (v.Type() != this->Type())
		{
			return this->Type() < v.Type();
		}
		if (v.Type() == T_INT)
		{
			return this->IntVal() < v.IntVal();
		}
		else if (v.Type() == T_STR)
		{
			return this->StrVal() < v.StrVal();
		}
		else if (v.Type() == T_FLOAT)
		{
			return this->FloatVal() < v.FloatVal();
		}
		else if (v.Type() == T_PTR)
		{
			fprintf(stderr, "%s %d: logic error!!!\n", __FILE__, __LINE__);
			exit(-1);
		}
	}
	static void getTheLeast(CVar & v)
	{
		v.Type() =  T_INT;
		v.IntVal() = LONG_LONG_MIN;
		//v.IntVal() = LLONG_MIN;
	};
	static void getTheMost(CVar & v)
	{
		v.Type() =  T_STR;
		v.StrVal() = string(1u, (unsigned char)UCHAR_MAX);
	};

private:
	int m_nType;

	long long m_intval;
	double m_floatval;
	string  m_stringval;
	CVar * m_ptrval;


};
template<class T>
class VarLess
{
public:
	bool operator()(const T & a, const T & b) const
	{
		return a.less(b);
	}
};

CVar operator+(const CVar & lop, const CVar & rop);
CVar operator-(const CVar & lop, const CVar & rop);
CVar operator-(const CVar & op);
CVar operator*(const CVar & lop, const CVar & rop);
CVar operator/(const CVar & lop, const CVar & rop);
CVar operator<(const CVar & lop, const CVar & rop);
CVar operator<=(const CVar & lop, const CVar & rop);
CVar operator>(const CVar & lop, const CVar & rop);
CVar operator>=(const CVar & lop, const CVar & rop);
CVar operator==(const CVar & lop, const CVar & rop);
CVar operator!=(const CVar & lop, const CVar & rop);
CVar operator!(const CVar & op);
};

#endif
