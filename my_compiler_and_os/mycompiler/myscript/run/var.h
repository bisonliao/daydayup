#ifndef _VAR_H_INCLUDED_
#define _VAR_H_INCLUDED_

#include <string>
using namespace std;

class CVar
{
public:
	//变量的类型可取的值
	enum {T_INT, T_FLOAT, T_STR};

	int m_nType;

	int m_intval;
	double m_floatval;
	string m_stringval;


	CVar();
	CVar(const CVar& another);
	const CVar& operator=(const CVar& another);
	const string ToString() const;

};
CVar operator+(const CVar & lop, const CVar & rop);
CVar operator-(const CVar & lop, const CVar & rop);
CVar operator*(const CVar & lop, const CVar & rop);
CVar operator/(const CVar & lop, const CVar & rop);
CVar operator<(const CVar & lop, const CVar & rop);
CVar operator<=(const CVar & lop, const CVar & rop);
CVar operator>(const CVar & lop, const CVar & rop);
CVar operator>=(const CVar & lop, const CVar & rop);
CVar operator==(const CVar & lop, const CVar & rop);
CVar operator!=(const CVar & lop, const CVar & rop);
CVar operator!(const CVar & op);

#endif
