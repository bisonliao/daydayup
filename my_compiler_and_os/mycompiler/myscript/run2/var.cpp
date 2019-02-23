#include "var.h"
#include "util.h"

using namespace lnb;

CVar::CVar() :m_nType(T_INT), m_intval(0), m_floatval(0.0),  m_ptrval(NULL), m_stringval()
{
}
CVar::CVar(const CVar& another) : m_stringval()
{
 	m_nType = another.m_nType;
 	if (m_nType == CVar::T_INT) { m_intval = another.m_intval; return;}
 	if (m_nType == CVar::T_FLOAT) { m_floatval = another.m_floatval; return; }
 	if (m_nType == CVar::T_STR) {StrVal() = another.StrVal(); return; }
	if (m_nType == CVar::T_PTR) {m_ptrval = another.m_ptrval; return;}
}

const CVar& CVar::operator=(const CVar& another)
{
 	Type() = another.Type();
 	if (Type() == CVar::T_INT) { IntVal() = another.IntVal(); return *this;}
 	if (Type() == CVar::T_FLOAT) { FloatVal() = another.FloatVal(); return *this; }
 	if (Type() == CVar::T_STR) {StrVal() = another.StrVal(); return *this; }
	if (Type() == CVar::T_PTR) {m_ptrval = another.m_ptrval; return *this;}
}
const string CVar::ToString() const
{
	if (Type() == T_INT)
	{
		char buf[100];
		sprintf(buf, "%lld", IntVal());
		return string(buf);
	}
	else if (Type() == T_FLOAT)
	{
		char buf[100];
		sprintf(buf, "%f", FloatVal());
		return string(buf);
	}
	else if (Type() == T_STR)
	{
		string rtnstr ;
		StringUnescape(StrVal(), rtnstr);
		return rtnstr;
	}
	else
	{
		return string("");
	}
}
CVar &CVar::operator*()
{
	if (Type() == CVar::T_PTR)
	{
		return *(m_ptrval);
	}
	else
	{
		return *(this);
	}
}
CVar lnb::operator+(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Add(lop, rop, result);
	return result;
}
void CVar::Add(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = lop.IntVal() + rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() + rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%lld", lop.IntVal());
			result.StrVal() = string(tmpbuf) + rop.StrVal();
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = rop.IntVal() + lop.FloatVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = rop.FloatVal() + lop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%f", lop.FloatVal());
			result.StrVal() = string(tmpbuf) + rop.StrVal();
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%lld", rop.IntVal());
			result.StrVal() = lop.StrVal() + string(tmpbuf);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%f", rop.FloatVal());
			result.StrVal() = lop.StrVal() + string(tmpbuf);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_STR;
			result.StrVal() = lop.StrVal() + rop.StrVal();
		}
	}
}
CVar lnb::operator-(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Sub(lop, rop, result);
	return result;
}
CVar lnb::operator-(const CVar & op)
{
	CVar result;
	CVar::Minus(op, result);
	return result;
}
void CVar::Sub(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = lop.IntVal() - rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() - rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() - atof(rop.StrVal().c_str());
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() - rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() - rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() - atof(rop.StrVal().c_str());
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) - rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) - rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) - atof(rop.StrVal().c_str());
		}
	}
}
void CVar::Minus(const CVar & op, CVar & result)
{
	if (op.Type() == CVar::T_INT)
	{
		result.Type() = CVar::T_INT;
		result.IntVal() = -( op.IntVal() );
	}
	else if (op.Type() == CVar::T_FLOAT)
	{
		result.Type() = CVar::T_FLOAT;
		result.FloatVal() = -( op.FloatVal() );
	}
	else if (op.Type() == CVar::T_STR)
	{
		result.Type() = CVar::T_FLOAT;
		result.FloatVal() = - atof(op.StrVal().c_str());
	}
}
CVar lnb::operator*(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Mul(lop, rop, result);
	return result;
}
void CVar::Mul(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = lop.IntVal() * rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() * rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() * atof(rop.StrVal().c_str());
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() * rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() * rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() * atof(rop.StrVal().c_str());
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) * rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) * rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) * atof(rop.StrVal().c_str());
		}
	}
}
CVar lnb::operator/(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Mul(lop, rop, result);
	return result;
}
void CVar::Div(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = lop.IntVal() / rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() / rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.IntVal() / atof(rop.StrVal().c_str());
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() / rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() / rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = lop.FloatVal() / atof(rop.StrVal().c_str());
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) / rop.IntVal();
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) / rop.FloatVal();
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_FLOAT;
			result.FloatVal() = atof(lop.StrVal().c_str()) / atof(rop.StrVal().c_str());
		}
	}
}
CVar lnb::operator<(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Lt(lop, rop, result);
	return result;
}
void CVar::Lt(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() < rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() < rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() < atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() < rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() < rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() < atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) < rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) < rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) < atof(rop.StrVal().c_str()) 
				? 1 : 0);
		}
	}
}
CVar lnb::operator>(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Gt(lop, rop, result);
	return result;
}
void CVar::Gt(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() > rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() > rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() > atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() > rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() > rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() > atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) > rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) > rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) > atof(rop.StrVal().c_str()) 
					? 1 : 0);
		}
	}
}
CVar lnb::operator>=(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Ge(lop, rop, result);
	return result;
}
void CVar::Ge(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() >= rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() >= rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() >= atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() >= rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() >= rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() >= atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) >= rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) >= rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) >= atof(rop.StrVal().c_str()) 
					? 1 : 0);
		}
	}
}
CVar lnb::operator<=(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Le(lop, rop, result);
	return result;
}
void CVar::Le(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() <= rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() <= rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() <= atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() <= rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() <= rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() <= atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) <= rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) <= rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) <= atof(rop.StrVal().c_str()) 
					? 1 : 0);
		}
	}
}
CVar lnb::operator==(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Eq(lop, rop, result);
	return result;
}
void CVar::Eq(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() == rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() == rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() == atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() == rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() == rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() == atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) == rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) == rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.StrVal() == rop.StrVal() ? 1 : 0);
		}
	}
}
CVar lnb::operator!=(const CVar & lop, const CVar & rop)
{
	CVar result;
	CVar::Ne(lop, rop, result);
	return result;
}
void CVar::Ne(const CVar & lop, const CVar & rop, CVar & result)
{
	if (lop.Type() == CVar::T_INT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() != rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() != rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.IntVal() != atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_FLOAT)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() != rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() != rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.FloatVal() != atof(rop.StrVal().c_str()) ? 1 : 0);
		}
	}
	else if (lop.Type() == CVar::T_STR)
	{
		if (rop.Type() == CVar::T_INT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) != rop.IntVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_FLOAT)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (atof(lop.StrVal().c_str()) != rop.FloatVal() ? 1 : 0);
		}
		else if (rop.Type() == CVar::T_STR)
		{
			result.Type() = CVar::T_INT;
			result.IntVal() = (lop.StrVal() != rop.StrVal() ? 1 : 0);
		}
	}
}
CVar lnb::operator!(const CVar & op)
{
	CVar result;
	CVar::Not(op,  result);
	return result;
}
void CVar::Not(const CVar & op, CVar & result)
{
	if (op.Type() == CVar::T_INT)
	{
		result.Type() = CVar::T_INT;
		long long tmpint = op.IntVal();
		result.IntVal() = !(tmpint);
	}
	else if (op.Type() == CVar::T_FLOAT)
	{
		result.Type() = CVar::T_INT;
		long long tmpint = op.FloatVal();
		result.IntVal() = !(tmpint);
	}
	else if (op.Type() == CVar::T_STR)
	{
		result.Type() = CVar::T_INT;
		long long tmplint = atoll(op.StrVal().c_str());
		result.IntVal() = !(tmplint);
	}
}
int CVar::FormatStr(const deque<CVar*> &arglist, string & ss)
{
	list<int> startpos, endpos;
	if (!re(arglist[0]->StrVal(), "%[0-9\\.\\-]*(s|e|d|c|l|u|x|f|(lld)|(ld)|(lu))", true,  startpos,  endpos))
	{
		fprintf(stderr, "failed to analyze format string!"); 
		return -1;
	}

	list<int>::const_iterator it1, it2;
	int iNonFmtStart = 0;
	ss = "";
	int iArgPos = 1;
	int iRet = 0;
	static char s_acStrBuf[100*1024];
	for (it1 = startpos.begin(), it2 = endpos.begin();
		it1 != startpos.end(), it2 != endpos.end();
		++it1, ++it2, ++iArgPos)
	{
		//不是格式化的部分
		int iNonFmtLen = *it1 - iNonFmtStart;
		ss += arglist[0]->StrVal().substr(iNonFmtStart, iNonFmtLen);

		if (iArgPos >= arglist.size())
		{
			break;
		}

		//替换格式化部分
		string fmt = arglist[0]->StrVal().substr(*it1, *it2 - *it1);
		char cc =fmt[fmt.length()-1];
		++iRet;
		if ('s' == cc)
		{
			snprintf(s_acStrBuf, sizeof(s_acStrBuf), fmt.c_str(), arglist[iArgPos]->StrVal().c_str());
		}
		else if ('f' == cc)
		{
			snprintf(s_acStrBuf, sizeof(s_acStrBuf), fmt.c_str(), arglist[iArgPos]->FloatVal());
		}
		else
		{
			snprintf(s_acStrBuf, sizeof(s_acStrBuf), fmt.c_str(), arglist[iArgPos]->IntVal());
		}
		ss += string(s_acStrBuf);
		
		iNonFmtStart = *it2;
	}
	if (iNonFmtStart < arglist[0]->StrVal().length())
	{
		ss += arglist[0]->StrVal().substr(iNonFmtStart);
	}
	return iRet;
}
