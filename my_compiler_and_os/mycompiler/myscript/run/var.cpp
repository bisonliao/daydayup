#include "var.h"

CVar::CVar()
{
	m_nType = T_INT;
	m_intval = 0;
	m_floatval = 0.0;
	m_stringval = "";
}
CVar::CVar(const CVar& another)
{
 	m_nType = another.m_nType;
 	m_intval = another.m_intval;
 	m_floatval = another.m_floatval;
 	m_stringval = another.m_stringval;
}
const CVar& CVar::operator=(const CVar& another)
{
 	m_nType = another.m_nType;
 	m_intval = another.m_intval;
 	m_floatval = another.m_floatval;
 	m_stringval = another.m_stringval;
	return *this;
}
const string CVar::ToString() const
{
	if (m_nType == T_INT)
	{
		char buf[100];
		sprintf(buf, "%d", m_intval);
		return string(buf);
	}
	else if (m_nType == T_FLOAT)
	{
		char buf[100];
		sprintf(buf, "%f", m_floatval);
		return string(buf);
	}
	else if (m_nType == T_STR)
	{
		string rtnstr = m_stringval;
		int pos;
		while (1)
		{
			pos = rtnstr.find("\\n");
			if (pos == -1)
			{
				break;
			}
			rtnstr.replace(pos, 2, "\n");
		}
		while (1)
		{
			pos = rtnstr.find("\\t");
			if (pos == -1)
			{
				break;
			}
			rtnstr.replace(pos, 2, "\t");
		}
		return rtnstr;
	}
	else
	{
		return string("");
	}
}
CVar operator+(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = lop.m_intval + rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval + rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%d", lop.m_intval);
			result.m_stringval = string(tmpbuf) + rop.m_stringval;
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = rop.m_intval + lop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = rop.m_floatval + lop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%f", lop.m_floatval);
			result.m_stringval = string(tmpbuf) + rop.m_stringval;
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%d", rop.m_intval);
			result.m_stringval = lop.m_stringval + string(tmpbuf);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_STR;
			char tmpbuf[100];
			sprintf(tmpbuf, "%f", rop.m_floatval);
			result.m_stringval = lop.m_stringval + string(tmpbuf);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_STR;
			result.m_stringval = lop.m_stringval + rop.m_stringval;
		}
	}
	return result;
}
CVar operator-(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = lop.m_intval - rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval - rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval - atof(rop.m_stringval.c_str());
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval - rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval - rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval - atof(rop.m_stringval.c_str());
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) - rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) - rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) - atof(rop.m_stringval.c_str());
		}
	}
	return result;
}
CVar operator*(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = lop.m_intval * rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval * rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval * atof(rop.m_stringval.c_str());
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval * rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval * rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval * atof(rop.m_stringval.c_str());
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) * rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) * rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) * atof(rop.m_stringval.c_str());
		}
	}
	return result;
}
CVar operator/(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = lop.m_intval / rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval / rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_intval / atof(rop.m_stringval.c_str());
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval / rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval / rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = lop.m_floatval / atof(rop.m_stringval.c_str());
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) / rop.m_intval;
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) / rop.m_floatval;
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_FLOAT;
			result.m_floatval = atof(lop.m_stringval.c_str()) / atof(rop.m_stringval.c_str());
		}
	}
	return result;
}
CVar operator<(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval < rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval < rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval < atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval < rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval < rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval < atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) < rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) < rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) < atof(rop.m_stringval.c_str()) 
				? 1 : 0);
		}
	}
	return result;
}
CVar operator>(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval > rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval > rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval > atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval > rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval > rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval > atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) > rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) > rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) > atof(rop.m_stringval.c_str()) 
					? 1 : 0);
		}
	}
	return result;
}
CVar operator>=(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval >= rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval >= rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval >= atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval >= rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval >= rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval >= atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) >= rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) >= rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) >= atof(rop.m_stringval.c_str()) 
					? 1 : 0);
		}
	}
	return result;
}
CVar operator<=(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval <= rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval <= rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval <= atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval <= rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval <= rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval <= atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) <= rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) <= rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) <= atof(rop.m_stringval.c_str()) 
					? 1 : 0);
		}
	}
	return result;
}
CVar operator==(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval == rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval == rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval == atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval == rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval == rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval == atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) == rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) == rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_stringval == rop.m_stringval ? 1 : 0);
		}
	}
	return result;
}
CVar operator!=(const CVar & lop, const CVar & rop)
{
	CVar result;
	if (lop.m_nType == CVar::T_INT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval != rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval != rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_intval != atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_FLOAT)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval != rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval != rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_floatval != atof(rop.m_stringval.c_str()) ? 1 : 0);
		}
	}
	else if (lop.m_nType == CVar::T_STR)
	{
		if (rop.m_nType == CVar::T_INT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) != rop.m_intval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_FLOAT)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (atof(lop.m_stringval.c_str()) != rop.m_floatval ? 1 : 0);
		}
		else if (rop.m_nType == CVar::T_STR)
		{
			result.m_nType = CVar::T_INT;
			result.m_intval = (lop.m_stringval != rop.m_stringval ? 1 : 0);
		}
	}
	return result;
}
CVar operator!(const CVar & op)
{
	CVar result;
	if (op.m_nType == CVar::T_INT)
	{
		result.m_nType = CVar::T_INT;
		int tmpint = op.m_intval;
		result.m_intval = !(tmpint);
	}
	else if (op.m_nType == CVar::T_FLOAT)
	{
		result.m_nType = CVar::T_INT;
		int tmpint = op.m_floatval;
		result.m_intval = !(tmpint);
	}
	else if (op.m_nType == CVar::T_STR)
	{
		result.m_nType = CVar::T_INT;
		int tmpint = atoi(op.m_stringval.c_str());
		result.m_intval = !(tmpint);
	}
	return result;
}
