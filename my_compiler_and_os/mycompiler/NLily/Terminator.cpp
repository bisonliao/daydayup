// Terminator.cpp: implementation of the CTerminator class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Terminator.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CTerminator CTerminator::FINIS("¦Õ");
CTerminator CTerminator::EPSL("¦Å");

CTerminator::CTerminator(const AnsiString& tostring):m_sToString(tostring)
{
}
CTerminator::CTerminator(const CTerminator& another)
{
	m_sToString = another.m_sToString;
	m_yylval = another.m_yylval;
}
bool CTerminator::operator==(const CTerminator& another) const
{
	return (m_sToString == another.m_sToString);
}

const CTerminator& CTerminator::operator=(const CTerminator& another)
{
	m_sToString = another.m_sToString;
	m_yylval = another.m_yylval;
	return *this;
}

CTerminator::~CTerminator()
{

}
int CTerminator::GetSymType() const
{
	return SYMBOL_TERMINATOR;
}
const AnsiString  CTerminator::ToString() const
{
	return m_sToString;
}

CTerminator::CTerminator()
{

}
