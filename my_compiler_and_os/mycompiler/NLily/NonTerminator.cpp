// NonTerminator.cpp: implementation of the CNonTerminator class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "NonTerminator.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CNonTerminator::CNonTerminator(const AnsiString& tostring):m_sToString(tostring)
{
}
CNonTerminator::CNonTerminator(const CNonTerminator& another)
{
	m_sToString = another.m_sToString;
	m_yylval = another.m_yylval;
}
const CNonTerminator& CNonTerminator::operator=(const CNonTerminator& another)
{
	m_sToString = another.m_sToString;
	m_yylval = another.m_yylval;
	return *this;
}
bool CNonTerminator::operator==(const CNonTerminator& another) const
{
	return (m_sToString == ((CNonTerminator)another).m_sToString);
}
CNonTerminator::~CNonTerminator()
{

}
int CNonTerminator::GetSymType() const
{
	return SYMBOL_NONTERMINATOR;
}
const AnsiString  CNonTerminator::ToString() const
{
	return m_sToString;
}

CNonTerminator::CNonTerminator()
{

}
