// NonTerminator.h: interface for the CNonTerminator class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NONTERMINATOR_H__F6D13E31_DFBB_4E86_99D7_F3ADFAE43AD1__INCLUDED_)
#define AFX_NONTERMINATOR_H__F6D13E31_DFBB_4E86_99D7_F3ADFAE43AD1__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"


////////////////////////////////////
//���ս��
class CNonTerminator : public CSymbol  
{
public:
	CNonTerminator(const AnsiString& tostring);
	CNonTerminator(const CNonTerminator& another);

	const CNonTerminator& operator=(const CNonTerminator& another);
	bool operator==(const CNonTerminator& another) const;

	virtual ~CNonTerminator();
	virtual const AnsiString  ToString() const;
	virtual int GetSymType() const;

private:
	AnsiString m_sToString;
public:
	CNonTerminator();
	YYLVAL m_yylval;//���ŵ��ۺ�����

};

#endif // !defined(AFX_NONTERMINATOR_H__F6D13E31_DFBB_4E86_99D7_F3ADFAE43AD1__INCLUDED_)
