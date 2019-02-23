// Terminator.h: interface for the CTerminator class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_TERMINATOR_H__C2CB560B_03B9_4693_AAC5_3D63BB90375B__INCLUDED_)
#define AFX_TERMINATOR_H__C2CB560B_03B9_4693_AAC5_3D63BB90375B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"

///////////////////////////////////
//终结符
class CTerminator : public CSymbol  
{
public:
	CTerminator(const AnsiString& tostring);
	CTerminator(const CTerminator& another);

	bool operator==(const CTerminator& another) const;
	const CTerminator& operator=(const CTerminator& another);


	virtual ~CTerminator();
	virtual int GetSymType() const;
	const AnsiString  ToString() const;

private:
	AnsiString m_sToString;
public:
	CTerminator();
	static CTerminator FINIS;
	static CTerminator EPSL;
	YYLVAL m_yylval;//符号的综合属性

};


#endif // !defined(AFX_TERMINATOR_H__C2CB560B_03B9_4693_AAC5_3D63BB90375B__INCLUDED_)
