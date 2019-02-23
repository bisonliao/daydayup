// Symbol.h: interface for the CSymbol class.
// 
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SYMBOL_H__8CEB7287_08A3_4EAF_A38D_C4734350F0F9__INCLUDED_)
#define AFX_SYMBOL_H__8CEB7287_08A3_4EAF_A38D_C4734350F0F9__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "common.h"
#include "AnsiString.h"

	

class CSymbol  
{
public:
	CSymbol();
	virtual ~CSymbol();
	virtual int GetSymType() const  = 0;
	virtual const AnsiString  ToString()const = 0;

};

#endif // !defined(AFX_SYMBOL_H__8CEB7287_08A3_4EAF_A38D_C4734350F0F9__INCLUDED_)
