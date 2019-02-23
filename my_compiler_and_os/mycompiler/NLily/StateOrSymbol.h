// StateOrSymbol.h: interface for the StateOrSymbol class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_STATEORSYMBOL_H__6FCB44BE_7D86_48D9_8AAE_AE0CFA57B98D__INCLUDED_)
#define AFX_STATEORSYMBOL_H__6FCB44BE_7D86_48D9_8AAE_AE0CFA57B98D__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"
#include "Terminator.h"
#include "NonTerminator.h"
#include "Dot.h"

////////////////////////////////
//一个状态标号或者是一个文法符号
class StateOrSymbol  
{
public:
	StateOrSymbol(const CNonTerminator& nt);
	StateOrSymbol(const CTerminator& term);
	int getState() const;
	const CSymbol * getSymbol() const;
	bool isState() const;
	const StateOrSymbol & operator=(const StateOrSymbol&another);
	StateOrSymbol(const StateOrSymbol & another);
	StateOrSymbol(int nState);
	virtual ~StateOrSymbol();
private:
	int m_nState;
	CSymbol * m_pSymbol;

};

#endif // !defined(AFX_STATEORSYMBOL_H__6FCB44BE_7D86_48D9_8AAE_AE0CFA57B98D__INCLUDED_)
