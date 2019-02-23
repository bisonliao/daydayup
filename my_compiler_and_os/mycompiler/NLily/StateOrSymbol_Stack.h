// StateOrSymbol_Stack.h: interface for the StateOrSymbol_Stack class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_STATEORSYMBOL_STACK_H__00797F10_E902_4F81_86E7_F0ABB6EC650B__INCLUDED_)
#define AFX_STATEORSYMBOL_STACK_H__00797F10_E902_4F81_86E7_F0ABB6EC650B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"
#include "Terminator.h"
#include "NonTerminator.h"
#include "Dot.h"
#include <stack>
#include "StateOrSymbol.h"

using namespace std;

typedef StateOrSymbol* PSTATEORSYMBOL;
typedef stack<PSTATEORSYMBOL> PSTATEORSYMBOL_STACK;

class StateOrSymbol_Stack  
{
public:
	void push(const StateOrSymbol & ele);
	void pop();
	int peek(StateOrSymbol & ele);
	int size() const;
	StateOrSymbol_Stack();
	virtual ~StateOrSymbol_Stack();
private:
	PSTATEORSYMBOL_STACK m_stack;

};

#endif // !defined(AFX_STATEORSYMBOL_STACK_H__00797F10_E902_4F81_86E7_F0ABB6EC650B__INCLUDED_)
