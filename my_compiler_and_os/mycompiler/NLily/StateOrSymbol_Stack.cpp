// StateOrSymbol_Stack.cpp: implementation of the StateOrSymbol_Stack class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "StateOrSymbol_Stack.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

StateOrSymbol_Stack::StateOrSymbol_Stack()
{

}

StateOrSymbol_Stack::~StateOrSymbol_Stack()
{
	PSTATEORSYMBOL ptr = NULL;
	while (m_stack.size() > 0)
	{
		ptr = m_stack.top();
		m_stack.pop();
		delete ptr;
	}
}

int StateOrSymbol_Stack::size() const
{
	return m_stack.size();
}

int StateOrSymbol_Stack::peek(StateOrSymbol &ele)
{
	if (m_stack.size() <= 0)
	{
		return -1;
	}
	PSTATEORSYMBOL ptr = m_stack.top();
	ele = *ptr;
	return 0;
}

void StateOrSymbol_Stack::pop()
{
	PSTATEORSYMBOL ptr = NULL;
	if (m_stack.size() > 0)
	{
		ptr = m_stack.top();
	}
	m_stack.pop();
	delete ptr;
}

void StateOrSymbol_Stack::push(const StateOrSymbol &ele)
{
	PSTATEORSYMBOL ptr = new StateOrSymbol(ele);
	if (NULL == ptr)
	{
		fprintf(stderr, "[%s][%d]∑÷≈‰ƒ⁄¥Ê ß∞‹!\n",
				__FILE__,
				__LINE__);
		exit(-1);
	}
	m_stack.push(ptr);
}
