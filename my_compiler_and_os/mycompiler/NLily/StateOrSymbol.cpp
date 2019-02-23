// StateOrSymbol.cpp: implementation of the StateOrSymbol class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "StateOrSymbol.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

StateOrSymbol::StateOrSymbol(int nState)
{
	assert(nState >= 0);

	m_nState = nState;
	this->m_pSymbol = NULL;
}

StateOrSymbol::~StateOrSymbol()
{
	if (m_pSymbol != NULL)
	{
		delete m_pSymbol;
		m_pSymbol = NULL;
	}
}



StateOrSymbol::StateOrSymbol(const StateOrSymbol &another)
{
	if (another.m_nState >= 0)
	{
		m_nState = another.m_nState;
		m_pSymbol = NULL;
	}
	else
	{
		m_nState = -1;
		CSymbol* psym = another.m_pSymbol;
		if (psym->GetSymType() == SYMBOL_TERMINATOR)
		{
			m_pSymbol = new CTerminator(*(CTerminator*)psym);
		}
		else
		{
			m_pSymbol = new CNonTerminator(*(CNonTerminator*)psym);
		}
		if (NULL == m_pSymbol)
		{
			fprintf(stderr, "[%s][%d]ƒ⁄¥Ê∑÷≈‰ ß∞‹!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
	}
}

const StateOrSymbol & StateOrSymbol::operator =(const StateOrSymbol &another)
{
	if (m_pSymbol != NULL)
	{
		delete m_pSymbol;
		m_pSymbol = NULL;
	}
	
	if (another.m_nState >= 0)
	{
		m_nState = another.m_nState;
		m_pSymbol = NULL;
	}
	else
	{
		m_nState = -1;
		const CSymbol* psym = another.m_pSymbol;
		if (psym->GetSymType() == SYMBOL_TERMINATOR)
		{
			m_pSymbol = new CTerminator(*(CTerminator*)psym);
		}
		else
		{
			m_pSymbol = new CNonTerminator(*(CNonTerminator*)psym);
		}
		if (NULL == m_pSymbol)
		{
			fprintf(stderr, "[%s][%d]ƒ⁄¥Ê∑÷≈‰ ß∞‹!\n",
				__FILE__,
				__LINE__);
			exit(-1);
		}
	}
	return *this;
}


bool StateOrSymbol::isState() const
{
	return (m_nState >= 0);
}

const CSymbol * StateOrSymbol::getSymbol() const
{	
	return m_pSymbol;
}

int StateOrSymbol::getState() const
{
	return m_nState;
}

StateOrSymbol::StateOrSymbol(const CTerminator &term)
{
	m_nState = -1;
	m_pSymbol = new CTerminator(term);
	if (NULL == m_pSymbol)
	{
		fprintf(stderr, "[%s][%d]ƒ⁄¥Ê∑÷≈‰ ß∞‹!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
}

StateOrSymbol::StateOrSymbol(const CNonTerminator &nt)
{
	m_nState = -1;
	m_pSymbol = new CNonTerminator(nt);
	if (NULL == m_pSymbol)
	{
		fprintf(stderr, "[%s][%d]ƒ⁄¥Ê∑÷≈‰ ß∞‹!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
}
