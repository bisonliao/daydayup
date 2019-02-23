// Producer.cpp: implementation of the CProducer class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Producer.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CProducer::CProducer(const CNonTerminator &left, 
					 const CSymbol_List& right,
					 PRODUCER_FUNC func)
	:m_left(left),m_right(right),m_func(func)
{
}
CProducer::CProducer(const CProducer& another)
	:m_left(another.m_left), m_right(another.m_right), m_func(another.m_func)
{
}

const CProducer& CProducer::operator=(const CProducer& another)
{
	m_left = another.m_left;
	m_right = another.m_right;
	m_func = another.m_func;
	return *this;
}
bool CProducer::operator==(const CProducer& another) const
{
	return (m_left == another.m_left && m_right == another.m_right);
}

CProducer::~CProducer()
{

}

int CProducer::GetDotIndex() const
{	
	m_right.begin_iterator();
	const CSymbol* p ;
	int index = -1;
	while ( (p = m_right.next()) != NULL)
	{
		++index;
		if (p->GetSymType() == SYMBOL_DOT)
		{
			return index;
		}
	}
	return -1;
}

const AnsiString  CProducer::ToString() const
{
	AnsiString s;
	s.concat(m_left.ToString());
	s.concat("::=");

	m_right.begin_iterator();
	const CSymbol * p ;
	while ( (p = m_right.next()) != NULL)
	{
		s.concat(p->ToString());
		s.concat(" ");
	}	
	return s;
}

const CSymbol* CProducer::GetSymbolAt(int index) const
{
	if (index < 0 || index >= m_right.size())
	{
		return NULL;
	}
	m_right.begin_iterator();
	for (int i = 0; i < index; ++i)
	{
		m_right.next();
	}
	return m_right.next();
}

const CNonTerminator& CProducer::GetLeft() const
{
	return m_left;
}

const CSymbol_List& CProducer::GetRight() const
{
	return m_right;
}

PRODUCER_FUNC CProducer::GetFunc() const
{
	return m_func;
}

void CProducer::SetLeft(const CNonTerminator &left)
{
	m_left = left;
}

void CProducer::SetRight(const CSymbol_List &right)
{
	m_right = right;
}

void CProducer::SetFunc(PRODUCER_FUNC func)
{
	m_func = func;
}
