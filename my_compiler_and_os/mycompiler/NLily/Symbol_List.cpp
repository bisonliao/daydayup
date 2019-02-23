// Symbol_List.cpp: implementation of the CSymbol_List class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Symbol_List.h"
#include "Dot.h"
#include "Terminator.h"
#include "NonTerminator.h"
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CSymbol_List::CSymbol_List()
{

}

CSymbol_List::~CSymbol_List()
{
	while (!m_list.empty())
	{
		PSYMBOL p = m_list.back();
		m_list.pop_back();
		delete p;
	}
}
bool CSymbol_List::operator==(const CSymbol_List& another) const
{
	if (m_list.size() != another.m_list.size())
	{
		return FALSE;
	}
	PSYMBOL_LIST::const_iterator p1;
	PSYMBOL_LIST::const_iterator p2;
	p1 = m_list.begin();
	PSYMBOL_LIST l = another.m_list;
	p2 = l.begin();
	for ( ;	p1 != m_list.end() && p2 != l.end(); ++p1, ++p2)
	{
	
		if ( (*p1)->GetSymType() != (*p2)->GetSymType())
		{
			return FALSE;
		}
		if ( (*p1)->GetSymType() == SYMBOL_DOT)
		{
			CDot ele1 = *((CDot*)(*p1));
			CDot ele2 = *((CDot*)(*p2));
			if ( ! (ele1 == ele2))
			{
				return FALSE;
			}
		}
		else if ( (*p1)->GetSymType() == SYMBOL_TERMINATOR)
		{
			CTerminator ele1 = *((CTerminator*)(*p1));
			CTerminator ele2 = *((CTerminator*)(*p2));
			if ( ! (ele1 == ele2))
			{
				return FALSE;
			}
		}
		else
		{
			CNonTerminator ele1 = *((CNonTerminator*)(*p1));
			CNonTerminator ele2 = *((CNonTerminator*)(*p2));
			if ( ! (ele1 == ele2))
			{
				return FALSE;
			}
		}	
	}
	
	return TRUE;
}
const CSymbol_List & CSymbol_List::operator=(const CSymbol_List& another)
{
	PSYMBOL_LIST::const_iterator p;

	/*清除现有的符号*/
	while (!m_list.empty())
	{
		PSYMBOL p = m_list.back();
		m_list.pop_back();
		delete p;
	}

	for (p = another.m_list.begin(); p != another.m_list.end(); ++p)
	{
		PSYMBOL ps = *p;
		PSYMBOL ptr = NULL;
		if (ps->GetSymType() == SYMBOL_DOT)
		{
			ptr = new CDot();
		}
		else if (ps->GetSymType() == SYMBOL_TERMINATOR)
		{
			ptr = new CTerminator(ps->ToString());
		}
		else
		{
			ptr = new CNonTerminator(ps->ToString());
		}
		if (NULL == ptr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
		m_list.push_back(ptr);
	}

	return *this;
}

CSymbol_List::CSymbol_List(const CSymbol_List &another)
{
	PSYMBOL_LIST list = another.m_list;
	PSYMBOL_LIST::iterator p;


	for (p = list.begin(); p != list.end(); ++p)
	{
		PSYMBOL ps = *p;
		PSYMBOL ptr = NULL;
		if (ps->GetSymType() == SYMBOL_DOT)
		{
			ptr = new CDot();
		}
		else if (ps->GetSymType() == SYMBOL_TERMINATOR)
		{
			ptr = new CTerminator(ps->ToString());
		}
		else
		{
			ptr = new CNonTerminator(ps->ToString());
			
		}
		if (NULL == ptr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
		m_list.push_back(ptr);
	}
}

void CSymbol_List::push_back(const CSymbol* pele)
{
	assert(pele != NULL);
	PSYMBOL ptr = NULL;
	if (pele->GetSymType() == SYMBOL_DOT)
	{
		ptr = new CDot(*((CDot*)pele));
	}
	else if (pele->GetSymType() == SYMBOL_TERMINATOR)
	{
		ptr = new CTerminator(*((CTerminator*)pele));
	}
	else
	{
		ptr = new CNonTerminator(*((CNonTerminator*)pele));
	}
	if (NULL == ptr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}	
	m_list.push_back(ptr);
} 
 


void CSymbol_List::push_front(const CSymbol* pele)
{
	assert(pele != NULL);
	PSYMBOL ptr = NULL;
	if (pele->GetSymType() == SYMBOL_DOT)
	{
		ptr = new CDot(*((CDot*)pele));
	}
	else if (pele->GetSymType() == SYMBOL_TERMINATOR)
	{
		ptr = new CTerminator(*((CTerminator*)pele));
	}
	else
	{
		ptr = new CNonTerminator(*((CNonTerminator*)pele));
	}
	if (NULL == ptr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}	
	m_list.push_front(ptr);
}

void CSymbol_List::begin_iterator() const
{
	this->m_iterator = m_list.begin();
}

const CSymbol * CSymbol_List::next() const
{
	if (m_iterator == m_list.end())
	{
		return NULL;
	}
	PSYMBOL p = *m_iterator;
	++m_iterator;
	return p;
}

void CSymbol_List::clear()
{
	while (!m_list.empty())
	{
		PSYMBOL p = m_list.back();
		m_list.pop_back();
		delete p;
	} 
}

void CSymbol_List::insert_before(const CSymbol *pele, int index)
{
	assert(pele != NULL);
	if (index < 0)
	{
		this->push_front(pele);
		return;
	}
	if (index + 1 > m_list.size())
	{
		this->push_back(pele);
		return;
	}

	
	PSYMBOL ptr = NULL;
	if (pele->GetSymType() == SYMBOL_DOT)
	{
		ptr = new CDot(*((CDot*)pele));
	}
	else if (pele->GetSymType() == SYMBOL_TERMINATOR)
	{
		ptr = new CTerminator(*((CTerminator*)pele));
	}
	else
	{
		ptr = new CNonTerminator(*((CNonTerminator*)pele));
	}
	if (NULL == ptr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	PSYMBOL_LIST::iterator it = m_list.begin();
	for (int i = 0; i < index; ++i)
	{
		++it;
	}
	m_list.insert(it, ptr);
}

int CSymbol_List::removeAt(int index)
{
	if (index < 0 || index >= m_list.size())
	{
		return -1;
	}
	PSYMBOL_LIST::iterator it = m_list.begin();
	for (int i = 0; i < index; ++i)
	{
		++it;
	}
	m_list.erase(it);
	return 0;
}

int CSymbol_List::size() const
{
	return m_list.size();
}

bool CSymbol_List::contain(const CSymbol *p) const
{
	assert(p != NULL);
	PSYMBOL_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		const CSymbol * ptr = *it;
		if (ptr->GetSymType() != p->GetSymType())
		{
			continue;
		}
		if (ptr->GetSymType() == SYMBOL_DOT)
		{
			return TRUE;
		}
		else if (ptr->GetSymType() == SYMBOL_TERMINATOR)
		{
			if ( (*((CTerminator*)ptr)) == (*((CTerminator*)p)) )
			{
				return TRUE;
			}
		}
		else if (ptr->GetSymType() == SYMBOL_NONTERMINATOR)
		{
			if ( (*((CNonTerminator*)ptr)) == (*((CNonTerminator*)p)) )
			{
				return TRUE;
			}
		}
	}
	return FALSE;
}

void CSymbol_List::addAll(const CSymbol_List &another)
{
	const CSymbol * psym = NULL;
	another.begin_iterator();
	while ( (psym = another.next()) != NULL)
	{
		this->push_back(psym);
	}
}
/*
*删除不必要的ε
*/
void CSymbol_List::trim()
{
	PSYMBOL_LIST::iterator it;
	it = m_list.begin();
	while ( it != m_list.end())
	{
		CSymbol * psym = *it;
		if (psym->GetSymType() == SYMBOL_TERMINATOR &&
			*(CTerminator*)psym == CTerminator::EPSL)
		{
			m_list.erase(it);
			it = m_list.begin();
#ifdef _DEBUG
		//	printf("______________________________________\n");
#endif
			continue;
		}

		++it;
	}
	if (m_list.empty())
	{
		this->push_back(&CTerminator::EPSL);
	}
	
}

const CSymbol * CSymbol_List::GetSymbolAt(int index) const
{
	if (index < 0 || index >= m_list.size())
	{
		return NULL;
	}
	PSYMBOL_LIST::const_iterator it = m_list.begin();
	for (int i = 0; i < index; ++i)
	{
		++it;
	}
	return *it;
}
