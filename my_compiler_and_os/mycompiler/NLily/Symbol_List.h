// Symbol_List.h: interface for the CSymbol_List class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SYMBOL_LIST_H__05C661F5_0308_41CD_AF62_2452A55B2B44__INCLUDED_)
#define AFX_SYMBOL_LIST_H__05C661F5_0308_41CD_AF62_2452A55B2B44__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"
#include "common.h"
#include <list>
#include "Terminator.h"
#include "NonTerminator.h"
#include "Dot.h"

using namespace std;  

typedef CSymbol* PSYMBOL;
typedef list<PSYMBOL> PSYMBOL_LIST;

class CSymbol_List  
{
public:
	const CSymbol * GetSymbolAt(int index) const;
	void trim();
	void addAll(const CSymbol_List & another);
	bool contain(const CSymbol* p) const;
	int size() const;
	int removeAt(int index);
	void insert_before(const CSymbol* pele, int index);
	void clear();
	const CSymbol * next() const;
	void begin_iterator() const;
	void push_back(const CSymbol* pele);
	void push_front(const CSymbol* pele);
	CSymbol_List(const CSymbol_List& another);
	CSymbol_List();
	virtual ~CSymbol_List();

	bool operator==(const CSymbol_List& another) const;
	const CSymbol_List & operator=(const CSymbol_List& another);


	
private:
	PSYMBOL_LIST m_list;
	mutable PSYMBOL_LIST::const_iterator m_iterator;

};

#endif // !defined(AFX_SYMBOL_LIST_H__05C661F5_0308_41CD_AF62_2452A55B2B44__INCLUDED_)
