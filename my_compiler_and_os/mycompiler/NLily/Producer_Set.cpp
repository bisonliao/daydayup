// Producer_Set.cpp: implementation of the CProducer_Set class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Producer_Set.h"
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CProducer_Set::CProducer_Set()
{
	
}
CProducer_Set::CProducer_Set(const CProducer_Set & another)
{
	PPRODUCER_LIST::const_iterator it;
	for (it = another.m_list.begin(); it != another.m_list.end(); ++it)
	{
		PPRODUCER ptr = new CProducer(**it);
		if (NULL == ptr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		}
		m_list.push_back(ptr);
	}
}

CProducer_Set::~CProducer_Set()
{
	while ( !m_list.empty())
	{
		PPRODUCER ptr = m_list.back();
		m_list.pop_back();
		delete ptr;
	}
}

int CProducer_Set::insert(PPRODUCER p)
{
	assert(NULL != p);
	
	PPRODUCER_LIST::iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ((*ptr) == (*p))	/*已经存在*/
		{
			return 0;
		}
	}
	PPRODUCER ptr = new CProducer(*p);
	if (NULL == ptr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
	}
	m_list.push_back(ptr);
	return 0;
}

bool CProducer_Set::operator ==(const CProducer_Set &another) const
{
	if (m_list.size() != another.m_list.size())
	{
		return FALSE;
	}
	PPRODUCER_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ( !(another.contain(*ptr)) )
		{
			return FALSE;
		}
	}
	return TRUE;
}

bool CProducer_Set::contain(const CProducer &p) const
{
	PPRODUCER_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ((*ptr) == p)
		{
			return TRUE;
		}
	}
	

	return FALSE;
}

const CProducer_Set& CProducer_Set::operator =(const CProducer_Set &another)
{
	/*清除已有的元素*/
	while ( !m_list.empty())
	{
		PPRODUCER ptr = m_list.back();
		m_list.pop_back();
		delete ptr;
	}
	PPRODUCER_LIST::const_iterator it;
	for (it = another.m_list.begin(); it != another.m_list.end(); ++it)
	{
		CProducer * ptr = new CProducer(**it);
		if (NULL == ptr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		}
		m_list.push_back(ptr);
	}
	return *this;
}

void CProducer_Set::begin_iterator() const
{
	m_iterator = m_list.begin();
}

const CProducer* CProducer_Set::next() const
{
	if (m_iterator == m_list.end())
	{
		return NULL;
	}		
	PPRODUCER ptr = *m_iterator;
	++m_iterator;
	return ptr;
}

int CProducer_Set::remove(const CProducer &p)
{
	PPRODUCER_LIST::iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ( (*ptr) == p)
		{
			m_list.erase(it);
			delete ptr;
			return 0;
		}
	}	
	return -1;
}

int CProducer_Set::size() const
{
	return m_list.size();
}

//DEL CProducer_Set CProducer_Set::closure(const CProducer_Set &I, const CProducer_Set & G)
//DEL {
//DEL 	CProducer_Set ret;
//DEL 	ret = I;
//DEL 
//DEL 	
//DEL 
//DEL 	//repeat
//DEL 	bool bNewItemAdded;
//DEL 	do
//DEL 	{
//DEL 		bNewItemAdded = FALSE;
//DEL 		const CProducer_Set J = ret;	//J是每次循环的时候 ret的一个 只读拷贝
//DEL 		J.begin_iterator();
//DEL 		const CProducer * p1 = NULL;
//DEL 		const CProducer * p2 = NULL;
//DEL 		while ( (p1 = J.next()) != NULL)	/*for J的每个项目A->α.Bβ*/
//DEL 		{
//DEL 			int dotIndex = p1->GetDotIndex();
//DEL 			const CSymbol * B = p1->GetSymbolAt(dotIndex + 1);
//DEL 			if (B == NULL)
//DEL 			{
//DEL 				continue;
//DEL 			}
//DEL 			if (B->GetSymType() != SYMBOL_NONTERMINATOR)
//DEL 			{
//DEL 				continue;
//DEL 			}
//DEL 			G.begin_iterator();
//DEL 			while ( (p2 = G.next()) != NULL)	/*for G的每个产生式B->γ*/
//DEL 			{
//DEL 				CNonTerminator left = p2->GetLeft();
//DEL 				if ( !( *((CNonTerminator*)B) == left )  )
//DEL 				{
//DEL 					continue;
//DEL 				}
//DEL 				/*	加入B->.γ	*/
//DEL 				CSymbol_List right = p2->GetRight();
//DEL 				PRODUCER_FUNC func = p2->GetFunc();
//DEL 				CDot dot;
//DEL 				right.push_front(&dot);
//DEL 				CProducer newitem(left, right, func);
//DEL 				if (!ret.contain(newitem))
//DEL 				{
//DEL 					ret.insert(&newitem);
//DEL 					bNewItemAdded = TRUE;
//DEL 				}
//DEL 			}
//DEL 		}
//DEL 	}
//DEL 	while (bNewItemAdded);
//DEL 	//until没有新项目可加入J;
//DEL 	return ret;
//DEL }

int CProducer_Set::GetProducerIndex(const CProducer &prd) const
{
	PPRODUCER_LIST::const_iterator it;
	int index;
	for (index = 0, it = m_list.begin(); 
		it != m_list.end(); 
		++index, ++it)
	{
		if ( (**it) == prd)
		{
			return index;
		}
	}	
	return -1;
}

void CProducer_Set::clear()
{
	while ( !m_list.empty())
	{
		PPRODUCER ptr = m_list.back();
		m_list.pop_back();
		delete ptr;
	}
}

/*
*删除非终结符nt的所有产生式
*/
void CProducer_Set::RmPrdcOfNonTerminator(const CNonTerminator &nt)
{
	PPRODUCER_LIST::iterator it = m_list.begin();
	while (it != m_list.end())
	{
		PPRODUCER ptr = *it;
		if ( (ptr->GetLeft()) == nt)
		{
			m_list.erase(it);
			delete ptr;
			it = m_list.begin();
			continue;
		}
		++it;
	}
}


/*
*得到非终结符nt的所有直接左递归产生式
*/
CProducer_Set CProducer_Set::GetDirectRcrsPrdcOfNonTerm(const CNonTerminator &nt) const
{
	CProducer_Set ret;
	PPRODUCER_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ( (ptr->GetLeft()) == nt)
		{
			CSymbol_List right = ptr->GetRight();
			right.begin_iterator();
			const CSymbol* sp = right.next();
			if (sp->GetSymType() == SYMBOL_NONTERMINATOR &&
				(*(CNonTerminator*)sp) == nt)
			{
#ifdef _DEBUG
	CNonTerminator ntnt = *(CNonTerminator*)sp;
#endif
				ret.insert(ptr);
			}
		}
	}
	return ret;
}

CProducer_Set CProducer_Set::GetNonRcrsPrdcOfNonTerm(const CNonTerminator &nt) const
{
	CProducer_Set ret;
	PPRODUCER_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ( (ptr->GetLeft()) == nt)
		{
			CSymbol_List right = ptr->GetRight();
			right.begin_iterator();
			const CSymbol* sp = right.next();
			if (sp->GetSymType() != SYMBOL_NONTERMINATOR ||
				sp->GetSymType() == SYMBOL_NONTERMINATOR &&	!((*(CNonTerminator*)sp) == nt) )
			{
				ret.insert(ptr);
			}
		}
	}
	return ret;	
}

CProducer_Set CProducer_Set::GetProducerOfNonTerm(const CNonTerminator &nt) const
{
	CProducer_Set ret;
	PPRODUCER_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PPRODUCER ptr = *it;
		if ( (ptr->GetLeft()) == nt)
		{
			ret.insert(ptr);
		}
	}
	return ret;
}

/*
*删除所有形如Ai->Aj γ的产生式
*/
CProducer_Set CProducer_Set::RmAiAjProducer(const CNonTerminator &Ai, const CNonTerminator &Aj)
{
	CProducer_Set ret; 

	PPRODUCER_LIST::iterator it = m_list.begin();
	while (it != m_list.end())
	{
		PPRODUCER ptr = *it;
		if ( (ptr->GetLeft()) == Ai)
		{
			CSymbol_List right = ptr->GetRight();
			right.begin_iterator();
			const CSymbol* pAj = right.next();
			if ( pAj->GetSymType() == SYMBOL_NONTERMINATOR &&
				(*(CNonTerminator*)pAj) == Aj) 
			{
				m_list.erase(it);
				ret.insert(ptr);
				delete ptr;
				it = m_list.begin();
				continue;
			}
		}

		++it;
	}
	return ret;
}

const CProducer* CProducer_Set::GetProducerAt(int index) const
{
	if (index < 0 || (index + 1) > m_list.size())
	{
		return NULL;
	}
	PPRODUCER_LIST::const_iterator it = m_list.begin();
	int i = 0;
	while (i < index)
	{
		++it;
		++i;
	}
	return *it;
}
