// Item_Set.cpp: implementation of the CItem_Set class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Item_Set.h"
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CItem_Set::CItem_Set()
{
	
}
CItem_Set::CItem_Set(const CItem_Set & another)
{
	PITEM_LIST::const_iterator it;
	for (it = another.m_list.begin(); it != another.m_list.end(); ++it)
	{
		PITEM ptr = new CItem(**it);
		if (NULL == ptr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		}
		m_list.push_back(ptr);
	}
}

CItem_Set::~CItem_Set()
{
	while ( !m_list.empty())
	{
		PITEM ptr = m_list.back();
		m_list.pop_back();
		delete ptr;
	}
}

int CItem_Set::insert(PITEM p)
{
	assert(NULL != p);
	
	PITEM_LIST::iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PITEM ptr = *it;
		if ((*ptr) == (*p))	/*已经存在*/
		{
			return 0;
		}
	}
	PITEM ptr = new CItem(*p);
	if (NULL == ptr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
	}
	m_list.push_back(ptr);
	return 0;
}

bool CItem_Set::operator ==(const CItem_Set &another) const
{
	if (m_list.size() != another.m_list.size())
	{
		return FALSE;
	}
	PITEM_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PITEM ptr = *it;
		if ( !(another.contain(*ptr)) )
		{
			return FALSE;
		}
	}
	return TRUE;
}

bool CItem_Set::contain(const CItem &p) const
{
	PITEM_LIST::const_iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		PITEM ptr = *it;
		if ((*ptr) == p)
		{
			return TRUE;
		}
	}
	

	return FALSE;
}

const CItem_Set& CItem_Set::operator =(const CItem_Set &another)
{
	/*清除已有的元素*/
	while ( !m_list.empty())
	{
		PITEM ptr = m_list.back();
		m_list.pop_back();
		delete ptr;
	}
	PITEM_LIST::const_iterator it;
	for (it = another.m_list.begin(); it != another.m_list.end(); ++it)
	{
		CItem * ptr = new CItem(**it);
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

void CItem_Set::begin_iterator() const
{
	m_iterator = m_list.begin();
}

const CItem* CItem_Set::next() const
{
	if (m_iterator == m_list.end())
	{
		return NULL;
	}		
	PITEM ptr = *m_iterator;
	++m_iterator;
	return ptr;
}

int CItem_Set::remove(const CItem &p)
{
	PITEM_LIST::iterator it;
	for (it = m_list.begin(); it != m_list.end(); ++it)
	{
		if ( (**it) == p)
		{
			m_list.erase(it);
			return 0;
		}
	}	
	return -1;
}

int CItem_Set::size() const
{
	return m_list.size();
}



void CItem_Set::clear()
{
	while ( !m_list.empty())
	{
		PITEM ptr = m_list.back();
		m_list.pop_back();
		delete ptr;
	}
}
