// Item_Set.h: interface for the CItem_Set class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(__ITEM_SET_H_INCLUDED__)
#define __ITEM_SET_H_INCLUDED__

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Item.h"
#include <list>

typedef CItem* PITEM;
typedef list<PITEM> PITEM_LIST;

////////////////////////////
//项目的集合
class CItem_Set  
{
public:
	void clear();
	int size() const;
	int remove(const CItem& p);
	const CItem* next() const;
	void begin_iterator() const;
	bool contain(const CItem & p) const;
	int insert(PITEM p);
	bool operator ==(const CItem_Set &another) const;
	const CItem_Set & operator=(const CItem_Set &another);
	CItem_Set();
	CItem_Set(const CItem_Set & another);
	virtual ~CItem_Set();


private:
	PITEM_LIST m_list;
	mutable PITEM_LIST::const_iterator m_iterator;

};

#endif 
