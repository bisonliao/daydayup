// items.h: interface for the items class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ITEMS_H__E863D540_506B_4B33_8840_58E3AD051CE2__INCLUDED_)
#define AFX_ITEMS_H__E863D540_506B_4B33_8840_58E3AD051CE2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Item_Set.h"


class items  
{
public:
	items();
	virtual ~items();
public:
	int size() const;
	items(const items&another);
	int GetItemSetIndex(const CItem_Set &ele);
	const CItem_Set * GetItemSetAt(int index);
	const CItem_Set * next() const;
	void begin_iterator() const;
	void clear();
	void add(const CItem_Set &ele);
	bool contain(const CItem_Set & ele) const;
private:
	enum{ARRAY_MAX=2000};
	//CItem_Set m_array[ARRAY_MAX];
	CItem_Set * m_array;
	int m_nCount;
	mutable int m_nIndex;

};

#endif // !defined(AFX_ITEMS_H__E863D540_506B_4B33_8840_58E3AD051CE2__INCLUDED_)
