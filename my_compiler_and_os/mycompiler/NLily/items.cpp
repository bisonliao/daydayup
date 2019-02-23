// items.cpp: implementation of the items class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "items.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

items::items()
{
	m_nCount = 0;
	m_array = new CItem_Set[ARRAY_MAX];
	if (NULL == m_array)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
		exit(-1);
	}
}

items::~items()
{
	delete[] m_array;
}

bool items::contain(const CItem_Set &ele) const
{
	for (int i = 0; i < m_nCount; ++i)
	{
		if (ele == m_array[i])
		{
			return TRUE;
		}
	}
	return FALSE;
}

void items::add(const CItem_Set &ele)
{
	if (m_nCount >= ARRAY_MAX)
	{
		fprintf(stderr, "items的数组开辟空间太小!\n");
		exit(-1);
	}
	m_array[m_nCount++] = ele;
}

void items::clear()
{
	m_nCount = 0;
}

void items::begin_iterator() const
{
	m_nIndex = 0;
}

const CItem_Set * items::next() const
{
	if (m_nIndex >= m_nCount)
	{
		return NULL;
	}
	return &m_array[m_nIndex++];
}

const CItem_Set * items::GetItemSetAt(int index)
{
	if (index < 0 || (index + 1) >= m_nCount)
	{
		return NULL;
	}
	return &m_array[index];
}

int items::GetItemSetIndex(const CItem_Set &ele)
{
	for (int i = 0; i < m_nCount; i++)
	{
		if (m_array[i] == ele)
		{
			return i;
		}
	}
	return -1;
}

items::items(const items &another)
{
	m_nCount = another.m_nCount;
	m_array = new CItem_Set[ARRAY_MAX];
	if (NULL == m_array)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
				__FILE__,
				__LINE__);
		exit(-1);
	}
	for (int i = 0; i < m_nCount; ++i)
	{
		m_array[i] = another.m_array[i];
	}
}

int items::size() const
{
	return m_nCount;
}
