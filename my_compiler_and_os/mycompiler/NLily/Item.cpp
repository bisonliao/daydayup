// Item.cpp: implementation of the Item class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Item.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CItem::CItem(const CProducer &p, const CTerminator & t)
	:m_producer(p), m_terminator(t)
{

}

CItem::~CItem()
{

}

CItem::CItem(const CItem &another)
	:m_producer(another.m_producer), m_terminator(another.m_terminator)
{
}

bool CItem::operator ==(const CItem &another) const
{
	return (m_producer == another.m_producer && m_terminator == another.m_terminator);
}

const CItem& CItem::operator =(const CItem &another)
{
	m_producer = another.m_producer;
	m_terminator = another.m_terminator;
	return *this;
}

AnsiString CItem::ToString() const
{
	AnsiString ret = "[";
	ret.concat(m_producer.ToString());
	ret.concat(", ");
	ret.concat(m_terminator.ToString());
	ret.concat("]");
	return ret;
}
