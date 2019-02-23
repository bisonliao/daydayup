// Item.h: interface for the Item class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ITEM_H__693CB0D9_59AE_4F0F_B765_3054D1C07650__INCLUDED_)
#define AFX_ITEM_H__693CB0D9_59AE_4F0F_B765_3054D1C07650__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Producer.h"
#include "Terminator.h"
#include "AnsiString.h"	// Added by ClassView

/////////////////////////////////
//ÏîÄ¿
class CItem  
{
public:
	AnsiString ToString() const;
	CItem(const CItem& another);
	CItem(const CProducer &p, const CTerminator & t);
	bool operator ==(const CItem &another) const;
	const CItem& operator =(const CItem &another);
	virtual ~CItem();

	const CProducer& GetProducer() const { return m_producer;};
	const CTerminator& GetTerminator() const { return m_terminator;};

private:
	CProducer m_producer;
	CTerminator m_terminator;

};

#endif // !defined(AFX_ITEM_H__693CB0D9_59AE_4F0F_B765_3054D1C07650__INCLUDED_)
