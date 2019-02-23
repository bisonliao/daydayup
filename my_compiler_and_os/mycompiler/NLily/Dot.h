// Dot.h: interface for the CDot class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_DOT_H__1936BCCC_5E38_4F71_ACEB_46E86A4FE5E4__INCLUDED_)
#define AFX_DOT_H__1936BCCC_5E38_4F71_ACEB_46E86A4FE5E4__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"

class CDot : public CSymbol  
{
public:
	CDot();
	CDot(const CDot& another);

	bool operator==(const CDot& another) const;
	const CDot& operator=(const CDot& another);

	virtual ~CDot();
	virtual int GetSymType() const ;
	virtual const AnsiString  ToString() const;

};

#endif // !defined(AFX_DOT_H__1936BCCC_5E38_4F71_ACEB_46E86A4FE5E4__INCLUDED_)
