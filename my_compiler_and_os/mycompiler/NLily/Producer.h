// Producer.h: interface for the CProducer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PRODUCER_H__5372C945_BC43_4565_AF38_77F7A013C23C__INCLUDED_)
#define AFX_PRODUCER_H__5372C945_BC43_4565_AF38_77F7A013C23C__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol_List.h"
#include "NonTerminator.h"
#include "Terminator.h"
#include "Dot.h"
#include "AnsiString.h"


typedef void (*PRODUCER_FUNC)(void*);
////////////////////////////
//产生式、项目
class CProducer  
{
public:
	void SetFunc(PRODUCER_FUNC func);
	void SetRight(const CSymbol_List& right);
	void SetLeft(const CNonTerminator& left);
	PRODUCER_FUNC GetFunc() const;
	const CSymbol_List& GetRight() const;
	const CNonTerminator& GetLeft() const;
	const CSymbol* GetSymbolAt(int index) const;
	const AnsiString  ToString() const;
	int GetDotIndex() const;	/*取得产生式右部中的分割点的位置，-1表示右部没有分割点*/
	CProducer(const CNonTerminator &left, const CSymbol_List& right, PRODUCER_FUNC func);
	CProducer(const CProducer& another);

	bool operator==(const CProducer& another) const;
	const CProducer& operator=(const CProducer& another);
	virtual ~CProducer();

private:
	CNonTerminator m_left;
	CSymbol_List m_right;
	PRODUCER_FUNC m_func;/*产生式对应的动作*/

};




#endif // !defined(AFX_PRODUCER_H__5372C945_BC43_4565_AF38_77F7A013C23C__INCLUDED_)
