// Producer_Set.h: interface for the CProducer_Set class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PRODUCER_SET_H__F58CE868_E6E9_41C8_969D_83996F18FB66__INCLUDED_)
#define AFX_PRODUCER_SET_H__F58CE868_E6E9_41C8_969D_83996F18FB66__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Producer.h"
#include <list>

typedef CProducer* PPRODUCER;
typedef list<PPRODUCER> PPRODUCER_LIST;

////////////////////////////
//产生式的集合
class CProducer_Set  
{
public:
	const CProducer* GetProducerAt(int index) const;
	CProducer_Set RmAiAjProducer(const CNonTerminator &Ai,const CNonTerminator &Aj);
	CProducer_Set GetProducerOfNonTerm(const CNonTerminator & nt) const;
	CProducer_Set GetNonRcrsPrdcOfNonTerm(const CNonTerminator &nt) const;
	CProducer_Set GetDirectRcrsPrdcOfNonTerm(const CNonTerminator& nt) const;
	void RmPrdcOfNonTerminator(const CNonTerminator & nt);
	void clear();
	int GetProducerIndex(const CProducer& prd) const;
	int size() const;
	int remove(const CProducer& p);
	const CProducer* next() const;
	void begin_iterator() const;
	bool contain(const CProducer & p) const;
	int insert(PPRODUCER p);
	bool operator ==(const CProducer_Set &another) const;
	const CProducer_Set & operator=(const CProducer_Set &another);
	CProducer_Set();
	CProducer_Set(const CProducer_Set & another);
	virtual ~CProducer_Set();


private:
	PPRODUCER_LIST m_list;
	mutable PPRODUCER_LIST::const_iterator m_iterator;

};

#endif // !defined(AFX_PRODUCER_SET_H__F58CE868_E6E9_41C8_969D_83996F18FB66__INCLUDED_)
