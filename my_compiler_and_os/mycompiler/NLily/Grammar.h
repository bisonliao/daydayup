// Grammar.h: interface for the CGrammar class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GRAMMAR_H__3E80963B_76D7_4446_97A4_FFA576B2065A__INCLUDED_)
#define AFX_GRAMMAR_H__3E80963B_76D7_4446_97A4_FFA576B2065A__INCLUDED_

#include "symbol_list.h"	// Added by ClassView
#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Symbol.h"
#include "Terminator.h"
#include "NonTerminator.h"
#include "Dot.h"
#include "Producer_Set.h"
#include "Item_Set.h"	// Added by ClassView
#include "items.h"
#include "AnalyseTable.h"


/////////////////////////////
//文法
class CGrammar  
{
public:
	void CalculateAnalyseTable();
	int parse();
	CGrammar(const CNonTerminator& StartSymbol,
			const CProducer_Set& PS );
	virtual ~CGrammar();
	const CGrammar & operator=(const CGrammar& another);
	bool operator ==(const CGrammar &another)const ;
	CGrammar(const CGrammar& g);
public:	
	static CSymbol_List FIRST(const CSymbol_List &X, CProducer_Set G);
	static void CalculateAnalyseTable(CAnalyseTable& table, const CSymbol_List  SL, const CProducer_Set G, const CItem_Set  I0, const CNonTerminator  StartSymbol, const CProducer_Set& GG);
	static void Items(items& C, const CItem_Set  I0, const CSymbol_List SL, const CProducer_Set G, const CProducer_Set& GG);
	static CItem_Set Goto(const CItem_Set I, const CSymbol*X, const CProducer_Set  G, const CProducer_Set& GG);
	static CItem_Set Closure(const CItem_Set I, const CProducer_Set G, const CProducer_Set &GG);
	static CSymbol_List FIRST(const CSymbol* X, CProducer_Set G);

	CGrammar ClearRecursion() const;	
	CTerminator lex() const;	
	static CNonTerminator GetTmpNonTerminator();
public:
	int ReadAnalyseTableFrmFile(const char *filename);
	int WriteAnalyseTableToFile(const char *filename);
	CNonTerminator m_StartSymbol;	/*开始符号*/
	CSymbol_List m_SL;	/*文法符号的集合*/
	CProducer_Set m_PS;	/*产生式的集合*/
	CAnalyseTable m_table;	/*分析表*/
};

#endif // !defined(AFX_GRAMMAR_H__3E80963B_76D7_4446_97A4_FFA576B2065A__INCLUDED_)
