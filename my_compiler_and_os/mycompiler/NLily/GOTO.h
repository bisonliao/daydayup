// GOTO.h: interface for the GOTO class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_GOTO_H__22F000E9_9A82_41D8_A7AE_9D26F1258A9B__INCLUDED_)
#define AFX_GOTO_H__22F000E9_9A82_41D8_A7AE_9D26F1258A9B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "NonTerminator.h"

class GOTO  
{
public:
	GOTO();
	virtual ~GOTO();
	const GOTO& operator =(const GOTO &another);
public:
	GOTO(const GOTO&another);
	int state;	//状态号
	CNonTerminator nonterminator;	//非终结符
	int gotostate;	//跳转的目标状态号

};

#endif // !defined(AFX_GOTO_H__22F000E9_9A82_41D8_A7AE_9D26F1258A9B__INCLUDED_)
