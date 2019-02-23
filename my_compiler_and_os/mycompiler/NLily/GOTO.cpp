// GOTO.cpp: implementation of the GOTO class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "GOTO.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

GOTO::GOTO()
{

}

GOTO::~GOTO()
{

}

GOTO::GOTO(const GOTO &another)
{
	state = another.state;
	nonterminator = another.nonterminator;
	gotostate = another.gotostate;
}

const GOTO& GOTO::operator =(const GOTO &another)
{
	state = another.state;
	nonterminator = another.nonterminator;
	gotostate = another.gotostate;
	return *this;
}
