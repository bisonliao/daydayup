// ACTION.cpp: implementation of the ACTION class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "ACTION.h"

#include <string.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ACTION::ACTION()
{

}

ACTION::~ACTION()
{

}

ACTION::ACTION(const ACTION &another)
{
	state = another.state;
	terminator = another.terminator;
	strcpy(action, another.action);
}

const ACTION & ACTION::operator =(const ACTION &another)
{
	state = another.state;
	terminator = another.terminator;
	strcpy(action, another.action);
	return *this;
}
