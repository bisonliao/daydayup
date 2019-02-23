// Dot.cpp: implementation of the CDot class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Dot.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CDot::CDot()
{

}
CDot::CDot(const CDot& another)
{
	
}
bool CDot::operator ==(const CDot& another) const
{
	return TRUE;
}
const CDot& CDot::operator =(const CDot& another)
{
	return *this;
}

CDot::~CDot()
{

}
int CDot::GetSymType() const
{
	return SYMBOL_DOT;
}
const AnsiString   CDot::ToString() const
{
	return ".";
}
