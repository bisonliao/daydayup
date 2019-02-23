// ACTION.h: interface for the ACTION class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ACTION_H__7825632D_6831_48EB_9FC8_6F3FD7B4B38B__INCLUDED_)
#define AFX_ACTION_H__7825632D_6831_48EB_9FC8_6F3FD7B4B38B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Terminator.h"

class ACTION  
{
public:
	ACTION();
	virtual ~ACTION();
	const ACTION & operator =(const ACTION &another);
public:
	ACTION(const ACTION& another);
	int state;	//×´Ì¬ºÅ
	CTerminator terminator;	//ÖÕ½á·ûºÅ
	char action[10];	//¶¯×÷

};

#endif // !defined(AFX_ACTION_H__7825632D_6831_48EB_9FC8_6F3FD7B4B38B__INCLUDED_)
