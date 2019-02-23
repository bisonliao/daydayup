// AnalyseTable.h: interface for the CAnalyseTable class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ANALYSETABLE_H__D2A201CC_7B9C_40D8_A27C_E81FE3BFFBB2__INCLUDED_)
#define AFX_ANALYSETABLE_H__D2A201CC_7B9C_40D8_A27C_E81FE3BFFBB2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "Terminator.h"
#include "NonTerminator.h"
#include "ACTION.h"
#include "GOTO.h"



/////////////////////////////
//”Ô∑®∑÷Œˆ±Ì
class CAnalyseTable  
{
public:
	void clear();
	int GetGoto(GOTO& ele);
	int GetAction(ACTION &ele);
	void AddGoto(const GOTO& ele);
	void AddAction(const ACTION& ele);
	CAnalyseTable();
	virtual ~CAnalyseTable();
private:
	enum{ACTION_MAX=2000, GOTO_MAX=500};
	ACTION * m_action;
	GOTO * m_goto;
	int m_nActionCount;
	int m_nGotoCount;
public:
	int ReadFrmFile(const char* filename);
	int WriteToFile(const char * filename);
	CAnalyseTable(const CAnalyseTable &another);
	const CAnalyseTable & operator =(const CAnalyseTable &another);
};

#endif // !defined(AFX_ANALYSETABLE_H__D2A201CC_7B9C_40D8_A27C_E81FE3BFFBB2__INCLUDED_)
