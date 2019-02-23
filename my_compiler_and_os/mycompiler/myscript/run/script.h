#ifndef _SCRIPT_H_INCLUDED_
#define _SCRIPT_H_INCLUDED_

#include <vector>
#include <string>
#include <list>
#include <map>
#include "mem.h"
#include "var.h"
using namespace std;

class CScript
{
public:
	CScript();
	CScript(const CScript & another);
	const CScript& operator=(const CScript & another);

	void Clear();
	int PushInstruct(const vector<string> & instructs);
	int Run(const list<CVar> &args, CVar& rtnval);
private:
	vector<string> m_vInstructs;
	map<int, int> m_mLabels; //��ǩ�����������λ��
	CMem m_mem; 	//�����ڴ���

	CVar RunFunction(const string & funcname, const list<CVar> &arglist);
};

#endif
