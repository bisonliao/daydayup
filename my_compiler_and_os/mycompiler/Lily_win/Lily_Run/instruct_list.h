#ifndef __INSTRUCT_LIST_H__
#define __INSTRUCT_LIST_H__

#include "common.h"
#include "LabelIndex_List.h"
#include "AnsiString.h"

class Instruct_List
{
public:
	Instruct_List();
	~Instruct_List();
	int Initialize(const char* filename, char *errmsg);
	int JmpToInstruct(int index);
	void BeginExecute();
	int NextInstruct(INSTRUCT & inst);
	int GetConstString(int index, AnsiString &s);
	int GetIndexByLabel(const char* label);
private:
	AnsiString * m_pConstString;
	int m_nConstStringCount;	/*��Ч��ConstString����*/
	int m_nConstStringMax;		/*�����ɵ�ConstString��������*/
	INSTRUCT * m_pInstruct;
	int m_nInstructCount;	/*��Ч��Instruct����*/
	int m_nInstructMax;	/*m_pInstructָ��Ļ����������ɵ�Instruct����*/
	int m_nCurrentIndex;	/*��ǰ����ִ�е�instruct������*/
	int m_nStart;	/*ָ��LABEL F_main_BEGIN��������*/

	LabelIndex_List labelindex_list;	/*��ǩ�����ǩ��������ŵĶ�Ӧ*/
private:
	int ReallocConstStringBuf();	/*����б�Ҫ������ConstString�������Ĵ�С*/
	int ReallocInstructBuf();	/*����б�Ҫ������Instruct�������Ĵ�С*/
	void RemoveNL(char * buf);	/*ȥ���ַ���ĩβ�Ļ��з���*/
};

#endif
