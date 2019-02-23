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
	int m_nConstStringCount;	/*有效的ConstString个数*/
	int m_nConstStringMax;		/*能容纳的ConstString的最大个数*/
	INSTRUCT * m_pInstruct;
	int m_nInstructCount;	/*有效的Instruct数量*/
	int m_nInstructMax;	/*m_pInstruct指向的缓冲区能容纳的Instruct数量*/
	int m_nCurrentIndex;	/*当前正在执行的instruct的索引*/
	int m_nStart;	/*指令LABEL F_main_BEGIN的索引号*/

	LabelIndex_List labelindex_list;	/*标签名与标签语句索引号的对应*/
private:
	int ReallocConstStringBuf();	/*如果有必要，增大ConstString缓冲区的大小*/
	int ReallocInstructBuf();	/*如果有必要，增大Instruct缓冲区的大小*/
	void RemoveNL(char * buf);	/*去除字符串末尾的换行符号*/
};

#endif
