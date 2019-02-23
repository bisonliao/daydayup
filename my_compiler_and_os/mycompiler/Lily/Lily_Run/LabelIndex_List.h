#ifndef _LabelIndex_LIST_H_
#define _LabelIndex_LIST_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

typedef struct
{
	char li_label[ID_MAX];	/*��ǩ����*/
	int li_index;		/*��ǩ�����ָ���б��е�������*/
} LabelIndex;

class LabelIndex_List
{
public:
	LabelIndex_List() ;
	LabelIndex_List(LabelIndex_List&ll) ;
	~LabelIndex_List();
	bool AddTail(const LabelIndex& ele);
	LabelIndex PopHead();
	bool GetAt(int index, LabelIndex&ele);
	int GetSize();
	bool IsEmpty();
	LabelIndex_List &operator=(const LabelIndex_List&ll) ;
	int GetIndexByLabel(const char *label);	
	void removeAll();
private:
	LabelIndex * m_pHdr;//����Ԫ�صĻ�������ͷָ��

	int m_nEleNum;//Ԫ�ظ���
	long m_nBufSize;//����ȥ����ܱ����Ԫ�ظ���
private:
	int enlarge();//��������������ʱ�����·��䣬���󻺳���
	bool isFull();//��ǰ�������Ƿ�������
};

#endif
