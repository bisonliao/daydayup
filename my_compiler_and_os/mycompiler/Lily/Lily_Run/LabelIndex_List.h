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
	char li_label[ID_MAX];	/*标签名称*/
	int li_index;		/*标签语句在指令列表中的索引号*/
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
	LabelIndex * m_pHdr;//保存元素的缓冲区的头指针

	int m_nEleNum;//元素个数
	long m_nBufSize;//缓冲去最大能保存的元素个数
private:
	int enlarge();//当缓冲区不够大时，重新分配，扩大缓冲区
	bool isFull();//当前缓冲区是否用完了
};

#endif
