#include "LabelIndex_List.h"
#include "common.h"

LabelIndex_List::LabelIndex_List()
{
	m_nBufSize = 10;
	m_nEleNum = 0;
	m_pHdr = new LabelIndex[10];
	if (NULL == m_pHdr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
}
LabelIndex_List::LabelIndex_List(LabelIndex_List&ll)
{
	this->m_nEleNum = ll.m_nEleNum;
	this->m_nBufSize = ll.m_nBufSize;
	this->m_pHdr = new LabelIndex[m_nBufSize];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		this->m_pHdr[i] = ll.m_pHdr[i];
	}
}
LabelIndex_List::	~LabelIndex_List()
{
	if (NULL != m_pHdr)
	{
		delete[] m_pHdr;
	}
}
LabelIndex_List&LabelIndex_List::operator=(const LabelIndex_List&ll)
{
	if (&ll == this)
	{
		return *this;
	}
	if (NULL != m_pHdr)
	{
		delete[] m_pHdr;
		m_pHdr = NULL;
	}
	//重新初始化
	this->m_nEleNum = ll.m_nEleNum;
	this->m_nBufSize = ll.m_nBufSize;
	this->m_pHdr = new LabelIndex[m_nBufSize];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
			__FILE__,
			__LINE__);
		exit(-1);
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		this->m_pHdr[i] = ll.m_pHdr[i];
	}
	return *this;
	
}
bool LabelIndex_List::AddTail(const LabelIndex& ele)
{
	if (this->isFull())
	{
		if (enlarge() != 0)
		{
			return FALSE;
		}
	}
	m_pHdr[m_nEleNum++] = ele;
	return TRUE;
}
LabelIndex LabelIndex_List::PopHead()
{
	LabelIndex head = m_pHdr[0];
	for (int i = 0; i <m_nEleNum - 1; i++)
	{
		m_pHdr[i] = m_pHdr[i + 1];
	}
	return head;

}
bool LabelIndex_List::GetAt(int index, LabelIndex&ele)
{
	if (index < 0 || index >= m_nEleNum)
	{
		return FALSE;
	}
	ele = m_pHdr[index];
	return TRUE;
}
int LabelIndex_List::GetSize()
{
	return m_nEleNum;
}
bool LabelIndex_List::IsEmpty()
{
	return (m_nEleNum <= 0);
}
bool LabelIndex_List::isFull()
{
	return (m_nEleNum >= m_nBufSize);
}
int LabelIndex_List::enlarge()
{
	LabelIndex * ptr = new LabelIndex[m_nBufSize + 10];
	if (NULL == ptr)
	{
		return -1;
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		ptr[i] = m_pHdr[i];
	}
	LabelIndex * tmp = m_pHdr;
	delete[] tmp;
	m_pHdr = ptr;
	m_nBufSize += 10;
	return 0;
}
int LabelIndex_List::GetIndexByLabel(const char *label)
{
	if (NULL == label)
	{
		return -1;
	}
	for (int j = 0; j < m_nEleNum; j++)
	{
		if ( strcmp(m_pHdr[j].li_label, label) == 0)
		{
			return m_pHdr[j].li_index;
		}
	}
	return -1;
}
void LabelIndex_List::removeAll()
{
		m_nEleNum = 0;
}
