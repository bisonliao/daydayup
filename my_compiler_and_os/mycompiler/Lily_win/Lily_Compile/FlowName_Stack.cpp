#include "FlowName_Stack.h"

FlowName_Stack::FlowName_Stack() 
{
	this->m_nBufSize = 100;
	this->m_nEleNum = 0;
	this->m_pHdr = new FlowName[100];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存空间分配失败!\n",
			__FILE__,
			__LINE__);
		fflush(stderr);
		exit(-1);
	}
}
FlowName_Stack::FlowName_Stack(FlowName_Stack &ss)
{
	this->m_nEleNum = ss.m_nEleNum;
	this->m_nBufSize = ss.m_nBufSize;
	this->m_pHdr = new FlowName[m_nBufSize];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存空间分配失败!\n",
			__FILE__,
			__LINE__);
		fflush(stderr);
		exit(-1);
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		this->m_pHdr[i] = ss.m_pHdr[i];
	}
}
FlowName_Stack::~FlowName_Stack()
{
	if (NULL != m_pHdr)
	{
		delete[] m_pHdr;
	}
}
FlowName_Stack&FlowName_Stack::operator=(const FlowName_Stack&ss) 
{
	if (this == &ss)
	{
		return *this;
	}
	if (m_pHdr != NULL)
	{
		delete[] m_pHdr;
		m_pHdr = NULL;
	}
	this->m_nEleNum = ss.m_nEleNum;
	this->m_nBufSize = ss.m_nBufSize;
	this->m_pHdr = new FlowName[m_nBufSize];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存空间分配失败!\n",
			__FILE__,
			__LINE__);
		fflush(stderr);
		exit(-1);
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		this->m_pHdr[i] = ss.m_pHdr[i];
	}
	return *this;
}
bool FlowName_Stack::push(const FlowName & ele) 
{
	if (isFull())
	{
		if (enlarge() != 0)
		{
			return FALSE;
		}
	}
	m_pHdr[m_nEleNum++] = ele;
	return TRUE;
}
bool FlowName_Stack::pop(FlowName& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[--m_nEleNum];
	return TRUE;		
}
bool FlowName_Stack::peek(FlowName& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[m_nEleNum - 1];
	return TRUE;	
}
bool FlowName_Stack::isEmpty()
{
	return (m_nEleNum <= 0);
}
int FlowName_Stack::getSize()
{
	return m_nEleNum;
}
bool FlowName_Stack::isFull()
{
	return (m_nEleNum >= m_nBufSize);
}
int FlowName_Stack::enlarge()
{
	FlowName * ptr = new FlowName[m_nBufSize + 100];
	if (NULL == ptr)
	{
		return -1;
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		ptr[i] = m_pHdr[i];
	}
		
	FlowName * tmp = m_pHdr;
	delete[] tmp;
	m_pHdr = ptr;
	m_nBufSize += 100;
	return 0;
}
bool FlowName_Stack::contain(FlowName& ele)
{
	for (int i = 0; i < m_nEleNum; i++)
	{
		if(strcmp(m_pHdr[i].fn_name, ele.fn_name) == 0)
		{
			return TRUE;
		}
	}
	return FALSE;
}
