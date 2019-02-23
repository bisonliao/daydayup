#include "int_Stack.h"

int_Stack::int_Stack() 
{
	this->m_nBufSize = 100;
	this->m_nEleNum = 0;
	this->m_pHdr = new int[100];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存空间分配失败!\n",
			__FILE__,
			__LINE__);
		fflush(stderr);
		exit(-1);
	}
}
int_Stack::int_Stack(int_Stack &ss) 
{
	this->m_nEleNum = ss.m_nEleNum;
	this->m_nBufSize = ss.m_nBufSize;
	this->m_pHdr = new int[m_nBufSize];
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
int_Stack::~int_Stack()
{
	if (NULL != m_pHdr)
	{
		delete[] m_pHdr;
	}
}
int_Stack&int_Stack::operator=(const int_Stack&ss) 
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
	this->m_pHdr = new int[m_nBufSize];
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
bool int_Stack::push(const int & ele) 
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
bool int_Stack::pop(int& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[--m_nEleNum];
	return TRUE;		
}
bool int_Stack::peek(int& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[m_nEleNum - 1];
	return TRUE;	
}
bool int_Stack::isEmpty()
{
	return (m_nEleNum <= 0);
}
int int_Stack::getSize()
{
	return m_nEleNum;
}
bool int_Stack::isFull()
{
	return (m_nEleNum >= m_nBufSize);
}
int int_Stack::enlarge()
{
	int * ptr = new int[m_nBufSize + 100];
	if (NULL == ptr)
	{
		return -1;
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		ptr[i] = m_pHdr[i];
	}
		
	int * tmp = m_pHdr;
	delete[] tmp;
	m_pHdr = ptr;
	m_nBufSize += 100;
	return 0;
}
