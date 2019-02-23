#include "Token_Stack.h"

Token_Stack::Token_Stack() 
{
	this->m_nBufSize = 100;
	this->m_nEleNum = 0;
	this->m_pHdr = new Token[100];
	if (this->m_pHdr == NULL)
	{
		fprintf(stderr, "[%s][%d]内存空间分配失败!\n",
			__FILE__,
			__LINE__);
		fflush(stderr);
		exit(-1);
	}
}
Token_Stack::Token_Stack(Token_Stack &ss)
{
	this->m_nEleNum = ss.m_nEleNum;
	this->m_nBufSize = ss.m_nBufSize;
	this->m_pHdr = new Token[m_nBufSize];
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
Token_Stack::~Token_Stack()
{
	if (NULL != m_pHdr)
	{
		delete[] m_pHdr;
	}
}
Token_Stack&Token_Stack::operator=(const Token_Stack&ss) 
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
	this->m_pHdr = new Token[m_nBufSize];
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
bool Token_Stack::push(const Token & ele) 
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
bool Token_Stack::pop(Token& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[--m_nEleNum];
	return TRUE;		
}
bool Token_Stack::peek(Token& ele)
{
	if (m_nEleNum <= 0)
	{
		return FALSE;
	}
	ele = m_pHdr[m_nEleNum - 1];
	return TRUE;	
}
bool Token_Stack::isEmpty()
{
	return (m_nEleNum <= 0);
}
int Token_Stack::getSize()
{
	return m_nEleNum;
}
bool Token_Stack::isFull()
{
	return (m_nEleNum >= m_nBufSize);
}
int Token_Stack::enlarge()
{
	Token * ptr = new Token[m_nBufSize + 100];
	if (NULL == ptr)
	{
		return -1;
	}
	for (int i = 0; i < m_nEleNum; i++)
	{
		ptr[i] = m_pHdr[i];
	}
		
	Token * tmp = m_pHdr;
	delete[] tmp;
	m_pHdr = ptr;
	m_nBufSize += 100;
	return 0;
}
void Token_Stack::BeginPeekFrmTop()
{
	m_nPeekIndex = 0;
}
bool Token_Stack::PeekNextFrmTop(Token & ele)
{
	if ( (m_nEleNum - 1 - m_nPeekIndex) < 0)
	{
		return FALSE;
	}
	ele = m_pHdr[m_nEleNum - 1 - m_nPeekIndex];
	m_nPeekIndex++;
	return TRUE;
}
