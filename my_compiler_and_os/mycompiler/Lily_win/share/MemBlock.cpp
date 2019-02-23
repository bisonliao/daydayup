// MemBlock.cpp: implementation of the MemBlock class.
//
//////////////////////////////////////////////////////////////////////

#include "MemBlock.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#undef SHOW_STEP
#if defined(_DEBUG)
	#undef SHOW_STEP
	#define SHOW_STEP printf("执行%s的第%d行...\n", __FILE__, __LINE__);
	//#define SHOW_STEP ;
#else
	#define SHOW_STEP ;
#endif

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

MemBlock::MemBlock()
{
	m_pHdr = NULL;
	m_bufsize = 0;
}
MemBlock::MemBlock(unsigned int size)
{
	m_bufsize = size;
	if (size > 0)
	{
		m_pHdr = malloc(size);
		
		if (NULL == m_pHdr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
		memset(m_pHdr, 0, m_bufsize);
	}
}
MemBlock::MemBlock(const MemBlock & mb)
{
	unsigned int size = mb.m_bufsize;
	if (size > 0)
	{
		m_pHdr = realloc(m_pHdr, size);
		m_bufsize = size;
		if (NULL == m_pHdr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
		memcpy(m_pHdr, mb.m_pHdr, size);
	}
	else
	{
		if (NULL != m_pHdr)
		{
			free(m_pHdr);
			m_pHdr = NULL;
			m_bufsize = 0;
		}
	}
}
bool MemBlock::operator==(const MemBlock & mb) const
{
	if (mb.m_bufsize == m_bufsize)
	{
		if (m_bufsize == 0)
		{
			return TRUE;
		}
		if (memcmp(mb.m_pHdr, m_pHdr, m_bufsize) == 0)
		{
			return TRUE;
		}
	}
	return FALSE;
}
const MemBlock & MemBlock::operator=(const MemBlock& mb)
{
	if (&mb == this)
	{
		return *this;
	}
	unsigned int size = mb.m_bufsize;
	
	if (size > 0)
	{
		m_pHdr = realloc(m_pHdr, size);
	
			
	
		m_bufsize = size;
		if (NULL == m_pHdr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
		memcpy(m_pHdr, mb.m_pHdr, size);
	}
	else
	{
		if (NULL != m_pHdr)
		{
			free(m_pHdr);
			m_pHdr = NULL;
			m_bufsize = 0;
		}
			
	}
	return *this;
}
void MemBlock::Realloc(unsigned int size)
{
	if (size == 0)
	{
		if (NULL != m_pHdr)
		{
			free(m_pHdr);
			m_pHdr = NULL;
			m_bufsize = 0;
		}	
	}
	else
	{
		m_pHdr = realloc(m_pHdr, size);
		m_bufsize = size;
		if (NULL == m_pHdr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
	}
}
void MemBlock::SetZero()
{
	if (m_bufsize > 0)
	{
		memset(m_pHdr, 0, m_bufsize);
	}
}

MemBlock::~MemBlock()
{
	if (NULL != m_pHdr)
	{
		free(m_pHdr);
		m_pHdr = NULL;
		m_bufsize = 0;
	}
}
int MemBlock::SetValue(const char* buffer, unsigned int buffersize)
{
	if (buffersize == 0 || NULL == buffer)
	{
		free(m_pHdr);
		m_pHdr = NULL;
		m_bufsize = 0;		
		return 0;
	}	
	m_bufsize = buffersize;
	m_pHdr = realloc(m_pHdr, m_bufsize);

	if (NULL == m_pHdr)
	{
		fprintf(stderr, "[%s][%d]内存分配失败!\n",
					__FILE__,
					__LINE__);
		exit(-1);
	}
	
	memcpy(m_pHdr, buffer, m_bufsize);
	return 0;	
}
int MemBlock::GetSize() const
{
	if (NULL == m_pHdr)
	{
		return 0;	
	}
	else
	{
		return m_bufsize;	
	}
}
const char * MemBlock::GetBufferPtr() const
{
	return (char*)m_pHdr;	
}
void MemBlock::MemSet(int val, unsigned int size)
{
	if (NULL == m_pHdr || m_bufsize == 0)
	{
		return;
	}
	if (size > m_bufsize)
	{
		memset(m_pHdr, val, m_bufsize);
	}
	else
	{
		memset(m_pHdr, val, size);
	}
}
MemBlock MemBlock::MemSub(int offset, int len)
{
	MemBlock mb;
	if (NULL == m_pHdr || 0 == m_bufsize)
	{
		return mb;
	}
	if (offset < 0 || len < 0)
	{
		offset = 0;
		len = 0;
	}
	if (offset >= m_bufsize)
	{
		offset = m_bufsize -1;
	}
	if (offset + len > m_bufsize)
	{
		len = m_bufsize - offset;
	}
	
	char * p = (char*)(m_pHdr);
	mb.SetValue(p + offset, len);
	return mb;
}
void MemBlock::Append(const char* buffer, unsigned int buffersize)
{
	if (buffersize == 0 || NULL == buffer)
	{
		return;
	}
	else
	{
		int oldsize = m_bufsize;
		m_bufsize = m_bufsize + buffersize;
		m_pHdr = realloc(m_pHdr, m_bufsize);
		if (NULL == m_pHdr)
		{
			fprintf(stderr, "[%s][%d]内存分配失败!\n",
						__FILE__,
						__LINE__);
			exit(-1);
		}
		char * p = (char*)m_pHdr;
		memcpy( p + oldsize, buffer, buffersize);
	}
}
AnsiString MemBlock::toString() const
{
	AnsiString ret("");
	if (m_bufsize == 0 || NULL == m_pHdr)
	{
		return ret;
	}
	char tmp[10];
	for (int i = 0; i < m_bufsize; ++i)
	{
		char c = *((char*)m_pHdr + i);
		_snprintf(tmp, sizeof(tmp), "%2X ", c);
		ret.concat(tmp);
	}
	return ret;
}
