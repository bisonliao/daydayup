// MemBlock.h: interface for the MemBlock class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __MEMBLOCK_H__
#define __MEMBLOCK_H__

#include "AnsiString.h"

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

class MemBlock  
{
public:
	MemBlock();
	MemBlock(unsigned int size);
	MemBlock(const MemBlock & mb);
	bool operator==(const MemBlock & mb) const;	/*�����ڴ��Ƿ����*/
	const MemBlock & operator=(const MemBlock& mb);	/*�໥��ֵ*/
	void Realloc(unsigned int size);	/*���·���ռ�*/
	void SetZero();
	int SetValue(const char* buffer, unsigned int buffersize);	/*����һ��bufferָ�����ڴ浽��ǰʵ��*/
	int GetSize() const;	/*�õ��ڴ��Ĵ�С*/
	const char * GetBufferPtr() const;	/*�ڴ��� ͷָ��*/
	void MemSet(int val, unsigned int size);	/*�����ڴ��ֵ*/
	MemBlock MemSub(int offset, int len);	/*��ȡһ���ڴ�*/
	void Append(const char* buffer, unsigned int buffersize); /*׷��һ���ڴ�*/
	AnsiString toString() const;
	virtual ~MemBlock();
private:
	void * m_pHdr;	/*�ڴ���ͷָ��*/
	unsigned int m_bufsize;	/*�ڴ��Ĵ�С*/

};

#endif // !defined(AFX_MEMBLOCK_H__C16CC3A6_F352_4F5D_861A_B2755531DACF__INCLUDED_)
