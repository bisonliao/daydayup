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
	bool operator==(const MemBlock & mb) const;	/*两块内存是否相等*/
	const MemBlock & operator=(const MemBlock& mb);	/*相互赋值*/
	void Realloc(unsigned int size);	/*重新分配空间*/
	void SetZero();
	int SetValue(const char* buffer, unsigned int buffersize);	/*拷贝一块buffer指定的内存到当前实例*/
	int GetSize() const;	/*得到内存块的大小*/
	const char * GetBufferPtr() const;	/*内存块的 头指针*/
	void MemSet(int val, unsigned int size);	/*设置内存的值*/
	MemBlock MemSub(int offset, int len);	/*截取一段内存*/
	void Append(const char* buffer, unsigned int buffersize); /*追加一段内存*/
	AnsiString toString() const;
	virtual ~MemBlock();
private:
	void * m_pHdr;	/*内存块的头指针*/
	unsigned int m_bufsize;	/*内存块的大小*/

};

#endif // !defined(AFX_MEMBLOCK_H__C16CC3A6_F352_4F5D_861A_B2755531DACF__INCLUDED_)
