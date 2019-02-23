#include "cycle_buf.h"


#define NULL    ( (void*)0 )


int cycle_buf_MemAttach(const unsigned char * ptr, unsigned int size, unsigned int ulUnitSz,
		CycleBufHead ** ppstHead, int bCreate)
{
	CycleBufHead * pstHead ;
	if (ptr == NULL ||  ppstHead == NULL)
	{
		return -1;
	}
	pstHead = (CycleBufHead *)ptr;
	if (bCreate)
	{
		pstHead->ulUnitSz = ulUnitSz;
		pstHead->ulReadPos = 0;
		pstHead->ulWritePos = 0;
		pstHead->ulSize = (size - sizeof(CycleBufHead)) / ulUnitSz;
		if (pstHead->ulSize < 10)
		{
			return -1;
		}
	}

	*ppstHead = pstHead;
	return 0;
}

int cycle_buf_IsEmpty(char * pcEmpty, const CycleBufHead * pstHead)
{
	if (pstHead == NULL || pcEmpty == NULL)
	{
		return -1;
	}
	*pcEmpty = 0;

	if (pstHead->ulReadPos == pstHead->ulWritePos)
	{
		*pcEmpty = 1;
	}
	return 0;
}
int cycle_buf_IsFull(char * pcFull, const CycleBufHead * pstHead)
{
	if (pstHead == NULL || pcFull == NULL)
	{
		return -1;
	}
	if (pstHead->ulReadPos > pstHead->ulWritePos  &&
		pstHead->ulReadPos-1 == pstHead->ulWritePos)
	{
		*pcFull = 1;
	}
	else
	{
		*pcFull = 0;
	}
	return 0;
}
int cycle_buf_push(CycleBufHead * pstHead, const void* val)
{
	char cFull;
	unsigned long ulWritePos;
	char * pCur = NULL;

	if (cycle_buf_IsFull(&cFull,  pstHead) )
	{
		return -1;
	}
	if (cFull)
	{
	/* 空间暂时不够 */
		return 1;
	}

	ulWritePos = pstHead->ulWritePos; /* 一定要等数据都拷贝好了才修改pstHead->ulWritePos */
	pCur =  ( (char*)pstHead + sizeof(CycleBufHead) );

	/* copy data of ONE unit */
	memcpy(pCur + ulWritePos*pstHead->ulUnitSz, val, pstHead->ulUnitSz);
	ulWritePos = (ulWritePos + 1) % pstHead->ulSize;

	/* 修改指针 */
	pstHead->ulWritePos = ulWritePos;

	return 0;
}
int cycle_buf_pop(CycleBufHead * pstHead,  void* val)
{
	char cEmpty;
	unsigned char *pCur = NULL;
	unsigned long ulReadPos;
	int len1, len2;
	unsigned int size;


	if (pstHead == NULL || val == NULL)
	{
		return -1;
	}
	if (cycle_buf_IsEmpty( &cEmpty, pstHead) != 0)
	{
		return -1;
	}
	if (cEmpty)
	{
		return 1;
	}
	pCur =  ( (char*)pstHead + sizeof(CycleBufHead) );
	ulReadPos = pstHead->ulReadPos;

	/* copy data of ONE unit */
	memcpy(val, pCur + ulReadPos*pstHead->ulUnitSz,  pstHead->ulUnitSz);
	ulReadPos = (ulReadPos+1) % pstHead->ulSize;

	pstHead->ulReadPos = ulReadPos;
	
	return 0;	
}
