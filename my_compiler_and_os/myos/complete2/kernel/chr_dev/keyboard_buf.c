#include "keyboard_buf.h"


int kbbuf_MemAttach(const unsigned char * ptr, unsigned int size, KBBufHead ** ppstHead, int bCreate)
{
	KBBufHead * pstHead ;
	if (ptr == NULL || size < ( sizeof(KBBufHead)+100) || ppstHead == NULL)
	{
		return -1;
	}
	pstHead = (KBBufHead *)ptr;
	if (bCreate)
	{
		pstHead->ulReadPos = 0;
		pstHead->ulWritePos = 0;
		pstHead->ulSize = size - sizeof(KBBufHead);
	}

	*ppstHead = pstHead;
	return 0;
}

int kbbuf_IsEmpty(char * pcEmpty, const KBBufHead * pstHead)
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
int kbbuf_IsFull(char * pcFull, const KBBufHead * pstHead)
{
	int size ;
	if (pstHead == NULL || pcFull == NULL)
	{
		return -1;
	}
	size = kbbuf_Capability(pstHead);
	if ( size < 0)
	{
		return -1;
	}
	if (size == 0)
	{
		*pcFull = 1;
	}
	else
	{
		*pcFull = 0;
	}
	return 0;
}
int kbbuf_Capability(const KBBufHead * pstHead)
{
	if (pstHead == NULL)
	{
		return -1;
	}
	if (pstHead->ulReadPos > pstHead->ulWritePos)
	{
		return pstHead->ulReadPos - pstHead->ulWritePos - 1;
	}
	if (pstHead->ulReadPos == pstHead->ulWritePos)
	{
		return pstHead->ulSize - 1;
	}
	if (pstHead->ulReadPos < pstHead->ulWritePos)
	{
		return (pstHead->ulSize - pstHead->ulWritePos) + (pstHead->ulReadPos) - 1;
	}
}
int kbbuf_PutScanCode(KBBufHead * pstHead, uint8_t c)
{
	int cap;
	unsigned char * pCur = NULL;
	int len1, len2;
	unsigned long ulWritePos = 0;
	if (pstHead == NULL )
	{
		return -1;
	}
	cap = kbbuf_Capability(pstHead);
	if (cap < 0)
	{
		return -1;
	}
	if (cap < 1)
	{
	/* 空间暂时不够 */
		return 1;
	}

	/* copy data */
	ulWritePos = pstHead->ulWritePos; /* 一定要等数据都拷贝好了才修改pstHead->ulWritePos */
	pCur =  ( (char*)pstHead + sizeof(KBBufHead) );

	*(pCur + ulWritePos) = c;
	ulWritePos = (ulWritePos + 1) % pstHead->ulSize;

	/* 修改指针 */
	pstHead->ulWritePos = ulWritePos;

	return 0;
}
int kbbuf_GetScanCode(KBBufHead * pstHead, uint8_t * pc)
{
	char cEmpty;
	unsigned char *pCur = NULL;
	unsigned long ulReadPos;
	int len1, len2;
	unsigned int size;

	if (pstHead == NULL || pc == NULL)
	{
		return -1;
	}
	if (kbbuf_IsEmpty( &cEmpty, pstHead) != 0)
	{
		return -1;
	}
	if (cEmpty)
	{
		return 1;
	}
	/* begin copy data */
	pCur =  ( (char*)pstHead + sizeof(KBBufHead) );
	ulReadPos = pstHead->ulReadPos;

	*pc = *(pCur+ulReadPos);
	ulReadPos = (ulReadPos+1) % pstHead->ulSize;

	pstHead->ulReadPos = ulReadPos;
	
	return 0;	
}
