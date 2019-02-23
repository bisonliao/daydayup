#ifndef _CYCLE_BUF_H_INCLUDED_
#define _CYCLE_BUF_H_INCLUDED_


#pragma pack(1)
typedef struct 
{
	volatile unsigned long ulReadPos ;
	volatile unsigned long ulWritePos ;
	volatile unsigned long ulSize ;
	volatile unsigned long ulUnitSz;
}  CycleBufHead;
#pragma pack()

int cycle_buf_MemAttach(const unsigned char *ptr, unsigned int size, unsigned ulUnitSz,
		CycleBufHead **ppstHead, int bCreate);
int cycle_buf_IsEmpty(char *pcEmpty, const CycleBufHead *pstHead);
int cycle_buf_IsFull(char *pcFull, const CycleBufHead *pstHead);
int cycle_buf_pop(CycleBufHead * pstHead,  void* val);
int cycle_buf_push(CycleBufHead * pstHead, const void* val);

#endif

