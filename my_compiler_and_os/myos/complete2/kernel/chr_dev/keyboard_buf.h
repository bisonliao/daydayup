#ifndef _KEYBOARD_BUF_H_INCLUDED_
#define _KEYBOARD_BUF_H_INCLUDED_

#include "struct.h"


#pragma pack(1)
typedef struct 
{
	volatile unsigned long ulReadPos ;
	volatile unsigned long ulWritePos ;
	volatile unsigned long ulSize ;
}  KBBufHead;
#pragma pack()

int kbbuf_MemAttach(const unsigned char *ptr, unsigned int size, KBBufHead **ppstHead, int bCreate);
int kbbuf_IsEmpty(char *pcEmpty, const KBBufHead *pstHead);
int kbbuf_IsFull(char *pcFull, const KBBufHead *pstHead);
int kbbuf_Capability(const KBBufHead *pstHead);
int kbbuf_PutScanCode(KBBufHead *pstHead, uint8_t c);
int kbbuf_GetScanCode(KBBufHead *pstHead, uint8_t * pc);

#endif

