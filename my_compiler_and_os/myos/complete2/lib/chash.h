#ifndef _C_HASH_H_INCLUDED_
#define _C_HASH_H_INCLUDED_

typedef struct
{
	volatile unsigned char head[100];
} THashHead;

int hash_init( THashHead * pstHead,
		size_t iBucketNum, size_t iMaxKeySize, size_t iMaxValSize,
		unsigned int (*fHash)(const unsigned char * keybuf, size_t keysize) ,
		int (*fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size),
		void (*fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) );

int hash_mem_attach(THashHead * pstHead, void * ptr, size_t size, int bCreat);

int hash_insert(THashHead * pstHead, 
		const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize);

int hash_update(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize);

///ɾ��ָ��key��key-val��
int hash_remove(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize);

/**
 * ����ָ��key������ҵ�����0��������Ӧ��val���浽(valbuf,pvalsize)
 * ���valbuf���Ȳ��������ضϣ�����2
 * ���û���ҵ�������1
 * ʧ�ܷ��ظ���
 */
int hash_find(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		unsigned char * valbuf, size_t * pvalsize);
int hash_find2(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		void * * ppVal);
/**
 * �ж�ָ��key��key-val���Ƿ����
 * ���ڷ���0
 * �����ڷ���1
 * ʧ�ܷ��ظ���
 */ 
int hash_exist(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize);

void hash_ScanUsedNode( THashHead * pstHead, 
			void (*fScan)(const unsigned char * key, 
			size_t keysize, 
			const unsigned char * val, 
			size_t valsize) );

#endif