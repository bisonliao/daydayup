#include "hash.h"

/*�����c++��������֪����ʲô����Ҫ�������*/
extern "C" int __gxx_personality_v0()
{
	return 0;
}

size_t CHash::GetSizeOfHashNode()
{
	return sizeof(CHashNode) + m_iMaxKeySize + m_iMaxValSize;
};
size_t CHash::GetSizeOfHashHead()
{
	return sizeof(CHashHead) + m_iBucketNum * sizeof(int);
};

int CHash::Initialize()
{
	unsigned int i;
	m_pHead->free_num = 0;
	m_pHead->used_num = 0;
	m_pHead->free_head = INVALID_NODE;
	m_pHead->used_head = INVALID_NODE;
	m_pHead->used_tail = INVALID_NODE;
	for (i = 0; i < m_iBucketNum; ++i)
	{
		m_pHead->bucket[i] = INVALID_NODE;
	}

	for (i = 0; i < m_iNodeNum; ++i)
	{
		if (InsertFreeList(i) < 0)
		{
			return -1;
		}
	}
	return 0;
};


int CHash::InsertFreeList(int node)
{
	int iNext = m_pHead->free_head;
	m_pHead->free_head = node;

	CHashNode *pstNode = Node2Ptr(node);
	pstNode->iNext = iNext;
	pstNode->iPrev = INVALID_NODE;
	pstNode->iNextInBucket = INVALID_NODE;
	pstNode->iPrevInBucket = INVALID_NODE;
	pstNode->iBucketID = INVALID_BUCKET_ID;


	if (pstNode->iNext != INVALID_NODE)
	{
		pstNode = Node2Ptr(pstNode->iNext);
		pstNode->iPrev = node;
	}
	++(m_pHead->free_num) ;
	return 0;
}
int CHash::RemoveNodeFromFreeList(int node)
{
	if (node < 0 || node >= m_iNodeNum)
	{
		return -1;
	}
	CHashNode *pstNode = Node2Ptr(node);
	if ( pstNode->iBucketID != INVALID_BUCKET_ID)
	{
		return -1;
	}
	int  iNext = pstNode->iNext;
	int  iPrev = pstNode->iPrev;
	CHashNode * pstNodeNext = NULL, *pstNodePrev = NULL;
	if (iNext != INVALID_NODE)
	{
		pstNodeNext = Node2Ptr(iNext);
	}
	if (iPrev != INVALID_NODE)
	{
		pstNodePrev = Node2Ptr(iPrev);
	}
	//�޸�ǰ��ڵ��ָ��
	if (pstNodeNext != NULL)
	{
		pstNodeNext->iPrev = iPrev;
	}
	if (pstNodePrev != NULL)
	{
		pstNodePrev->iNext = iNext;
	}
	else
	{
		//˵��node�ǵ�һ���ڵ�
		m_pHead->free_head = iNext;
	}
	--(m_pHead->free_num);
	return 0;
}
int CHash::ResetNode(int node)
{
	CHashNode * pstNode = Node2Ptr(node);
	new(pstNode) CHashNode();
	return 0;
}	
int CHash::GetFreeNode(int & node)
{
	//���free�������п��ýڵ㣬��ȡ֮
	if (m_pHead->free_head != INVALID_NODE)
	{
		node = m_pHead->free_head;		
		if (RemoveNodeFromFreeList(node) != 0)
		{
			return -1;
		}
		ResetNode(node);
		return 0;
	}
	//���򣬴�used���е�ĩβȡһ�� (��̭����ڵ�)
	if (m_pHead->used_tail != INVALID_NODE)
	{
		node = m_pHead->used_tail;
		if (m_fNodeSwapOut)
		{
			CHashNode *pstNode = Node2Ptr(node);
			m_fNodeSwapOut(pstNode->acKey, pstNode->iKeySize,
					pstNode->acVal, pstNode->iValSize);
		}
		if (RemoveNodeFromUsedList(node) != 0)
		{
			return -1;
		}
		if (RemoveNodeFromBucketList(node) != 0)
		{
			return -1;
		}

		ResetNode(node);
		return 0;
	}
	return -1;
}
int CHash::InsertUsedList(int node)
{
	//���뵽used����
	int iNext = m_pHead->used_head;
	m_pHead->used_head = node;

	CHashNode *pstNode = Node2Ptr(node);
	pstNode->iNext = iNext;
	pstNode->iPrev = INVALID_NODE;
	if (pstNode->iNext != INVALID_NODE)
	{
		pstNode = Node2Ptr(pstNode->iNext);
		pstNode->iPrev = node;
	}
	else
	{
		m_pHead->used_tail = node;
	}
	++(m_pHead->used_num);
	return 0;
}
int CHash::RemoveNodeFromUsedList(int node)
{
	if (node < 0 || node >= m_iNodeNum)
	{
		return -1;
	}
	CHashNode *pstNode = Node2Ptr(node);
	if ( pstNode->iBucketID == INVALID_BUCKET_ID)
	{
		return -1;
	}
	int  iNext = pstNode->iNext;
	int  iPrev = pstNode->iPrev;
	CHashNode * pstNodeNext = NULL, *pstNodePrev = NULL;
	if (iNext != INVALID_NODE)
	{
		pstNodeNext = Node2Ptr(iNext);
	}
	if (iPrev != INVALID_NODE)
	{
		pstNodePrev = Node2Ptr(iPrev);
	}
	//�޸�ǰ��ڵ��ָ��
	if (pstNodeNext != NULL)
	{
		pstNodeNext->iPrev = iPrev;
	}
	else
	{
		//˵��node���б�����һ���ڵ�
		m_pHead->used_tail = iPrev;
	}
	if (pstNodePrev != NULL)
	{
		pstNodePrev->iNext = iNext;
	}
	else
	{
		//˵��node�ǵ�һ���ڵ�
		m_pHead->used_head = iNext;
	}
	--(m_pHead->used_num);
	return 0;
}
int CHash::InsertBucketList(int bucketid, int node)
{
	if (bucketid < 0 || bucketid >= m_iBucketNum)
	{
		return -1;
	}
	//���뵽Ͱ��
	int iNext = m_pHead->bucket[ bucketid ] ;
	m_pHead->bucket[ bucketid ] = node;

	CHashNode *pstNode = Node2Ptr(node);
	pstNode->iNextInBucket = iNext;
	pstNode->iPrevInBucket = INVALID_NODE;
	pstNode->iBucketID = bucketid;
	if (iNext != INVALID_NODE)
	{
		pstNode = Node2Ptr( iNext );
		pstNode->iPrevInBucket = node;
	}
	return 0;
}
int CHash::RemoveNodeFromBucketList(int node)
{
	if (node < 0 || node >= m_iNodeNum)
	{
		return -1;
	}
	CHashNode *pstNode = Node2Ptr(node);
	if (pstNode->iBucketID < 0 || pstNode->iBucketID >= m_iBucketNum )

	{
		return -1;
	}
	int  iNextInBucket = pstNode->iNextInBucket;
	int  iPrevInBucket = pstNode->iPrevInBucket;
	CHashNode * pstNodeNext = NULL, *pstNodePrev = NULL;
	if (iNextInBucket != INVALID_NODE)
	{
		pstNodeNext = Node2Ptr(iNextInBucket);
	}
	if (iPrevInBucket != INVALID_NODE)
	{
		pstNodePrev = Node2Ptr(iPrevInBucket);
	}
	//�޸�ǰ��ڵ��ָ��
	if (pstNodeNext != NULL)
	{
		pstNodeNext->iPrevInBucket = iPrevInBucket;
	}
	if (pstNodePrev != NULL)
	{
		pstNodePrev->iNextInBucket = iNextInBucket;
	}
	else
	{
		//˵��node�ǵ�һ���ڵ�
		m_pHead->bucket[ pstNode->iBucketID ] = iNextInBucket;
	}
	return 0;
}

CHash::CHashNode * CHash::Node2Ptr(int node)
{
	if (node >= m_iNodeNum || node < 0)
	{
		return NULL;
	}
	char  * p = (char *)m_pHead;
	p += m_iHeadSize;
	p += node * m_iNodeSize;
	return (CHash::CHashNode *)p;
}
int CHash::Ptr2Node(void *p )
{
	if (NULL == p)
	{
		return INVALID_NODE;
	}
	char * ptr = (char *)p;
	int bytes = ptr - (char*)m_pHead;
	if (bytes < 0)
	{
		return INVALID_NODE;
	}
	size_t iOffset = (size_t)bytes;

	if (iOffset < m_iHeadSize)
	{
		return INVALID_NODE;
	}
	iOffset -= m_iHeadSize;
	if (iOffset % m_iNodeSize)
	{
		return INVALID_NODE;
	}
	return iOffset / m_iNodeSize;
}

CHash::CHash(size_t iBucketNum, size_t iMaxKeySize, size_t iMaxValSize, 
		unsigned int (*fHash)(const unsigned char * keybuf, size_t keysize) /*= NULL*/,
		bool (*fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size) /*= NULL*/,
		void (*fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) /*= NULL*/)
{
	m_pHead = NULL;
	m_iNodeNum = 0;
	m_iBucketNum = iBucketNum;
	m_iMaxKeySize = iMaxKeySize;
	m_iMaxValSize = iMaxValSize;
	m_iHeadSize = GetSizeOfHashHead();
	m_iNodeSize = GetSizeOfHashNode();

	if (fHash != NULL)
	{
		m_fHash = fHash;
	}
	else
	{
		m_fHash = CHash::hash;
	}

	if (fKeyEqual != NULL)
	{
		m_fKeyEqual = fKeyEqual;
	}
	else
	{
		m_fKeyEqual = CHash::KeyEqual;
	}

	if (fNodeSwapOut != NULL)
	{
		m_fNodeSwapOut = fNodeSwapOut;
	}
	else
	{
		m_fNodeSwapOut = NULL;
	}

}
int CHash::MemAttach(void * ptr, size_t size, bool bCreat /*= false*/)
{
	if (size < 	(m_iHeadSize+m_iNodeSize) )
	{
		return -1;
	}
	m_pHead = (CHashHead*)ptr;
	m_iNodeNum = (size - m_iHeadSize) / m_iNodeSize;
	if (bCreat)
	{
		if (Initialize() < 0)
		{
			return -1;
		}
	}
	return 0;
}
unsigned int CHash::hash(const unsigned char * keybuf, size_t keysize)
{
	size_t i = 0;
	unsigned int sum = 0;
	for (i = 0; i < keysize; ++i)
	{
		sum = sum * 256 + keybuf[i];
	}
	return sum;
}
bool CHash::KeyEqual(const unsigned char * key1, size_t key1size,
		const unsigned char * key2, size_t key2size)
{
	if (key1size != key2size)
	{
		return false;
	}
	if (memcmp(key1, key2, key1size) == 0)
	{
		return true;
	}
	return false;
}
// 0-�ҵ��� 1-û���ҵ� ����-ʧ��
int CHash::Find(int & node, int & bucketid, const unsigned char * keybuf, size_t keysize)
{
	bucketid = m_fHash(keybuf, keysize) % m_iBucketNum;
	CHashNode *pstNode = NULL;
	int iNext = m_pHead->bucket[ bucketid ];
	while (iNext != INVALID_NODE)
	{
		pstNode = Node2Ptr(iNext);

		if ( m_fKeyEqual(pstNode->acKey, pstNode->iKeySize,
					keybuf, keysize))
		{
			node = iNext;
			return 0;
		}
		iNext = pstNode->iNextInBucket;
	}
	return 1;
}
int CHash::Insert(const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize)
{
	if (NULL == keybuf || valbuf == NULL || keysize == 0 || valsize == 0 ||
			keysize > m_iMaxKeySize || valsize > m_iMaxValSize)
	{
		return -__LINE__;
	}
	if (m_pHead == NULL)
	{
		return -__LINE__;
	}
	int node, bucketid;
	int iRet = Find(node, bucketid, keybuf, keysize);
	if (iRet < 0)
	{
		return -__LINE__;
	}
	if (iRet == 0) //�Ѿ�����,ֱ�Ӹ���
	{
		CHashNode *pstNode = Node2Ptr(node);	
		pstNode->iValSize = valsize;
		memcpy(pstNode->acVal+m_iMaxKeySize, valbuf, valsize);
		return 0;
	}
	//�������
	bucketid = m_fHash(keybuf, keysize) % m_iBucketNum;

	TRACE("�����%d��Ͱ\n", bucketid);

	if (GetFreeNode(node) != 0)
	{
		return -__LINE__;
	}
	TRACE("�õ���%d�Ľڵ�\n", node);
	CHashNode *pstNode = Node2Ptr(node);
	pstNode->iKeySize = keysize;
	pstNode->iValSize = valsize;
	memcpy(pstNode->acKey, keybuf, keysize);
	memcpy(pstNode->acVal + m_iMaxKeySize, valbuf, valsize);

	TRACE("����used����\n");
	if ( InsertUsedList(node) != 0)
	{
		return -__LINE__;
	}
	TRACE("����bucket����\n");
	if ( InsertBucketList(bucketid, node) != 0)
	{
		return -__LINE__;
	}
	return 0;
}
int CHash::Update(const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize)
{
	if (NULL == keybuf || valbuf == NULL || keysize == 0 || valsize == 0 ||
			keysize > m_iMaxKeySize || valsize > m_iMaxValSize)
	{
		return -__LINE__;
	}
	if (m_pHead == NULL)
	{
		return -__LINE__;
	}
	int iRet;
	int node, bucketid;
	iRet = Find(node, bucketid, keybuf, keysize);
	if (iRet == 1) //û���ҵ���Ӧ�ڵ�
	{
		return 1;
	}
	if (iRet < 0)
	{
		return -__LINE__;
	}

	CHashNode * pstNode = Node2Ptr(node);

	pstNode->iValSize = valsize;
	memcpy(pstNode->acVal+m_iMaxKeySize, valbuf, valsize);

	return 0;

}
int CHash::Remove(const unsigned char * keybuf, size_t keysize)
{
	if (NULL == keybuf ||  keysize == 0  || keysize > m_iMaxKeySize )
	{
		return -__LINE__;
	}
	if (m_pHead == NULL)
	{
		return -__LINE__;
	}
	int iRet;
	int node, bucketid;
	iRet = Find(node, bucketid, keybuf, keysize);
	if (iRet == 1) //û���ҵ���Ӧ�ڵ�
	{
		return 1;
	}
	if (iRet < 0)
	{
		return -__LINE__;
	}
	if (RemoveNodeFromUsedList(node) != 0)
	{
		return -__LINE__;
	}
	if (RemoveNodeFromBucketList(node) != 0)
	{
		return -__LINE__;
	}
	if (InsertFreeList(node) != 0)
	{
		return -__LINE__;
	}
	return 0;
}
int CHash::Find(const unsigned char * keybuf, size_t keysize,
		unsigned char * valbuf, size_t * pvalsize)
{
	if (NULL == keybuf || valbuf == NULL || keysize == 0 || pvalsize == NULL || keysize > m_iMaxKeySize )
	{
		return -__LINE__;
	}
	if (m_pHead == NULL)
	{
		return -__LINE__;
	}
	int iRet;
	int node, bucketid;
	iRet = Find(node, bucketid, keybuf, keysize);
	if (iRet == 1) //û���ҵ���Ӧ�ڵ�
	{
		return 1;
	}
	if (iRet < 0)
	{
		return -__LINE__;
	}
	CHashNode * pstNode = Node2Ptr(node);
	if (pstNode->iValSize > *pvalsize)
	{
		memcpy(valbuf, pstNode->acVal+m_iMaxKeySize, *pvalsize);
		return 2;
	}
	*pvalsize = pstNode->iValSize;
	memcpy(valbuf, pstNode->acVal+m_iMaxKeySize, *pvalsize);
	return 0;
}
int CHash::Find(const unsigned char * keybuf, size_t keysize, void * * ppVal)
{
	if (NULL == keybuf ||  keysize == 0 ||  keysize > m_iMaxKeySize  || ppVal == NULL)
	{
		return -__LINE__;
	}
	if (m_pHead == NULL)
	{
		return -__LINE__;
	}
	int iRet;
	int node, bucketid;
	iRet = Find(node, bucketid, keybuf, keysize);
	if (iRet == 1) //û���ҵ���Ӧ�ڵ�
	{
		return 1;
	}
	if (iRet < 0)
	{
		return -__LINE__;
	}
	CHashNode * pstNode = Node2Ptr(node);
	*ppVal = pstNode->acVal+m_iMaxKeySize;
	return 0;
}
int CHash::Exist(const unsigned char * keybuf, size_t keysize)
{
	if (NULL == keybuf ||  keysize == 0  || keysize > m_iMaxKeySize )
	{
		return -__LINE__;
	}
	if (m_pHead == NULL)
	{
		return -__LINE__;
	}
	int iRet;
	int node, bucketid;
	iRet = Find(node, bucketid, keybuf, keysize);
	if (iRet == 1) //û���ҵ���Ӧ�ڵ�
	{
		return 1;
	}
	if (iRet < 0)
	{
		return -__LINE__;
	}
	return 0;	
}
void CHash::Dump()
{
#if 0
	printf("Begin Dump>>>\n");
	printf("m_iBucketNum:%lu\tm_iNodeNum:%lu\tm_iMaxKeySize:%lu\tm_iMaxValSize:%lu\tm_iHeadSize:%lu\tm_iNodeSize:%lu\n",
			m_iBucketNum, m_iNodeNum, m_iMaxKeySize, m_iMaxValSize, m_iHeadSize, m_iNodeSize);	
	int i;
	int j;
	int iNext;
	CHashNode *pstNode = NULL;

	printf("\n���нڵ���:\n");
	iNext = m_pHead->free_head;
	i = 0;
	while (iNext != INVALID_NODE)
	{
		++i;
		printf("%d ", iNext);
		pstNode = Node2Ptr(iNext);
		iNext = pstNode->iNext;
	}
	printf("\n����%d��\n", i);

	printf("\n�Ѿ�ʹ�õĽڵ���:\n");
	iNext = m_pHead->used_head;
	i = 0;
	while (iNext != INVALID_NODE)
	{
		++i;
		printf("%d ", iNext);
		pstNode = Node2Ptr(iNext);
		iNext = pstNode->iNext;
	}
	printf("\n����%d��\n", i);
	printf("used_tail=%d\n", m_pHead->used_tail);

	for (j = 0; j < m_iBucketNum; ++j)
	{
		printf("\nͰ%d�еĽڵ���:\n", j);
		iNext = m_pHead->bucket[j];
		i = 0;
		while (iNext != INVALID_NODE)
		{
			++i;
			printf("%d ", iNext);
			pstNode = Node2Ptr(iNext);
			iNext = pstNode->iNextInBucket;
		}
		printf("\n����%d��\n", i);
	}
	printf("Dump End<<<\n");
#endif
}
int CHash::Verify()
{
	if ( (m_pHead->used_num + m_pHead->free_num) != m_iNodeNum)
	{
		return -__LINE__;
	}
	int iNext, iPrev ;
	int iNum ;
	CHashNode *pstNode = NULL;

	//���used����
	iNext = m_pHead->used_head;
	iPrev = INVALID_NODE;
	iNum = 0;
	pstNode = NULL;
	while (iNext != INVALID_NODE)
	{
		++iNum;
		pstNode = Node2Ptr(iNext);

		if (pstNode->iNext == INVALID_NODE)//�����used��������һ���ڵ�
		{
			if (iNext != m_pHead->used_tail) //used_tailӦ��ָ����
			{
				return -__LINE__;
			}
		}
		if (pstNode->iBucketID == INVALID_BUCKET_ID)
		{
			return -__LINE__;
		}
		if (pstNode->iPrev != iPrev)
		{
			return -__LINE__;
		}

		iPrev = iNext;
		iNext = pstNode->iNext;
	}
	if (iNum != m_pHead->used_num)
	{
		return -3;
	}

	//���free����
	iNext = m_pHead->free_head;
	iNum = 0;
	iPrev = INVALID_NODE;
	pstNode = NULL;
	while (iNext != INVALID_NODE)
	{
		++iNum;
		pstNode = Node2Ptr(iNext);

		if (pstNode->iBucketID != INVALID_BUCKET_ID)
		{
			return -__LINE__;
		}
		if (pstNode->iPrev != iPrev)
		{
			return -__LINE__;
		}
		iPrev = iNext;
		iNext = pstNode->iNext;
	}
	if (iNum != m_pHead->free_num)
	{
		return -__LINE__;
	}

	//���bucket����
	iNum = 0;
	int i;
	for (i = 0; i < m_iBucketNum; ++i)
	{
		iNext = m_pHead->bucket[i];
		iPrev = INVALID_NODE;
		pstNode = NULL;
		while (iNext != INVALID_NODE)
		{
			++iNum;
			pstNode = Node2Ptr(iNext);

			if (pstNode->iBucketID != i)
			{
				return -__LINE__;
			}
			if (pstNode->iPrevInBucket != iPrev)
			{
				return -__LINE__;
			}
			iPrev = iNext;
			iNext = pstNode->iNextInBucket;
		}
	}
	if (iNum != m_pHead->used_num)
	{
		return -__LINE__;
	}
	return 0;
}

/////////////////////////////////////////////////////////////////////////
// c���Խӿ�
/////////////////////////////////////////////////////////////////////////
extern "C" int hash_init( THashHead * pstHead,
		size_t iBucketNum, size_t iMaxKeySize, size_t iMaxValSize,
		unsigned int (*fHash)(const unsigned char * keybuf, size_t keysize) ,
		bool (*fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size),
		void (*fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) )
{
	if (sizeof(THashHead) < sizeof(CHash))
	{
		return -1;
	}
	new( pstHead) CHash(iBucketNum, iMaxKeySize, iMaxValSize,
			fHash,
			fKeyEqual,
			fNodeSwapOut);
	return 0;
}
extern "C" int hash_mem_attach(THashHead * pstHead, void * ptr, size_t size, bool bCreat)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->MemAttach(ptr, size, bCreat);
}

extern "C" int hash_insert(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->Insert(keybuf, keysize, valbuf, valsize);
}

extern "C" int hash_update(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->Update(keybuf, keysize, valbuf, valsize);
}

///ɾ��ָ��key��key-val��
extern "C" int hash_remove(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->Remove(keybuf, keysize);
}

extern "C" int hash_find(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		unsigned char * valbuf, size_t * pvalsize)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->Find(keybuf, keysize, valbuf, pvalsize);
}
extern "C" int hash_find2(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		void ** ppVal)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->Find(keybuf, keysize, ppVal);
}
extern "C" int hash_exist(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize)
{
	CHash * pHashClass = (CHash *)pstHead;
	return pHashClass->Exist(keybuf, keysize);
}
