/**
 * @file ͨ��hash����ͷ�ļ�
 * @brief ʹ�ô�С��ȵĽڵ���key-val�ԣ��ڵ������һƬ�������ڴ�ռ�
 *        �ڲ�ʵ�ֲ��ô���Ͱ��ÿ��Ͱ��һ��˫������
 *	      ���ڵ㲻����ʱ������FILO�Ĳ�����̭���õĽڵ�,�����Իص�һ���û�����
 *        hash������key����жϺ���������
 *        ���ʺ�key-val��С�仯��Χ�ܴ�����
 */
#ifndef _HASH_H_INCLUDED_
#define _HASH_H_INCLUDED_

#include "types.h"

#define TRACE(fmt, args...) {;}


/**
 * hash����������
 */
class CHash
{
	private:
		enum
		{
			INVALID_BUCKET_ID	= -1, ///<��Ч��Ͱ���
			INVALID_NODE		= -1, ///<��Ч�Ľڵ���
		};

#pragma pack(1)
#pragma align(1)
		///hash�����ͷ���ṹ
		class CHashHead
		{
			private:
				volatile int free_head; ///<���нڵ������ͷָ��
				volatile int used_head; ///<��ʹ�ýڵ������ͷָ��
				volatile int used_tail; ///<��ʹ�ýڵ������βָ�룬��̭��
				volatile unsigned int free_num; ///< ���нڵ����
				volatile unsigned int used_num; ///< ��ʹ�ýڵ����
				volatile int bucket[0]; ///<Ͱ����
				friend class CHash;
		};

		/**
		 *�ڵ�����ݽṹ
		 *һ���ڵ�Ҫô��free�����У�Ҫô��used������
		 *��һ���ڵ���used�����е�ʱ����ͬʱλ��ĳ��Ͱ��
		 */
		class CHashNode
		{
			private:
				volatile int iNext; ///< �ڵ��� free/used�����У�ǰһ���ڵ��ָ��
				volatile int iPrev; ///< �ڵ��� free/used�����У���һ���ڵ��ָ��

				volatile int iBucketID; ///<���һ���ڵ���used�����У���ôiBucketIDΪ,���� [0,m_iBucketNum)֮���ֵ,����ΪINVALID_BUCKET_ID

				volatile int iNextInBucket; ///< �ڵ���ĳ��Ͱ�������У�ǰһ���ڵ��ָ��
				volatile int iPrevInBucket; ///< �ڵ���ĳ��Ͱ�������У���һ���ڵ��ָ��

				volatile size_t iKeySize; ///< keyֵ���ֽ���
				volatile size_t iValSize; ///< valֵ���ֽ���

				volatile unsigned char acKey[0]; ///< key
				volatile unsigned char acVal[0]; ///< val
			private:
				///Ĭ�Ϲ��캯��
				CHashNode()
				{
					iNext = INVALID_NODE;
					iPrev = INVALID_NODE;
					iKeySize = 0;
					iValSize = 0;
					iNextInBucket = INVALID_NODE;
					iPrevInBucket = INVALID_NODE;
					iBucketID = INVALID_BUCKET_ID;
				};

				friend class CHash;

		};

#pragma pack()
#pragma align()

	public:
		volatile CHashHead * m_pHead; ///< hash����ͷ�ṹ��ָ��
		volatile size_t m_iBucketNum; ///< Ͱ�ĸ���
		volatile size_t m_iNodeNum;   ///< �ڵ��ܸ���
		volatile size_t m_iMaxKeySize; ///< key������ֽ���
		volatile size_t m_iMaxValSize; ///< val������ֽ���
		volatile size_t m_iHeadSize;   ///< hash����ͷ���Ĵ�С
		volatile size_t m_iNodeSize;   ///< �ڵ�Ĵ�С

		///���ڼ���hashֵ�ĺ���
		unsigned int (*m_fHash)(const unsigned char * keybuf, size_t keysize);

		///���ڱȽ�����key�Ƿ���ȵĺ���
		bool (*m_fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size);

		///��һ������ʹ�õĽڵ㱻��̭����ʱ����Ҫ���еĶ������
		int (*m_fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize);


	private:
		CHash & operator=(const CHash &	 a);
		bool operator==(const CHash &	 a);
		CHash();
		CHash(const CHash &  a);

		///����ڵ�Ĵ�С
		size_t GetSizeOfHashNode();

		///����ͷ���Ĵ�С
		size_t GetSizeOfHashHead();

		///��ʼ��hash�����ڲ��ṹ
		int Initialize();

		/// �����Ϊnode�Ľڵ���뵽free����
		int InsertFreeList(int node);

		/// �����Ϊnode�Ľڵ���뵽used����
		int InsertUsedList(int node);

		/// �����Ϊnode�Ľڵ���뵽��bucket��Ͱ��
		int InsertBucketList(int bucket, int node);

		/// �����Ϊnode�Ľڵ��free������ɾ��
		int RemoveNodeFromFreeList(int node);

		/// �����Ϊnode�Ľڵ��used������ɾ��
		int RemoveNodeFromUsedList(int node);

		/// �����Ϊnode�Ľڵ��Ͱ��ɾ��
		int RemoveNodeFromBucketList(int node);

		/**
		 * ��ȡ��1�����нڵ㣬���free������
		 * û�п��нڵ㣬����used�����ĩβ��̭��һ��
		 */
		int GetFreeNode(int & node);

		///���ڵ���ת��Ϊ��Ӧ��ͷָ��
		CHashNode * Node2Ptr(int node);

		///���ڵ��ͷָ��ת��Ϊ��Ӧ�ı��
		int Ptr2Node(void *p );

		///����һ���ڵ���ڲ��ֶ�
		int ResetNode(int);

		///Ĭ�ϵ�hash����
		static unsigned int hash(const unsigned char * keybuf, size_t keysize);

		///Ĭ�ϵ�key����жϺ���
		static bool KeyEqual(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size);

		/**
		 * �ڻ����в���keyΪ(keybuf,keysize)�Ľڵ�
		 * ����ҵ��ˣ��ڵ��ű��浽node,�����ڵ�Ͱ��ű��浽bucketid
		 * ����ֵ��0-�ҵ��� 1-û���ҵ� ����-ʧ��
		 */
		int Find(int & node, int & bucketid, const unsigned char * keybuf, size_t keysize);

	public:
		///<���캯��
		CHash(size_t iBucketNum, size_t iMaxKeySize, size_t iMaxValSize,
				unsigned int (*fHash)(const unsigned char * keybuf, size_t keysize) ,
				bool (*fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size) ,
				int (*fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) );

		///��һ���ڴ��������
		int MemAttach(void * ptr, size_t size, bool bCreat = false);

		///����һ��key-val�����key��ȵ�key-val���Ѿ����ڣ���ô������
		int Insert(const unsigned char * keybuf, size_t keysize,
				const unsigned char * valbuf, size_t valsize);

		///����ָ��key��value
		int Update(const unsigned char * keybuf, size_t keysize,
				const unsigned char * valbuf, size_t valsize);

		///ɾ��ָ��key��key-val��
		int Remove(const unsigned char * keybuf, size_t keysize);

		/**
		 * ����ָ��key������ҵ�����0��������Ӧ��val���浽(valbuf,pvalsize)
		 * ���valbuf���Ȳ��������ضϣ�����2
		 * ���û���ҵ�������1
		 * ʧ�ܷ��ظ���
		 */
		int Find(const unsigned char * keybuf, size_t keysize,
				unsigned char * valbuf, size_t * pvalsize);
		int Find(const unsigned char * keybuf, size_t keysize, void * * ppVal);
		/**
		 * �ж�ָ��key��key-val���Ƿ����
		 * ���ڷ���0
		 * �����ڷ���1
		 * ʧ�ܷ��ظ���
		 */
		int Exist(const unsigned char * keybuf, size_t keysize);

		/**
		 *  ɨ��used����
		 *  ������һ������ָ��, ��ÿ���ڵ���øú���
		 */
		void ScanUsedNode( void (*fScan)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) );

		void Dump();

		/**
		 * ��hash���ڲ��ṹ�������Խ��м��
		 * ��������0
		 * �������ظ�������Ҫ����Initialize
		 */
		int Verify();


};
///////////////////////////////////////////////////////////////////////////
// c���Խӿ�
typedef struct 
{   
	volatile unsigned char head[100];
} THashHead;

extern "C" int hash_init( THashHead * pstHead,
		size_t iBucketNum, size_t iMaxKeySize, size_t iMaxValSize,
		unsigned int (*fHash)(const unsigned char * keybuf, size_t keysize) ,
		bool (*fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size),
		int (*fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) );

extern "C" int hash_mem_attach(THashHead * pstHead, void * ptr, size_t size, bool bCreat);

extern "C" int hash_insert(THashHead * pstHead, 
		const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize);

extern "C" int hash_update(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		const unsigned char * valbuf, size_t valsize);

///ɾ��ָ��key��key-val��
extern "C" int hash_remove(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize);

/**
 * ����ָ��key������ҵ�����0��������Ӧ��val���浽(valbuf,pvalsize)
 * ���valbuf���Ȳ��������ضϣ�����2
 * ���û���ҵ�������1
 * ʧ�ܷ��ظ���
 */
extern "C" int hash_find(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		unsigned char * valbuf, size_t * pvalsize);
extern "C" int hash_find2(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		void * * ppVal);
/**
 * �ж�ָ��key��key-val���Ƿ����
 * ���ڷ���0
 * �����ڷ���1
 * ʧ�ܷ��ظ���
 */ 
extern "C" int hash_exist(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize);
extern "C" void hash_ScanUsedNode( THashHead * pstHead,
			void (*fScan)(const unsigned char * key, 
			size_t keysize, 
			const unsigned char * val, 
			size_t valsize) );
#endif
