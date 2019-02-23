/**
 * @file 通用hash缓存头文件
 * @brief 使用大小相等的节点存放key-val对，节点存在于一片连续的内存空间
 *        内部实现采用大量桶，每个桶是一个双向链表
 *	      当节点不够用时，采用FILO的策略淘汰已用的节点,并可以回调一个用户函数
 *        hash函数、key相等判断函数可配置
 *        不适合key-val大小变化范围很大的情况
 */
#ifndef _HASH_H_INCLUDED_
#define _HASH_H_INCLUDED_

#include "types.h"

#define TRACE(fmt, args...) {;}


/**
 * hash缓冲描述类
 */
class CHash
{
	private:
		enum
		{
			INVALID_BUCKET_ID	= -1, ///<无效的桶编号
			INVALID_NODE		= -1, ///<无效的节点编号
		};

#pragma pack(1)
#pragma align(1)
		///hash缓存的头部结构
		class CHashHead
		{
			private:
				volatile int free_head; ///<空闲节点链表的头指针
				volatile int used_head; ///<已使用节点链表的头指针
				volatile int used_tail; ///<已使用节点链表的尾指针，淘汰用
				volatile unsigned int free_num; ///< 空闲节点个数
				volatile unsigned int used_num; ///< 已使用节点个数
				volatile int bucket[0]; ///<桶数组
				friend class CHash;
		};

		/**
		 *节点的数据结构
		 *一个节点要么在free链表中，要么在used链表中
		 *当一个节点在used链表中的时候，它同时位于某个桶中
		 */
		class CHashNode
		{
			private:
				volatile int iNext; ///< 节点在 free/used链表中，前一个节点的指针
				volatile int iPrev; ///< 节点在 free/used链表中，后一个节点的指针

				volatile int iBucketID; ///<如果一个节点在used链表中，那么iBucketID为,介于 [0,m_iBucketNum)之间的值,否则为INVALID_BUCKET_ID

				volatile int iNextInBucket; ///< 节点在某个桶的链表中，前一个节点的指针
				volatile int iPrevInBucket; ///< 节点在某个桶的链表中，后一个节点的指针

				volatile size_t iKeySize; ///< key值的字节数
				volatile size_t iValSize; ///< val值的字节数

				volatile unsigned char acKey[0]; ///< key
				volatile unsigned char acVal[0]; ///< val
			private:
				///默认构造函数
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
		volatile CHashHead * m_pHead; ///< hash缓存头结构的指针
		volatile size_t m_iBucketNum; ///< 桶的个数
		volatile size_t m_iNodeNum;   ///< 节点总个数
		volatile size_t m_iMaxKeySize; ///< key的最大字节数
		volatile size_t m_iMaxValSize; ///< val的最大字节数
		volatile size_t m_iHeadSize;   ///< hash缓存头部的大小
		volatile size_t m_iNodeSize;   ///< 节点的大小

		///用于计算hash值的函数
		unsigned int (*m_fHash)(const unsigned char * keybuf, size_t keysize);

		///用于比较两个key是否相等的函数
		bool (*m_fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size);

		///当一个正在使用的节点被淘汰出来时候，需要进行的额外操作
		int (*m_fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize);


	private:
		CHash & operator=(const CHash &	 a);
		bool operator==(const CHash &	 a);
		CHash();
		CHash(const CHash &  a);

		///计算节点的大小
		size_t GetSizeOfHashNode();

		///计算头部的大小
		size_t GetSizeOfHashHead();

		///初始化hash缓存内部结构
		int Initialize();

		/// 将编号为node的节点插入到free链表
		int InsertFreeList(int node);

		/// 将编号为node的节点插入到used链表
		int InsertUsedList(int node);

		/// 将编号为node的节点插入到第bucket个桶中
		int InsertBucketList(int bucket, int node);

		/// 将编号为node的节点从free链表中删除
		int RemoveNodeFromFreeList(int node);

		/// 将编号为node的节点从used链表中删除
		int RemoveNodeFromUsedList(int node);

		/// 将编号为node的节点从桶中删除
		int RemoveNodeFromBucketList(int node);

		/**
		 * 获取有1个空闲节点，如果free链表中
		 * 没有空闲节点，将从used链表的末尾淘汰出一个
		 */
		int GetFreeNode(int & node);

		///将节点编号转化为相应的头指针
		CHashNode * Node2Ptr(int node);

		///将节点的头指针转化为相应的编号
		int Ptr2Node(void *p );

		///重置一个节点的内部字段
		int ResetNode(int);

		///默认的hash函数
		static unsigned int hash(const unsigned char * keybuf, size_t keysize);

		///默认的key相等判断函数
		static bool KeyEqual(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size);

		/**
		 * 在缓存中查找key为(keybuf,keysize)的节点
		 * 如果找到了，节点编号保存到node,其所在的桶编号保存到bucketid
		 * 返回值：0-找到了 1-没有找到 其他-失败
		 */
		int Find(int & node, int & bucketid, const unsigned char * keybuf, size_t keysize);

	public:
		///<构造函数
		CHash(size_t iBucketNum, size_t iMaxKeySize, size_t iMaxValSize,
				unsigned int (*fHash)(const unsigned char * keybuf, size_t keysize) ,
				bool (*fKeyEqual)(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size) ,
				int (*fNodeSwapOut)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) );

		///与一块内存关联起来
		int MemAttach(void * ptr, size_t size, bool bCreat = false);

		///插入一对key-val，如果key相等的key-val对已经存在，那么将覆盖
		int Insert(const unsigned char * keybuf, size_t keysize,
				const unsigned char * valbuf, size_t valsize);

		///更新指定key的value
		int Update(const unsigned char * keybuf, size_t keysize,
				const unsigned char * valbuf, size_t valsize);

		///删除指定key的key-val对
		int Remove(const unsigned char * keybuf, size_t keysize);

		/**
		 * 查找指定key，如果找到返回0，并将对应的val保存到(valbuf,pvalsize)
		 * 如果valbuf长度不够，将截断，返回2
		 * 如果没有找到，返回1
		 * 失败返回负数
		 */
		int Find(const unsigned char * keybuf, size_t keysize,
				unsigned char * valbuf, size_t * pvalsize);
		int Find(const unsigned char * keybuf, size_t keysize, void * * ppVal);
		/**
		 * 判断指定key的key-val对是否存在
		 * 存在返回0
		 * 不存在返回1
		 * 失败返回负数
		 */
		int Exist(const unsigned char * keybuf, size_t keysize);

		/**
		 *  扫描used链表
		 *  参数是一个函数指针, 对每个节点调用该函数
		 */
		void ScanUsedNode( void (*fScan)(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize) );

		void Dump();

		/**
		 * 对hash的内部结构的完整性进行检查
		 * 完整返回0
		 * 其他返回负数，需要重新Initialize
		 */
		int Verify();


};
///////////////////////////////////////////////////////////////////////////
// c语言接口
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

///删除指定key的key-val对
extern "C" int hash_remove(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize);

/**
 * 查找指定key，如果找到返回0，并将对应的val保存到(valbuf,pvalsize)
 * 如果valbuf长度不够，将截断，返回2
 * 如果没有找到，返回1
 * 失败返回负数
 */
extern "C" int hash_find(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		unsigned char * valbuf, size_t * pvalsize);
extern "C" int hash_find2(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize,
		void * * ppVal);
/**
 * 判断指定key的key-val对是否存在
 * 存在返回0
 * 不存在返回1
 * 失败返回负数
 */ 
extern "C" int hash_exist(THashHead * pstHead,
		const unsigned char * keybuf, size_t keysize);
extern "C" void hash_ScanUsedNode( THashHead * pstHead,
			void (*fScan)(const unsigned char * key, 
			size_t keysize, 
			const unsigned char * val, 
			size_t valsize) );
#endif
