#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "chash.h"
#include "fdisk.h"

static THashHead hash;
static TBuffer g_tmp_buffer;

TProcess * g_current = NULL;
TProcess g_proc;


static void wait_on_buffer(TBuffer  * buffer);
void buffer_release(TBuffer* buf);

static unsigned int fHash(const unsigned char * keybuf, size_t keysize) 
{
	return *(uint32_t*)keybuf;
}
static int fKeyEqual(const unsigned char * key1, size_t key1size, const unsigned char * key2, size_t key2size)
{
	uint32_t t1, t2;
	t1 = *(uint32_t*)key1;
	t2 = *(uint32_t*)key2;

	return t1== t2;
}
/*
 * < 0����
 * == 0 ���Ա���̭Ų������
 * �������ܱ���̭
 */
static int fNodeSwapOut(const unsigned char * key, size_t keysize, const unsigned char * val, size_t valsize)
{
	TBuffer * buf = (TBuffer*)val;

	if (buf->flags & BUFFER_FLAG_LOCKED)
	{
		return 1;
	}
	if (buf->flags & BUFFER_FLAG_DIRTY)
	{
		buf->flags = buf->flags | BUFFER_FLAG_LOCKED;	
		if (hd_add_request( buf, buf->abs_sect, WIN_WRITE, g_current->pid) )
		{
			panic2("%s, %d:hd_add_request failed!", __FILE__, __LINE__);
		}
		g_current->status = PROC_STATUS_WAITING;
		schedule();	/*���������̿�ʼ���У���ǰ���̵ȴ�IO���*/
		buffer_release(buf);
	}
	if (buf->abs_sect >= (2+NSECT_FOR_INODE_BMP+NSECT_FOR_SECT_BMP)) /*�����顢λͼ���ܶ���*/
	{
		/*������*/
		buf->flags = 0;
		buf->abs_sect = 0;
		buf->wait = NULL;
		return 0;
	}
	else
	{
		return 1;
	}
}
static void fScan(const unsigned char * key,  size_t keysize,                                     
		const unsigned char * val,                                                                              
		size_t valsize) 
{
	TBuffer * buf = (TBuffer*)val;

	//printk("sync %d begin\n", buf->abs_sect);

	if (buf->flags & BUFFER_FLAG_DIRTY)
	{
		wait_on_buffer(buf);
		buf->flags = buf->flags | BUFFER_FLAG_LOCKED;	
		buf->locker_owner_pid = g_current->pid;
		if (hd_add_request( buf, buf->abs_sect, WIN_WRITE, g_current->pid) )
		{
			panic2("%s, %d:hd_add_request failed!", __FILE__, __LINE__);
		}
		g_current->status = PROC_STATUS_WAITING;
		schedule();	/*���������̿�ʼ���У���ǰ���̵ȴ�IO���*/
		buffer_release(buf);
	}
	//printk("sync %d end\n", buf->abs_sect);
}
void buffer_init(uint32_t buffer_start, uint32_t buffer_size);
void fs_init()
{
	static buffers[3000000];
	/*��BUFFER_ORG��FIRST_PROC_ORGһ���ڴ������ļ�ϵͳ��inode/buffer������*/
	uint32_t start = buffers;
	uint32_t size = 3000000;

	inode_init(0,0);
	buffer_init(start, size);
}
void buffer_sync()
{
	/*�����нڵ�ɨ��һ�飬����dirty��־��bufferˢ�µ�����*/
	hash_ScanUsedNode( &hash, fScan);
}

void buffer_init(uint32_t buffer_start, uint32_t buffer_size)
{
	TBuffer * bh = NULL;
	uint32_t abs_sect;
	int iret, i;

	if (hash_init( &hash, 1024, sizeof(uint32_t), sizeof(TBuffer), 
				fHash, 
				fKeyEqual,
				fNodeSwapOut))
	{
		panic2("%s:%d:hash_init() failed!", __FILE__, __LINE__);
	}
	if (hash_mem_attach(&hash, (void*)buffer_start, buffer_size, 1))
	{
		panic2("%s:%d:hash_mem_attach() failed!", __FILE__, __LINE__);
	}
	/*��ȡ������*/
	{
		abs_sect = 1;
		g_tmp_buffer.flags = 0;
		g_tmp_buffer.abs_sect = abs_sect;
		g_tmp_buffer.wait = NULL;
		if (hd_read_sync(abs_sect, g_tmp_buffer.data) != 0)
		{
			panic2("%s %d: read super block failed!\n", __FILE__, __LINE__);
		}
		g_tmp_buffer.flags = BUFFER_FLAG_UPTODATE;
		if ( iret = hash_insert(&hash,
					(const unsigned char*)&abs_sect, sizeof(abs_sect),
					(const unsigned char*)&g_tmp_buffer, sizeof(g_tmp_buffer)) )
		{
			panic2("%s, %d:hash_insert() failed!, iret=%d", __FILE__, __LINE__, iret);
		}
	}
	/*��ȡinodeλͼ*/
	for (i = 0; i < NSECT_FOR_INODE_BMP; ++i)
	{
		abs_sect = 2+i;
		g_tmp_buffer.flags = 0;
		g_tmp_buffer.abs_sect = abs_sect;
		g_tmp_buffer.wait = NULL;
		if (hd_read_sync(abs_sect, g_tmp_buffer.data) != 0)
		{
			panic2("%s %d: read super block failed!\n", __FILE__, __LINE__);
		}
		g_tmp_buffer.flags = BUFFER_FLAG_UPTODATE;
		if (hash_insert(&hash,
					(const unsigned char*)&abs_sect, sizeof(abs_sect),
					(const unsigned char*)&g_tmp_buffer, sizeof(g_tmp_buffer)) )
		{
			panic2("%s, %d:hash_insert() failed!", __FILE__, __LINE__);
		}
	}
	/*��ȡsectλͼ*/
	for (i = 0; i < NSECT_FOR_SECT_BMP; ++i)
	{
		abs_sect = 2+NSECT_FOR_INODE_BMP+i;
		g_tmp_buffer.flags = 0;
		g_tmp_buffer.abs_sect = abs_sect;
		g_tmp_buffer.wait = NULL;
		if (hd_read_sync(abs_sect, g_tmp_buffer.data) != 0)
		{
			panic2("%s %d: read super block failed!\n", __FILE__, __LINE__);
		}
		g_tmp_buffer.flags = BUFFER_FLAG_UPTODATE;
		if (hash_insert(&hash,
					(const unsigned char*)&abs_sect, sizeof(abs_sect),
					(const unsigned char*)&g_tmp_buffer, sizeof(g_tmp_buffer)) )
		{
			panic2("%s, %d:hash_insert() failed!", __FILE__, __LINE__);
		}
	}
}

static void wait_on_buffer(TBuffer  * buffer)
{
	_cli();
	while (buffer->flags & BUFFER_FLAG_LOCKED && buffer->locker_owner_pid != g_current->pid)
		sleep_on( (TProcess**)&(buffer->wait));
	/*_sti();*/
}
void buffer_release(TBuffer* buf)
{           
	if (!buf)
		return;
	buf->flags = buf->flags & (~BUFFER_FLAG_LOCKED); /* ���� */
	buf->locker_owner_pid = 0;
	wake_up( (TProcess**)&buf->wait);
}
/*��hash���еõ���Ӧ��buffer�飬����������־*/
TBuffer * buffer_lock(uint32_t abs_sect)
{
	TBuffer * bh = NULL;
	int iret;

	iret = hash_find2(&hash, (const unsigned char*)&abs_sect, sizeof(abs_sect), (void**)&bh);
	if (iret < 0)
	{
		panic2("%s, %d:hash_find2() failed!", __FILE__, __LINE__);
	}
	if (iret == 1) /*ľ���ҵ� */
	{
		g_tmp_buffer.flags = 0;
		g_tmp_buffer.abs_sect = abs_sect;
		g_tmp_buffer.wait = NULL;
		if (iret = hash_insert(&hash,
					(const unsigned char*)&abs_sect, sizeof(abs_sect),
					(const unsigned char*)&g_tmp_buffer, sizeof(g_tmp_buffer)) )
		{
			panic2("%s, %d:hash_insert() failed! abs_sect=%u, iret=%d", __FILE__, __LINE__, abs_sect, iret);
		}
		iret = hash_find2(&hash, (const unsigned char*)&abs_sect, sizeof(abs_sect), (void**)&bh);
		if (iret < 0)
		{
			panic2("%s, %d:hash_find2() failed!", __FILE__, __LINE__);
		}
		if (iret == 1) /*��û�ҵ�!*/
		{
			panic2("%s, %d:BUG!!", __FILE__, __LINE__);
		}
	}
	/*���ˣ�������hash�����ҵ���*/
	/* wait_on_buffer��������������������ɣ����ܱ��ж�*/
	wait_on_buffer(bh);
	bh->flags = bh->flags | BUFFER_FLAG_LOCKED; /*����*/
	bh->locker_owner_pid = g_current->pid;
	return bh;
}


/*��hash���еõ���Ӧ��buffer�飬����������־*/
TBuffer * buffer_read(uint32_t abs_sect)
{
	/*buffer_readȫ�����ǲ���Ӧ��cli ? */

	TBuffer * bh = NULL;
	int iret;

	bh = buffer_lock(abs_sect);


	if (bh->flags & BUFFER_FLAG_UPTODATE ||
			bh->flags & BUFFER_FLAG_DIRTY) /*�Ѿ�����*/
	{
		return bh;
	}
	if (hd_add_request( bh, abs_sect, WIN_READ, g_current->pid) )
	{
		panic2("%s, %d:hd_add_request failed!", __FILE__, __LINE__);
	}
	g_current->status = PROC_STATUS_WAITING;
	schedule();	/*���������̿�ʼ���У���ǰ���̵ȴ�IO���*/
	if (bh->flags & BUFFER_FLAG_UPTODATE) /*�Ѿ�����*/
	{
		return bh;
	}
	panic2("%s, %d: BUG!", __FILE__, __LINE__);
}         

/*����һ���µ�����*/
unsigned int alloc_block()
{
	static uint32_t sect = 0;
	uint32_t abs_sect;
	TBuffer * bh;
	int i, j, k;
	unsigned int res;
	unsigned char c, mask;


	for (i = 0; i < NSECT_FOR_SECT_BMP; 
			++i,  sect = (sect+1) % NSECT_FOR_SECT_BMP ) /* 32����*/
	{
		abs_sect = 2+NSECT_FOR_INODE_BMP+sect;
		bh = buffer_read(abs_sect);
		for (j = 0; j < 512; ++j) 		/*ÿ��512�ֽ�*/
		{
			c = (unsigned char)(bh->data[j]);
			if ( c != 0xff	)
			{
				for (k = 0; k < 8; ++k) 	/*ÿ�ֽ�8λ*/
				{
					mask = (0x1u << k);
					if (  c & mask) 	/* 1��ʾ�Ѿ�ʹ��*/
					{
						continue;
					}
					bh->data[j] = c | mask;
					bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
					buffer_release(bh);
					res = sect*8*512 + j*8 + k;
	//				printk("allocate block#%u\n", res);
					return res;
				}
				panic2("%s %d: BUG!", __FILE__, __LINE__);
			}
		}
		buffer_release(bh);
	}
	return 0;
}
/*�ͷ�һ������������nr���������Ա�ţ���0��ʼ*/
void free_block(uint32_t nr)
{
	uint32_t abs_sect, byte_offset, bit_offset, sect_offset;
	TBuffer * bh;
	unsigned char c, mask;

	if (nr  < FIRST_SECT_NO_FOR_DATA)
	{
		panic2("%s %d: invalid block number:%u\n", 
				__FILE__,
				__LINE__,
				nr);
	}
	sect_offset = nr/(8*512);
	byte_offset = nr / 8 - sect_offset * 512;
	bit_offset = nr % 8;

	if (sect_offset >= NSECT_FOR_SECT_BMP)
	{
		panic2("%s %d: BUG!\n", __FILE__, __LINE__);
	}
	if ( (sect_offset * 8 * 512 + byte_offset * 8 +bit_offset) != nr)
	{
		panic2("%s %d: BUG!\n", __FILE__, __LINE__);
	}

	abs_sect = 2+NSECT_FOR_INODE_BMP + sect_offset;
	bh = buffer_read(abs_sect);
	c = (unsigned char)(bh->data[byte_offset]);
	mask = 0x1u << bit_offset;
	if ( !(c & mask)) /*û�б�ʹ��*/
	{
		buffer_release(bh);
		return;
	}
	mask = ~mask;
	bh->data[byte_offset] = c & mask;
	bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
	buffer_release(bh);
	hash_remove(&hash, (const unsigned char *)&nr, sizeof(uint32_t));
	//printk("free block#%u\n", nr);
	return ;
}
