#include "const_def.h"
#include "fs.h"
#include "struct.h"


static void read_inode(struct m_inode * inode);
static void write_inode(struct m_inode * inode);

void inode_init(uint32_t start, uint32_t size)
{   
}   



/* block: 0 ��ʼ*/
static int _bmap(struct m_inode * inode,int block,int create)
{
	TBuffer * bh, *bh2;
	int i;

	/*
	 *  0-5ֱ�ӿ�
	 *  6-7 һ�μ�ӿ�
	 *  8-9 ���μ�ӿ�
	 *  ÿ����ӿ�128��ָ��
	 */


	if (block<0)
		panic("_bmap: block<0");
	if (block >= 6+128*2+ (128*128)*2)
		panic("_bmap: block too big");
	/* 0 - 5 ֱ�ӿ�*/
	if (block<6) {
		if (create && !inode->i_zone[block])
			if (inode->i_zone[block]=alloc_block()) {
				inode->i_ctime=current_time();
			}
		return inode->i_zone[block];
	}
	block -= 6;

	/*��һ��һ�μ�ӿ� 6*/
	if (block<128) {
		if (create && !inode->i_zone[6])
		{
			if (inode->i_zone[6]=alloc_block()) {
				inode->i_ctime=current_time();
			}
			if (!inode->i_zone[6])
			{
				return 0;
			}
			//��0
			bh = buffer_lock( inode->i_zone[6]);
			memset(bh->data, 0, sizeof(bh->data));
			bh->flags = bh->flags | BUFFER_FLAG_DIRTY | BUFFER_FLAG_UPTODATE;
			buffer_release(bh);
		}
		bh = buffer_read( inode->i_zone[6]);
		i = ((unsigned long *) (bh->data))[block];
		//printk(">>%s %d, i=%d\n", __FILE__, __LINE__, i);
		if (create && !i)
			if (i=alloc_block()) {
				((unsigned long *) (bh->data))[block]=i;
				bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
			}
		buffer_release(bh);
		//printk(">>%s %d, i=%d\n", __FILE__, __LINE__, i);
		return i;
	}
	block -= 128;
	/*�ڶ���һ�μ�ӿ� 7*/
	if (block<128) {
		if (create && !inode->i_zone[7])
		{
			if (inode->i_zone[7]=alloc_block()) {
				inode->i_ctime=current_time();
			}
			if (!inode->i_zone[7])
			{
				return 0;
			}
			//��0
			bh = buffer_lock( inode->i_zone[7]);
			memset(bh->data, 0, sizeof(bh->data));
			bh->flags = bh->flags | BUFFER_FLAG_DIRTY | BUFFER_FLAG_UPTODATE;
			buffer_release(bh);
		}
		if (!(bh = buffer_read(inode->i_zone[7])))
			return 0;
		i = ((unsigned long *) (bh->data))[block];
		if (create && !i)
			if (i=alloc_block()) {
				((unsigned long *) (bh->data))[block]=i;
				bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
			}
		buffer_release(bh);
		return i;
	}
	block -= 128;
	/* ��һ�����μ�ӿ� 8*/
	if (block < (128*128) )
	{
		if (create && !inode->i_zone[8])
		{
			if (inode->i_zone[8]=alloc_block()) {
				inode->i_ctime=current_time();
			}
			if (!inode->i_zone[8])
			{
				return 0;
			}
			//��0
			bh = buffer_lock( inode->i_zone[8]);
			memset(bh->data, 0, sizeof(bh->data));
			bh->flags = bh->flags | BUFFER_FLAG_DIRTY | BUFFER_FLAG_UPTODATE;
			buffer_release(bh);
		}
		if (!(bh=buffer_read(inode->i_zone[8])))
			return 0;
		i = ((unsigned long *)bh->data)[block>>7];
		if (create && !i)
		{
			if (i=alloc_block()) {
				((unsigned long *) (bh->data))[block>>7]=i;
				bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
			}
			//��0
			bh2 = buffer_lock(i);
			memset(bh2->data, 0, sizeof(bh2->data));
			bh2->flags = bh2->flags | BUFFER_FLAG_DIRTY | BUFFER_FLAG_UPTODATE;
			buffer_release(bh2);
		}
		buffer_release(bh);
		if (!i)
			return 0;
		if (!(bh=buffer_read(i)))
			return 0;
		i = ((unsigned long *)bh->data)[block&127];
		if (create && !i)
		{
			if (i=alloc_block()) {
				((unsigned long *) (bh->data))[block&127]=i;
				bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
			}
			//���ݿ���һ�㲻������0��
		}
		buffer_release(bh);
		return i;
	}
	block -= 128*128;
	/* �ڶ������μ�ӿ� 9*/
	if (block < (128*128) )
	{
		if (create && !inode->i_zone[9])
		{
			if (inode->i_zone[9]=alloc_block()) {
				inode->i_ctime=current_time();
			}
			else
			{
				return 0;
			}
			//��0
			bh = buffer_lock( inode->i_zone[9]);
			memset(bh->data, 0, sizeof(bh->data));
			bh->flags = bh->flags | BUFFER_FLAG_DIRTY | BUFFER_FLAG_UPTODATE;
			buffer_release(bh);
		}
		if (!(bh=buffer_read(inode->i_zone[9])))
			return 0;
		i = ((unsigned long *)bh->data)[block>>7];
		if (create && !i)
		{
			if (i=alloc_block()) {
				((unsigned long *) (bh->data))[block>>7]=i;
				bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
			}
			//��0
			bh2 = buffer_lock(i);
			memset(bh2->data, 0, sizeof(bh2->data));
			bh2->flags = bh2->flags | BUFFER_FLAG_DIRTY | BUFFER_FLAG_UPTODATE;
			buffer_release(bh2);
		}
		buffer_release(bh);
		if (!i)
			return 0;
		if (!(bh=buffer_read(i)))
			return 0;
		i = ((unsigned long *)bh->data)[block&127];
		if (create && !i)
			if (i=alloc_block()) {
				((unsigned long *) (bh->data))[block&127]=i;
				bh->flags = bh->flags | BUFFER_FLAG_DIRTY;
			}
		buffer_release(bh);
		return i;
	}
	return 0;
}

int bmap(struct m_inode * inode,int block)
{
	return _bmap(inode,block,0);
}

int create_block(struct m_inode * inode, int block)
{
	return _bmap(inode,block,1);
}

unsigned int alloc_inode()
{
	uint32_t abs_sect;
	static uint32_t sect;
	TBuffer * bh;
	int i, j, k;
	unsigned char c, mask;


	for (i = 0; i < NSECT_FOR_INODE_BMP; 
			++i, sect = (sect+1)%NSECT_FOR_INODE_BMP) /* 2����*/
	{
		abs_sect = 2+sect;
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
					return sect*8*512 + j*8 + k;
				}
			}
		}
		buffer_release(bh);
	}
	return 0;

}
/*�ͷ�һ��inode��nr��inode��ţ���0��ʼ*/
void free_inode(uint32_t nr)
{
	uint32_t abs_sect, byte_offset, bit_offset, sect_offset;
	TBuffer * bh;
	unsigned char c, mask;

	sect_offset = nr/(8*512);
	byte_offset = nr / 8 - sect_offset * 512;
	bit_offset = nr % 8;

	if (sect_offset >= NSECT_FOR_INODE_BMP)
	{
		panic("%s %d: BUG!\n", __FILE__, __LINE__);
	}
	if (( sect_offset * 8 * 512 + byte_offset * 8 +bit_offset) != nr)
	{
		panic("%s %d: BUG!\n", __FILE__, __LINE__);
	}

	abs_sect = 2 + sect_offset;
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
//	printk("free %u\n", nr);
	return ;
}
/* nr��0��ʼ�� 0 ��ʾ��һ��*/
struct m_inode * iget(unsigned int nr, TBuffer ** bh)
{
	unsigned int abs_sect;
	unsigned int offset;

	if (nr >= MAX_INODE_NR)
	{
		panic("%s %d: invalid nr:%u\n", __FILE__, __LINE__, nr);
	}

	offset = nr % INODE_NR_PER_SECT;
	abs_sect = 2 + NSECT_FOR_INODE_BMP + NSECT_FOR_SECT_BMP;
	abs_sect += (nr+1) / INODE_NR_PER_SECT;

	*bh = buffer_read(abs_sect);
	return ((struct m_inode *)(*bh)->data) + offset;
}
