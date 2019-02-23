#ifndef _FS_H_INCLUDED_
#define _FS_H_INCLUDED_

#include "fdisk.h"


void fs_init();


void buffer_init(uint32_t buffer_start, uint32_t buffer_size);
TBuffer * buffer_lock(uint32_t abs_sect);
void buffer_release(TBuffer* buf);
TBuffer* buffer_read(uint32_t abs_sect);
void buffer_sync();
unsigned int alloc_block();
void free_block(uint32_t nr);


void inode_init(uint32_t start, uint32_t size);
struct m_inode * iget(unsigned int nr, TBuffer ** bh);
unsigned int alloc_inode();
void free_inode(uint32_t nr);
int create_block(struct m_inode * inode, int block);
int bmap(struct m_inode * inode,int block);



void truncate(struct m_inode * inode);
#endif
