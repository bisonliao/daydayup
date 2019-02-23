#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "fdisk.h"

static void free_ind(uint32_t block)
{
	TBuffer * bh;
	unsigned int * p;
	int i;

	if (block < FIRST_SECT_NO_FOR_DATA)
	{
		return;
	}
	if (bh=buffer_read(block)) {
		p = (unsigned int *) bh->data;
		for (i=0;i<128;i++,p++)
		{
			if (*p)
			{
				free_block(*p);
				*p = 0;
			}
		}
		bh->flags |= BUFFER_FLAG_DIRTY;
		buffer_release(bh);
	}
	free_block(block);
}

static void free_dind(uint32_t block)
{
	TBuffer * bh;
	unsigned int * p;
	int i;

	if (block < FIRST_SECT_NO_FOR_DATA)
	{
		return;
	}
	if (bh=buffer_read(block)) {
		p = (unsigned int *) bh->data;
		for (i=0;i<128;i++,p++)
		{
			if (*p)
			{
				free_ind(*p);
				*p = 0;
			}
		}
		bh->flags |= BUFFER_FLAG_DIRTY;
		buffer_release(bh);
	}
	free_block(block);
}

void truncate(struct m_inode * inode)
{
	int i;

	for (i=0;i<5;i++)
		if (inode->i_zone[i]) {
			free_block(inode->i_zone[i]);
			inode->i_zone[i]=0;
		}
	free_ind(inode->i_zone[6]);
	free_ind(inode->i_zone[7]);
	free_dind(inode->i_zone[8]);
	free_dind(inode->i_zone[9]);
	for (i =0; i < 10; ++i)
	{
		inode->i_zone[i] = 0;
	}
	inode->i_size = 0;
	inode->i_mtime = inode->i_ctime = current_time();
}

