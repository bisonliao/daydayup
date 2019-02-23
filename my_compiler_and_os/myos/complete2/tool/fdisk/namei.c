#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "fdisk.h"

static int match(const struct dir_entry * ent, const char * name2, int namelen)
{
	int i;
	if (namelen > NAME_LEN)
	{
		namelen = NAME_LEN;
	}
	for (i = 0; i < namelen; ++i)
	{
		if (ent->name[i] != name2[i])
		{
			return 0;
		}
	}
	if (i < NAME_LEN)
	{
		if (ent->name[i] != '\0')
		{
			return 0;	
		}
	}
	return 1;
}
/**
 * ��Ŀ¼dir�в���Ŀ¼��name, ���ظ�Ŀ¼�����ڸ��ٻ�����ָ��, ppentry���ظ�Ŀ¼��ָ��
 * ���û���ҵ�������NULL
 * �ǵ�Ҫbuffer_release���صĸ��ٻ����ָ��
 */
static TBuffer * find_entry(struct m_inode * dir, const char * name, int namelen, struct dir_entry ** ppentry)
{
	int i, j, entries;
	TBuffer * bh = NULL;
	struct dir_entry * d_ent;
	uint32_t block, sect_no;

	if (namelen > NAME_LEN)
	{
		namelen = NAME_LEN;
	}

	entries = dir->i_size / sizeof(struct dir_entry);
	//printk("entries=%d\n", entries);
	*ppentry = NULL;
#if 0
	for (i = 0; i < entries; ++i)
	{
		block = i / DIR_ENTRY_NR_PER_BLOCK;
		sect_no =  bmap(dir, block);
		if (sect_no == 0)
		{
			panic2("%s %d: BUG! hollow not allowed in directory, %d", __FILE__, __LINE__, block);
		}
		bh = buffer_read(sect_no);

		d_ent = (struct dir_entry*)(bh->data);
		d_ent += (i % DIR_ENTRY_NR_PER_BLOCK);
		//printk("%d->%c%c\n", i, d_ent->name[0], d_ent->name[1]);
		if ( match(d_ent, name, namelen) )
		{
			*ppentry = d_ent;
			//printk("!@#$$%^&(\n");
			return bh;
		}
		buffer_release(bh);
	}
#else
	for (i = 0; i < entries; )
	{
		block = i / DIR_ENTRY_NR_PER_BLOCK;
		sect_no =  bmap(dir, block);
		if (sect_no == 0)
		{
			panic2("%s %d: BUG! hollow not allowed in directory, %d", __FILE__, __LINE__, block);
		}
		bh = buffer_read(sect_no);

		d_ent = (struct dir_entry*)(bh->data);

		for (j = 0; j < DIR_ENTRY_NR_PER_BLOCK; ++j, ++i, ++d_ent)
		{
			if (i >= (block * DIR_ENTRY_NR_PER_BLOCK) && 
					j >= (entries % DIR_ENTRY_NR_PER_BLOCK))
			{
				break;
			}
			//printk("%d->%c%c\n", i, d_ent->name[0], d_ent->name[1]);
			if ( match(d_ent, name, namelen) )
			{
				*ppentry = d_ent;
				return bh;
			}
		}
		buffer_release(bh);
	}

#endif
	return NULL;
}
/**
 * ��Ŀ¼dir�����Ŀ¼��name, ���ظ�Ŀ¼�����ڸ��ٻ�����ָ��, ppentry���ظ�Ŀ¼��ָ��
 * �ǵ�Ҫbuffer_release���صĸ��ٻ����ָ��
 */
static TBuffer * add_entry(struct m_inode * dir, const char * name, int namelen, struct dir_entry ** ppentry)
{
	int i, j, entries;
	TBuffer * bh = NULL;
	struct dir_entry * d_ent;
	uint32_t block, sect_no;
	uint32_t new_inode;


	if (namelen > NAME_LEN)
	{
		namelen = NAME_LEN;
	}
	/*�ȷ���һ��inode��˵*/
	new_inode = alloc_inode();
	if (new_inode == 0)
	{
		return NULL;
	}

	entries = dir->i_size / sizeof(struct dir_entry);
	*ppentry = NULL;


#if 0
	for (i = 0; i < entries;++i)
	{
		block = i / DIR_ENTRY_NR_PER_BLOCK;
		sect_no =  bmap(dir, block);
		if (sect_no == 0)
		{
			panic2("%s %d: BUG! hollow not allowed in directory", __FILE__, __LINE__);
		}
		bh = buffer_read(sect_no);

		d_ent = (struct dir_entry*)(bh->data);
		d_ent += i % DIR_ENTRY_NR_PER_BLOCK;
		if (d_ent->name[0] == '\0') /*�ҵ�һ������,��������*/
		{
			memcpy(d_ent->name, name, namelen);
			if (namelen < NAME_LEN)
			{
				d_ent->name[namelen] = '\0';
			}
			d_ent->inode = new_inode;
			*ppentry = d_ent;
			return bh;
		}
		buffer_release(bh);
	}
#else
	for (i = 0; i < entries; )
	{
		block = i / DIR_ENTRY_NR_PER_BLOCK;
		sect_no =  bmap(dir, block);
		if (sect_no == 0)
		{
			panic2("%s %d: BUG! hollow not allowed in directory, %d", __FILE__, __LINE__, block);
		}
		bh = buffer_read(sect_no);

		d_ent = (struct dir_entry*)(bh->data);

		for (j = 0; j < DIR_ENTRY_NR_PER_BLOCK; ++j, ++i, ++d_ent)
		{
			if (i >= (block * DIR_ENTRY_NR_PER_BLOCK) && 
					j >= (entries % DIR_ENTRY_NR_PER_BLOCK))
			{
				break;
			}
			if (d_ent->name[0] == '\0') /*�ҵ�һ������,��������*/
			{
				memcpy(d_ent->name, name, namelen);
				if (namelen < NAME_LEN)
				{
					d_ent->name[namelen] = '\0';
				}
				d_ent->inode = new_inode;
				*ppentry = d_ent;
				++dir->i_entry_nr;
				return bh;
			}
		}
		buffer_release(bh);
	}

#endif
	/*׷�� */
	if (dir->i_size + sizeof(struct dir_entry) > MAX_FILE_SIZE)
	{
		/*�����ļ�����С*/
		free_inode(new_inode);
		return NULL;
	}
	dir->i_size += sizeof(struct dir_entry);
	++dir->i_entry_nr;
	++entries;
	block = entries / DIR_ENTRY_NR_PER_BLOCK ;
	sect_no = create_block(dir, block); /*������ܻ��·���һ��block��Ҳ���ܲ��·���*/
	if (sect_no == 0)
	{
		free_inode(new_inode);
		return NULL;
	}
	bh = buffer_read(sect_no);
	d_ent = (struct dir_entry*)(bh->data);
	d_ent += (entries - block * DIR_ENTRY_NR_PER_BLOCK - 1);
	memcpy(d_ent->name, name, namelen);
	if (namelen < NAME_LEN)
	{
		d_ent->name[namelen] = '\0';
	}
	d_ent->inode = new_inode;
	*ppentry = d_ent;
	return bh;
}
/*
 * ��·����ֳ�basename��dirname
 * �ɹ�����0��ʧ�ܷ�������
 */
int pathname_split(const char * pathname, int *pathname_len, const char ** basename, int * basename_len)
{
	if (*pathname_len < 1)
	{
		return -1;
	}
	if (pathname[*pathname_len-1] == '/')
	{
		--(*pathname_len);
	}
	*basename_len = 0;
	while (*pathname_len > 0)
	{
		if (pathname[*pathname_len-1] != '/')
		{
			--(*pathname_len);
			++(*basename_len);
		}
		else
		{
			break;
		}
	}
	*basename = &(pathname[*pathname_len]);
	return 0;
}
int subdir(const char ** pathname, int *pathname_len, char *name)
{
	int name_len = 0;

	while ( *pathname_len > 0)
	{
		if ( (*pathname)[0] != '/')
		{
			if (name_len >= NAME_LEN) 
			{
				return -1;
			}
			name[name_len++] = (*pathname)[0];
			*pathname += 1;
			*pathname_len -= 1;
		}
		else
		{
			*pathname += 1;
			*pathname_len -= 1;
			break;
		}
	}
	if (name_len < NAME_LEN)
	{
		name[name_len] = '\0';
	}
	return name_len;
}
/**
 * ��pathname·��һ·���ң��ҵ�����inode��������inode������
 * �ɹ�����0�� û���ҵ�����1�� ������ <0
 */
int namei(const char * pathname, int pathname_len, uint32_t * inode)
{
	char name[NAME_LEN];
	int name_len;
	uint32_t inode_nr;
	TBuffer * parent_bh, *child_bh;
	struct m_inode * parent_inode;
	struct dir_entry * entry;

	if (pathname_len < 1)
	{
		return -1;
	}
	if (pathname[0] == '/')
	{
		//inode_nr = g_current->root_inode;		
		inode_nr = 0;
		pathname = pathname + 1;
		pathname_len -= 1;
	}
	else if (pathname_len > 2 && pathname[0] == '.' && pathname[1] == '/')
	{
		//inode_nr = g_current->cwd_inode;		
		inode_nr = 0;
		pathname = pathname + 2;
		pathname_len -= 2;
	}
	else
	{
		//inode_nr = g_current->cwd_inode;		
		inode_nr = 0;
	}

	while (1)
	{
		/*��ֳ�һ��Ŀ¼��*/
		name_len = subdir(&pathname, &pathname_len, name);
		if (name_len < 0)
		{
			return -1;
		}
		if (name_len == 0)
		{
			*inode = inode_nr;
			return 0;
		}
		/*��inode�����ݶ�����*/
		parent_inode = iget(inode_nr, &parent_bh); 
		if (parent_inode == NULL)
		{
			return -1;
		}
		if (parent_inode->i_type != FILE_TYPE_DIR)
		{
			buffer_release(parent_bh);
			return -1;
		}
		/*��inodeָʾ��Ŀ¼�в���Ŀ¼��*/
		//printk("[%s] %d\n", name, name_len);
		child_bh = find_entry(parent_inode, name, name_len, &entry);
		buffer_release(parent_bh);
		if (child_bh == NULL)
		{
			return 1; /*ľ���ҵ�*/
		}
		/*������һ��Ŀ¼�� */
		inode_nr = entry->inode;
		buffer_release(child_bh);
	}
	return 0; /*�ҵ��� */
}
uint32_t sys_mkdir(const char * _pathname)
{
	/*int mkdir(const char *pathname);*/
	const char * pathname;
	uint32_t pathname_len;
	uint32_t inode_nr;
	const char * name;
	uint32_t name_len = 0;
	uint32_t new_inode;
	TBuffer * parent_bh, *child_bh;
	struct m_inode * parent_inode, *child_inode;
	int iret;
	struct dir_entry * entry;

	pathname = (const char *)_pathname;
	pathname_len = strlen(pathname);


	/*��ֳ�basename/dirname*/
	if (pathname_split(pathname, &pathname_len, &name, &name_len)!= 0)
	{
		return -1;
	}
	/*��������һ��Ŀ¼*/
	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -2;
	}
	/*�޸���һ��Ŀ¼��inode��Ŀ¼��*/
	parent_inode = iget(inode_nr, &parent_bh); 
	if (parent_inode == NULL)
	{
		return -3;
	}
	child_bh = find_entry(parent_inode, name, name_len, &entry);
	if (child_bh != NULL)/* exists already*/
	{
		buffer_release(parent_bh);
		buffer_release(child_bh);
		return -4;
	}
	child_bh =  add_entry(parent_inode, name, name_len, &entry);
	if (child_bh == NULL)
	{
		buffer_release(parent_bh);
		return -5;
	}
	child_bh->flags |= BUFFER_FLAG_DIRTY;
	new_inode = entry->inode;
	parent_inode->i_atime = parent_inode->i_ctime = current_time();
	parent_bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(parent_bh);
	buffer_release(child_bh);

	/*�޸������ڵ������*/
	child_inode = iget(new_inode, &child_bh); 
	memset(child_inode, 0, sizeof(struct m_inode));
	child_inode->i_atime = child_inode->i_ctime = current_time();
	child_inode->i_type = FILE_TYPE_DIR;
	buffer_release(child_bh);
	return 0;
}
uint32_t sys_mkfile(const char *pathname)
{
	/*int mkfile(const char *pathname);*/
	uint32_t pathname_len;
	uint32_t inode_nr;
	const char * name;
	uint32_t name_len = 0;
	uint32_t new_inode;
	TBuffer * parent_bh, *child_bh;
	struct m_inode * parent_inode, *child_inode;
	int iret;
	struct dir_entry * entry;

	pathname_len = strlen(pathname);

	/*��ֳ�basename/dirname*/
	if (pathname_split(pathname, &pathname_len, &name, &name_len)!= 0)
	{
		return -1;
	}
	/*��������һ��Ŀ¼*/
	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -2;
	}
	/*�޸���һ��Ŀ¼��inode��Ŀ¼��*/
	parent_inode = iget(inode_nr, &parent_bh); 
	if (parent_inode == NULL)
	{
		return -3;
	}
	child_bh = find_entry(parent_inode, name, name_len, &entry);
	if (child_bh != NULL)/* exists already*/
	{
		buffer_release(parent_bh);
		buffer_release(child_bh);
		return -4;
	}
	child_bh =  add_entry(parent_inode, name, name_len, &entry);
	if (child_bh == NULL)
	{
		buffer_release(parent_bh);
		return -5;
	}
	child_bh->flags |= BUFFER_FLAG_DIRTY;
	new_inode = entry->inode;
	parent_inode->i_atime = parent_inode->i_ctime = current_time();
	buffer_release(parent_bh);
	buffer_release(child_bh);

	/*�޸������ڵ������*/
	child_inode = iget(new_inode, &child_bh); 
	memset(child_inode, 0, sizeof(struct m_inode));
	child_inode->i_atime = child_inode->i_ctime = current_time();
	child_inode->i_type = FILE_TYPE_REGULAR;
	buffer_release(child_bh);
	return 0;
}
uint32_t sys_rmdir(const char *pathname)
{
	/*int rmdir(const char *pathname);*/
	uint32_t pathname_len;
	uint32_t inode_nr;
	const char * name;
	uint32_t name_len = 0;
	uint32_t new_inode;
	TBuffer * parent_bh, *child_bh;
	struct m_inode * parent_inode, *child_inode;
	int iret;
	struct dir_entry * entry;

	pathname_len = strlen(pathname);

	//printk(">>>[%s]\n", pathname);

	/*���Ŀ¼��Ŀ¼�����������*/
	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		//printk("%d\n", iret);
		return -__LINE__;
	}
	child_inode = iget(inode_nr, &child_bh); 
	if (child_inode == NULL)
	{
		return -__LINE__;
	}
	if (child_inode->i_type != FILE_TYPE_DIR)
	{
		buffer_release(child_bh);
		return -__LINE__;	
	}
	if (child_inode->i_entry_nr != 0)
	{
		buffer_release(child_bh);
		return -__LINE__;	
	}
	truncate(child_inode); /*����ļ�*/
	child_bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(child_bh);

	/*��ֳ�basename/dirname*/
	if (pathname_split(pathname, &pathname_len, &name, &name_len)!= 0)
	{
		return -__LINE__;
	}
	/*��������һ��Ŀ¼*/
	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -__LINE__;
	}
	/*�޸���һ��Ŀ¼��inode��Ŀ¼��*/
	parent_inode = iget(inode_nr, &parent_bh); 
	if (parent_inode == NULL)
	{
		return -__LINE__;
	}
	child_bh = find_entry(parent_inode, name, name_len, &entry);
	if (child_bh == NULL)/* no exist*/
	{
		buffer_release(parent_bh);
		panic2("%s %d: BUG!", __FILE__, __LINE__);
	}
	free_inode(entry->inode);
	entry->inode = 0;
	entry->name[0] = '\0';
	child_bh->flags |= BUFFER_FLAG_DIRTY;
	parent_inode->i_atime = parent_inode->i_ctime = current_time();
	--parent_inode->i_entry_nr;
	parent_bh->flags |= BUFFER_FLAG_DIRTY;

	buffer_release(parent_bh);
	buffer_release(child_bh);

	return 0;
}
uint32_t sys_readdir(const char * pathname, uint32_t index, char * buf)
{
	/* int readdir(const char * pathname, uint32_t index, char * buf); */
	uint32_t pathname_len;

	uint32_t inode_nr;
	TBuffer * bh;
	struct m_inode  *inode;
	struct dir_entry * ent;
	int i, iret;
	uint32_t block_index, block_offset;

	pathname_len = strlen(pathname);

	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -3;
	}
	inode = iget(inode_nr, &bh); 
	if (inode == NULL)
	{
		return -4;
	}
	if (inode->i_type != FILE_TYPE_DIR)
	{
		buffer_release(bh);
		return -5;	
	}
	//printk("i_size=%d\n", inode->i_size);
	if (index * sizeof(struct dir_entry) >= inode->i_size)
	{
		buffer_release(bh);
		return 0;
	}

	block_index = index /  DIR_ENTRY_NR_PER_BLOCK;
	block_offset = index % DIR_ENTRY_NR_PER_BLOCK ;
	block_index =  bmap(inode,block_index);
	if (block_index == 0)
	{
		buffer_release(bh);
		return 0;
	}
	buffer_release(bh);
	bh = buffer_read(block_index);
	if (bh == NULL)
	{
		return -__LINE__;
	}
	ent = (struct dir_entry*)bh->data;
	ent += block_offset;
	for (i = 0; i < NAME_LEN; ++i)
	{
		buf[i] = ent->name[i];
	}
	buf[i] = '\0';
	buffer_release(bh);
	return 1;
}
uint32_t sys_rmfile(const char *pathname)
{
	/*int rmdir(const char *pathname);*/
	uint32_t pathname_len;
	uint32_t inode_nr;
	const char * name;
	uint32_t name_len = 0;
	uint32_t new_inode;
	TBuffer * parent_bh, *child_bh;
	struct m_inode * parent_inode, *child_inode;
	int iret;
	struct dir_entry * entry;

	pathname_len = strlen(pathname);

	/*����ļ�������*/
	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		buffer_release(child_bh);
		return -3;
	}
	child_inode = iget(inode_nr, &child_bh); 
	if (child_inode == NULL)
	{
		return -4;
	}
	if (child_inode->i_type != FILE_TYPE_REGULAR)
	{
		buffer_release(child_bh);
		return -5;	
	}
	if (child_inode->i_count > 0) /*���н�����ʹ���ã�����ɾ��*/
	{
		buffer_release(child_bh);
		return -__LINE__;
	}
	truncate(child_inode); /*����ļ�*/
	child_bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(child_bh);

	/*��ֳ�basename/dirname*/
	if (pathname_split(pathname, &pathname_len, &name, &name_len)!= 0)
	{
		return -1;
	}
	/*��������һ��Ŀ¼*/
	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -2;
	}
	/*�޸���һ��Ŀ¼��inode��Ŀ¼��*/
	parent_inode = iget(inode_nr, &parent_bh); 
	if (parent_inode == NULL)
	{
		return -3;
	}
	child_bh = find_entry(parent_inode, name, name_len, &entry);
	if (child_bh == NULL)/* no exist*/
	{
		buffer_release(parent_bh);
		panic2("%s %d: BUG!", __FILE__, __LINE__);
	}
	free_inode(entry->inode);
	entry->inode = 0;
	entry->name[0] = '\0';
	child_bh->flags |= BUFFER_FLAG_DIRTY;
	parent_inode->i_atime = parent_inode->i_ctime = current_time();
	--parent_inode->i_entry_nr;
	parent_bh->flags |= BUFFER_FLAG_DIRTY;

	buffer_release(parent_bh);
	buffer_release(child_bh);

	return 0;
}
static int file_read(uint32_t inode_nr,  uint32_t offset, uint32_t len, unsigned char * buf)
{
	TBuffer * bh, *bh2;
	struct m_inode  *inode;
	int i, iret, k;
	uint32_t block_index, block_offset, block_id;

	inode = iget(inode_nr, &bh); 
	if (inode == NULL)
	{
		return -__LINE__;
	}
	//printk("                        offset %u i_size=%u                        \n", offset, inode->i_size);
	if (inode->i_type != FILE_TYPE_REGULAR)
	{
		buffer_release(bh);
		return -__LINE__;
	}
	k = 0;
	for (i = 0; i < len;)
	{
		if ( (offset + i) >= inode->i_size)
		{
			break;
		}
		block_index = (offset+i) /  512;
		block_offset = (offset+i) %  512;
		block_id = bmap(inode,block_index);
		if (block_id == 0)
		{
			buffer_release(bh);
			return -1;
		}
		bh2 = buffer_read(block_id);
		do {
			buf[k++] = bh2->data[block_offset++];
			++i;
		}
		while (((offset+i) /  512) == block_index && i < len && (offset + i) < inode->i_size) ;

		buffer_release(bh2);
		bh2 = NULL;
	}
	if (bh2 != NULL) buffer_release(bh2);
	inode->i_atime  = current_time();
	bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(bh);

	return i;
}
static int file_write(uint32_t inode_nr,  uint32_t offset, uint32_t len, 
		const unsigned char * buf, uint8_t flags)
{
	TBuffer * bh, *bh2;
	struct m_inode  *inode;
	int i, iret, k;
	uint32_t block_index, block_offset, block_id;

	inode = iget(inode_nr, &bh); 
	if (inode == NULL)
	{
		return -__LINE__;
	}
	if (inode->i_type != FILE_TYPE_REGULAR)
	{
		buffer_release(bh);
		return -__LINE__;
	}
	if (flags & O_APPEND)
	{
		offset = inode->i_size;
	}
	if (offset > inode->i_size)
	{
		buffer_release(bh);
		return -__LINE__;
	}
	k = 0;
	for (i = 0; i < len;)
	{
		block_index = (offset+i) /  512;
		block_offset = (offset+i) %  512;
		block_id = create_block(inode,block_index);
		if (block_id == 0)
		{
			buffer_release(bh);
			return -__LINE__;
		}
		bh2 = buffer_read(block_id);
		do {
			bh2->data[block_offset++] = buf[k++];
			++i;
		}
		while (((offset+i) /  512) == block_index && i < len) ;
		bh2->flags = bh2->flags | BUFFER_FLAG_DIRTY;
		buffer_release(bh2);
		bh2 = NULL;
	}
	if (bh2 != NULL) buffer_release(bh2);

	if (inode->i_size < offset+len)
	{
		inode->i_size = offset+len;
	}
	//printk("i_size=%u\n", inode->i_size);
	inode->i_atime = inode->i_ctime = current_time();
	bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(bh);

	return i;
}
int sys_read(const char * pathname, unsigned char * buf, uint32_t offset, uint32_t len)
{
	int pathname_len;
	int iret;
	struct m_inode * p_inode;
	uint32_t inode_nr;
	TBuffer *bh;

	pathname_len = strlen(pathname);

	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -3;
	}
	p_inode = iget(inode_nr, &bh);
	if (p_inode == NULL)
	{
		return -4;
	}
	if (p_inode->i_type != FILE_TYPE_REGULAR)
	{
		buffer_release(bh);
		return -5;
	}
	buffer_release(bh);
	iret = file_read(inode_nr,  offset, len, buf);

	return iret;
}

int sys_write(const char * pathname, const unsigned char * buf, uint32_t offset, uint32_t len)
{
	int pathname_len;
	int iret;
	struct m_inode * p_inode;
	uint32_t inode_nr;
	TBuffer *bh;

	pathname_len = strlen(pathname);

	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -3;
	}
	p_inode = iget(inode_nr, &bh);
	if (p_inode == NULL)
	{
		return -4;
	}
	if (p_inode->i_type != FILE_TYPE_REGULAR)
	{
		buffer_release(bh);
		return -5;
	}
	buffer_release(bh);
	iret = file_write(inode_nr,  offset, len, buf, 0);
	/*
	static int file_write(uint32_t inode_nr,  uint32_t offset, uint32_t len, 
		const unsigned char * buf, uint8_t flags); */

	return iret;
}

