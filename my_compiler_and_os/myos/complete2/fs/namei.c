#include "fs.h"
#include "global.h"
#include "const_def.h"
#include "struct.h"
#include "types.h"
#include "redefine.h"

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
			panic("%s %d: BUG! hollow not allowed in directory, %d", __FILE__, __LINE__, block);
		}
		bh = buffer_read(sect_no);

		d_ent = (struct dir_entry*)(bh->data);
		d_ent += (i % DIR_ENTRY_NR_PER_BLOCK);
		printk("%d->%c%c\n", i, d_ent->name[0], d_ent->name[1]);
		if ( match(d_ent, name, namelen) )
		{
			*ppentry = d_ent;
			printk("!@#$$%^&(\n");
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
			panic("%s %d: BUG! hollow not allowed in directory, %d", __FILE__, __LINE__, block);
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
			panic("%s %d: BUG! hollow not allowed in directory", __FILE__, __LINE__);
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
			panic("%s %d: BUG! hollow not allowed in directory, %d", __FILE__, __LINE__, block);
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
		if (g_current == NULL) inode_nr = 0;
		else inode_nr = g_current->root_inode;		
		pathname = pathname + 1;
		pathname_len -= 1;
	}
	else if (pathname_len > 2 && pathname[0] == '.' && pathname[1] == '/')
	{
		if (g_current == NULL) inode_nr = 0;
		else inode_nr = g_current->cwd_inode;		
		pathname = pathname + 2;
		pathname_len -= 2;
	}
	else
	{
		 if (g_current == NULL) inode_nr = 0;
		 else inode_nr = g_current->cwd_inode;		
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
uint32_t sys_mkdir(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/*int mkdir(const char *pathname);*/
	uint32_t addr;
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

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
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
uint32_t sys_mkfile(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/*int mkfile(const char *pathname);*/
	uint32_t addr;
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

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
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
uint32_t sys_rmdir(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/*int rmdir(const char *pathname);*/
	uint32_t addr;
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

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
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
		panic("%s %d: BUG!", __FILE__, __LINE__);
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
uint32_t sys_access(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/* int access(const char * pathname, uint32_t mode);*/
    /*��ʱ��ֻ�Ǽ�������*/
	uint32_t addr;
	const char * pathname;
	uint32_t pathname_len;

    uint32_t mode;
	uint32_t inode_nr;
	TBuffer * bh;
	struct m_inode  *inode;
	struct dir_entry * ent;
	int i, iret;
	uint32_t block_index, block_offset;

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
	pathname_len = strlen(pathname);

    mode = ecx;

	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -3;
	}
	inode = iget(inode_nr, &bh); 
	if (inode == NULL )
	{
        if (mode & F_OK)
        {
		    return 1; /*������*/
        }
	}
    /*TODO: ������ӿ�ִ�С��ɶ�����д�ļ��*/
    
	buffer_release(bh);
	return 0;
}
uint32_t sys_readdir(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/* int readdir(const char * pathname, uint32_t index, char * buf); */
	uint32_t addr;
	const char * pathname;
	uint32_t pathname_len;
	char * buf;

	uint32_t inode_nr;
	TBuffer * bh;
	struct m_inode  *inode;
	struct dir_entry * ent;
	int i, iret;
	uint32_t block_index, block_offset;

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
	pathname_len = strlen(pathname);

	//user_space_vaddr_to_paddr( (TProcess *)g_current, edx, &addr, ds);
    addr = edx;
	buf = (char *)addr;

	iret =  namei(pathname, pathname_len, &inode_nr);
	if (iret != 0)
	{
		return -3;
	}
	//printk("inode_nr=%d\n", inode_nr);
	inode = iget(inode_nr, &bh); 
	if (inode == NULL)
	{
		return -4;
	}
	//printk("type = %d \n", inode->i_type);
	if (inode->i_type != FILE_TYPE_DIR)
	{
		buffer_release(bh);
		return -5;	
	}
	//printk("i_size=%d\n", inode->i_size);
	if (ecx * sizeof(struct dir_entry) >= inode->i_size)
	{
		buffer_release(bh);
		return 0;
	}

	block_index = ecx /  DIR_ENTRY_NR_PER_BLOCK;
	block_offset = ecx % DIR_ENTRY_NR_PER_BLOCK ;
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
uint32_t sys_rmfile(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/*int rmdir(const char *pathname);*/
	uint32_t addr;
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

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
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
		panic("%s %d: BUG!", __FILE__, __LINE__);
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

uint32_t sys_open(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/* int open(const char * filename, uint32_t flag) */
	int fd, i_file;
	int iret;
	const char * pathname;
	int pathname_len;
	uint32_t inode_nr, addr;
	TBuffer * bh;
	struct m_inode * p_inode;

	for (fd = 0; fd < MAX_FD_NR; ++fd)
	{
		if (g_current->fd[fd] == 0)
		{
			break;	
		}
	}
	if (fd >= MAX_FD_NR) /*���̴��ļ�̫��*/
	{
		return -1;
	}
	for (i_file = 1; i_file < MAX_FILE_TABLE; ++i_file)
	{
		if (g_file_table[i_file].f_count == 0 )
		{
			break;	
		}
	}
	if (i_file >= MAX_FILE_TABLE) /* ϵͳ�򿪵��ļ�̫��*/
	{
		return -2;
	}
	memset(&(g_file_table[i_file]), 0, sizeof(g_file_table[i_file]));
	if (ecx & O_APPEND)  g_file_table[i_file].f_flags |= O_APPEND;
	g_file_table[i_file].f_count = 1;

	//user_space_vaddr_to_paddr( (TProcess *)g_current, ebx, &addr, ds);
    addr = ebx;
	pathname = (const char *)addr;
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
	p_inode->i_count++;
	bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(bh);

	g_file_table[i_file].f_inode = inode_nr;
	g_current->fd[fd] = i_file;
	return fd;
}
uint32_t sys_close(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/* int close(uint32_t fd) */
	int fd;
	int i_file;
	uint32_t inode;
	struct m_inode *p_inode;
	TBuffer * bh;

	fd = ebx;

	if (fd < 0 || fd >= MAX_FD_NR)
	{
		return -1;
	}
	i_file = g_current->fd[fd];
	if (i_file == 0 || i_file >= MAX_FILE_TABLE)
	{
		return -2;
	}
	if (g_file_table[i_file].f_count > 0)
	{
		--g_file_table[i_file].f_count;
	}
	if (g_file_table[i_file].f_count == 0)
	{
		inode = g_file_table[i_file].f_inode;
		memset(&(g_file_table[i_file]), 0, sizeof(g_file_table[i_file]));
	}
	else
	{
		return 0;
	}


	p_inode = iget(inode, &bh); 
	if (p_inode == NULL)
	{
		return -4;
	}
	if (p_inode->i_count > 0)
	{
		p_inode->i_count--;
	}

	bh->flags |= BUFFER_FLAG_DIRTY;
	buffer_release(bh);
	return 0;
}
int file_read(uint32_t inode_nr,  uint32_t offset, uint32_t len, unsigned char * buf)
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
			//printk("!!!i=%u, offset=%u\n", i, offset);
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
uint32_t   sys_read(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
	/*  ssize_t read(int fd, void *buf, size_t count); */
	uint32_t addr;
	char chr;
	char * buf;
	uint32_t inode_nr;
	int len;
	if (ecx == 0 || edx < 1 || ebx >= MAX_FD_NR)
	{
		return -__LINE__;
	}
	if (g_current->fd[ebx] == 0 || g_current->fd[ebx]  >= MAX_FILE_TABLE)
	{
		return -__LINE__;
	}
	inode_nr = g_file_table[ g_current->fd[ebx] ].f_inode;
	//user_space_vaddr_to_paddr( (TProcess *)g_current, ecx, &addr, ds);
    addr = ecx;
	buf = (char*)addr;
	len = file_read(inode_nr, g_file_table[ g_current->fd[ebx] ].f_pos, edx, buf);
	if (len < 0)
	{
		return len;
	}
	g_file_table[ g_current->fd[ebx] ].f_pos += len;
	return len;
}
uint32_t   sys_write(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
/* ssize_t write(int fd, const void *buf, size_t count); */
	uint32_t addr;
	char chr;
	char * buf;
	uint32_t inode_nr;
	int len;
	if (ecx == 0 || edx < 1 || ebx >= MAX_FD_NR)
	{
		return -__LINE__;
	}
	if (g_current->fd[ebx] == 0 || g_current->fd[ebx]  >= MAX_FILE_TABLE)
	{
		return -__LINE__;
	}
	inode_nr = g_file_table[ g_current->fd[ebx] ].f_inode;
	//user_space_vaddr_to_paddr( (TProcess *)g_current, ecx, &addr, ds);
    addr = ecx;
	buf = (char*)addr;
	len = file_write(inode_nr, g_file_table[ g_current->fd[ebx] ].f_pos, edx, buf, 
		g_file_table[ g_current->fd[ebx] ].f_flags);
	if (len < 0)
	{
		return len;
	}
	g_file_table[ g_current->fd[ebx] ].f_pos += len;
	return len;
}

uint32_t   sys_lseek(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds)
{
/* uint32_t lseek(uint32_t fildes, int32_t offset, uint32_t whence);*/
	uint32_t inode_nr;
	TBuffer *bh;
	struct m_inode * inode;
	int32_t offset;
	uint32_t filesz;
	uint32_t old_pos;
	offset = ecx;

	if (ebx >= MAX_FD_NR)
	{
		return -__LINE__;
	}
	if (g_current->fd[ebx] == 0 || g_current->fd[ebx]  >= MAX_FILE_TABLE)
	{
		return -__LINE__;
	}
	inode_nr = g_file_table[ g_current->fd[ebx] ].f_inode;
	inode = iget(inode_nr, &bh); 
	if (inode == NULL)
	{
		return -__LINE__;
	}
	filesz = inode->i_size;
	buffer_release(bh);
	bh = NULL;

	old_pos = g_file_table[ g_current->fd[ebx] ].f_pos;
	if (edx == SEEK_SET)
	{
	}
	else if (edx == SEEK_CUR)
	{
		offset = g_file_table[ g_current->fd[ebx] ].f_pos + offset;
	}
	else if (edx == SEEK_END)
	{
		offset = filesz + offset;
	}
	else
	{
		return -__LINE__;
	}
	if (offset < 0) g_file_table[ g_current->fd[ebx] ].f_pos = 0;
	else if (offset > filesz) g_file_table[ g_current->fd[ebx] ].f_pos = filesz;
	else g_file_table[ g_current->fd[ebx] ].f_pos  = offset;

	return old_pos;
}

