/**
 * 制作硬盘文件系统，并在其中拷贝sh等必须程序的工具
 */
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "fdisk.h"

int g_scr_line;
int g_scr_colume;

void format()
{
	unsigned char buf[512];
	int i, offset, left;
	unsigned char mask;
	struct m_inode *inode;

	memset(buf, 0, sizeof(buf));
	hd_write_sync(0, buf) ;	/*引导块*/
	hd_write_sync(1, buf) ;	/*超级块*/

	/* 2,3用于inode位图 */
	memset(buf, 0, sizeof(buf));
	buf[0] = 0x1;	/*第一个inode用掉了*/
	hd_write_sync(2, buf);
	buf[0] = 0;
	hd_write_sync(3, buf);

	/*FIRST_SECT_NO_FOR_DATA以前的扇区都被使用掉了*/
	memset(buf, 0, sizeof(buf));
	for (i = 0; i < FIRST_SECT_NO_FOR_DATA; ++i)
	{
		offset = i / 8;	
		left = i % 8;
		mask = 0x01u << left;

		buf[offset] |= mask;
	}
	hd_write_sync(4, buf);

	memset(buf, 0, sizeof(buf));
	for (i = 5; i < 5+NSECT_FOR_SECT_BMP; ++i)
	{
		hd_write_sync(i, buf);
	}

	//写第一个inode
	memset(buf, 0, sizeof(buf));
	inode = (struct m_inode*)buf;
	inode->i_type = FILE_TYPE_DIR;
	hd_write_sync(2+NSECT_FOR_INODE_BMP+NSECT_FOR_SECT_BMP, buf);
}

int main()
{
	static unsigned char buf[1024*100];
	int len;
	int i;
	int fd;

	g_current = &g_proc;
#if 1
	format();
	fs_init();
	printf("mkdir return %d\n", sys_mkdir("/bin"));

	printf("mkfile return %d\n", sys_mkfile("/bin/sh"));
	fd = open("../../coreutil/sh.bin", O_RDONLY);
	if (fd < 0) {perror("open:"); return -1;}
	len = read(fd, buf, sizeof(buf));
	if (len < 0 || len >= sizeof(buf)) {perror("read:"); return -1;}
	close(fd);
	printf("write return %d\n", sys_write("/bin/sh", buf, 0, len));

	printf("mkfile return %d\n", sys_mkfile("/bin/sh2"));
	fd = open("../../coreutil/sh2.bin", O_RDONLY);
	if (fd < 0) {perror("open:"); return -1;}
	len = read(fd, buf, sizeof(buf));
	if (len < 0 || len >= sizeof(buf)) {perror("read:"); return -1;}
	close(fd);
	printf("write return %d\n", sys_write("/bin/sh2", buf, 0, len));

	buffer_sync();
#else
	fs_init();
	memset(buf, 0, sizeof(buf));

	len = sys_read("/bin/sh", buf, 0, sizeof(buf));
	printf("read return %d\n", len);
#endif
	return 0;
}
