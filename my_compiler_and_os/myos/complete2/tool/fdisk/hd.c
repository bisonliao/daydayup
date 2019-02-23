#include </usr/include/time.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "fdisk.h"

const char filename[] = "./c.img";

int hd_add_request( TBuffer* buf, uint32_t abs_sect, int cmd, int pid)
{
	int fd;
	fd = open(filename, O_RDWR);
	if (fd < 0)
	{
		perror("open");
		return -1;
	}
	if (cmd == WIN_READ)
	{
		lseek(fd, abs_sect * 512, SEEK_SET);
		if (read(fd, buf->data, 512) != 512)
		{
			perror("read");
			return -1;
		}
		buf->flags |= BUFFER_FLAG_UPTODATE;
	}
	else if (cmd == WIN_WRITE)
	{
		lseek(fd, abs_sect * 512, SEEK_SET);
		if (write(fd, buf->data, 512) != 512)
		{
			perror("write");
			return -1;
		}
		buf->flags |= BUFFER_FLAG_UPTODATE;
	}
	else 
	{
        fprintf(stderr, "invalid cmd:%d\n", cmd);
		return -2;
	}
	close(fd);
	return 0;
}
int hd_read_sync(uint32_t abs_sector, unsigned char * buf)
{
	int fd;
	fd = open(filename, O_RDWR);
	if (fd < 0)
	{
		perror("open");
		return -1;
	}
	lseek(fd, abs_sector * 512, SEEK_SET);
	if (read(fd, buf, 512) != 512)
	{
		perror("read");
		return -1;
	}
	close(fd);
	return 0;
}
int hd_write_sync(uint32_t abs_sector, unsigned char * buf)
{
	int fd;
	fd = open(filename, O_RDWR);
	if (fd < 0)
	{
		perror("open");
		return -1;
	}
	lseek(fd, abs_sector * 512, SEEK_SET);
	if (write(fd, buf, 512) != 512)
	{
		perror("write");
		return -1;
	}
	close(fd);
	return 0;
}

void panic2(const char * fmt, ...)
{
	va_list va;

	va_start(va, fmt);
	vprintf(fmt, va);
	va_end(va);

	exit(-1);
}
