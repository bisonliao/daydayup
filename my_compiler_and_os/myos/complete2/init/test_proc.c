#include "const_def.h"

#include "api.h"
#include "global.h"


#define XXX (512*8*2)

void format()
{
	unsigned char buf[512];
	int i, offset, left;
	unsigned char mask;
	struct m_inode *inode;

	memset(buf, 0, sizeof(buf));
	_hd(0, buf, WIN_WRITE) ;	/*引导块*/
	_hd(1, buf, WIN_WRITE) ;	/*超级块*/

	/* 2,3用于inode位图 */
	memset(buf, 0, sizeof(buf));
	buf[0] = 0x1;	/*第一个inode用掉了*/
	_hd(2, buf, WIN_WRITE) ;
	buf[0] = 0;
	_hd(3, buf, WIN_WRITE) ;

	/*FIRST_SECT_NO_FOR_DATA以前的扇区都被使用掉了*/
	memset(buf, 0, sizeof(buf));
	for (i = 0; i < FIRST_SECT_NO_FOR_DATA; ++i)
	{
		offset = i / 8;	
		left = i % 8;
		mask = 0x01u << left;

		buf[offset] |= mask;
	}
	_hd(4, buf, WIN_WRITE) ;

	memset(buf, 0, sizeof(buf));
	for (i = 5; i < 5+NSECT_FOR_SECT_BMP; ++i)
	{
		_hd(i, buf, WIN_WRITE) ;
	}

	//写第一个inode
	memset(buf, 0, sizeof(buf));
	inode = (struct m_inode*)buf;
	inode->i_type = FILE_TYPE_DIR;
	_hd(2+NSECT_FOR_INODE_BMP+NSECT_FOR_SECT_BMP, buf, WIN_WRITE) ;
}
static int gets(char * buf, size_t sz)
{
	int i;
	int ret;
	for (i = 0; i < sz; ++i)
	{
		ret = _read(0, &buf[i], 1);
		if (ret < 0)
		{
			return -1;
		}
		if (buf[i] == '\n')
		{
			buf[i] = '\0';
			break;
		}
	}
	return 0;
}
/*进程体A*/
void proc_A()
{
	int fd;
	static unsigned char buf[1000];
	int len;
	int i;

	fd = _open("/data/perform2.txt", 0);
	if (fd < 0)
	{
		printk("_open failed!\n");
		goto A_end;
	}
	len = _read(fd, buf, sizeof(buf));
	if (len < 0)
	{
		printk("_read failed!\n");
		goto A_end;
	}
	for (i = 0; i < len; ++i)
	{
		if (buf[i] != ((i*i)&0xff))
		{
			printk("mismatch!\n");
			goto A_end;
		}
	}
	printk("OK!\n");
	_close(fd);

A_end:
	while (1)
	{
		_sleep(10);
	}
}
/*进程体B*/
void proc_B()
{
	int i = 0;
	while (1)
	{
		_sleep(1000000);
		printk("in B %d...\n", ++i);
	}
}
