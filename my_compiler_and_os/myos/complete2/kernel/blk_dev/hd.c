#include "global.h"
#include "struct.h"
#include "cycle_buf.h"

typedef struct
{
	int cmd; 	/*读/写*/
	unsigned int abs_sect;	/*操作的绝对扇区编号*/
	TBuffer * buf;	/*缓冲区地址*/
	int pid;
} THdRequest;


static CycleBufHead * pstHead  __attribute__ ((section ("data"))) = NULL;
static unsigned char request_buf[sizeof(CycleBufHead)+sizeof(THdRequest)*20];
static THdRequest current_hd_request;
static volatile uint8_t   suspend_cmd_nr  __attribute__ ((section ("data")))  = 0;	/*已发送到驱动器但是驱动器还未通过中断返回的命令的个数*/

static void read_intr();
static void write_intr();
static int hd_read(uint32_t abs_sector);
static int hd_write(uint32_t abs_sector, const unsigned char * buf);

/**
 *  abs_sector表示绝对扇区号，cyl表示柱面号，head表示磁头号，sector表示扇区号，
 *  nr_head表示磁头数，nr_spt表示每磁道扇区数
 */
static int hd_abs_sect_to_phys_sect(uint32_t abs_sector,
		uint8_t nr_spt,
		uint8_t nr_head,
		uint16_t * cyl,
		uint8_t * head,
		uint8_t * sector)
{
	uint32_t track;
	if (NULL == cyl || NULL == head || NULL == sector)
	{
		return -1;
	}

	*sector =  abs_sector % nr_spt + 1;
	track	=  abs_sector / nr_spt;
	*head	=  track % nr_head;
	*cyl	=  track / nr_head;
	return 0;
}
static int hd_controller_ready(void)
{
	int retries=10000;

	while (--retries && (in_byte(HD_STATUS)&0xc0)!=0x40);

	return (retries);
}
static void hd_out(unsigned int nsect,
		unsigned int sect,
		unsigned int head,
		unsigned int cyl,
		unsigned int cmd)
{
	int port;
	if (!hd_controller_ready())
	{
		panic("HD controller not ready");
	}
	out_byte(HD_CMD, g_hd_param.hd_ctl);
	port=HD_DATA;
	out_byte(++port, g_hd_param.hd_wpcom>>2);
	out_byte(++port, nsect);
	out_byte(++port, sect);
	out_byte(++port, cyl);
	out_byte(++port, cyl>>8);
	out_byte(++port, 0xA0|head);
	out_byte(++port, cmd);
}

static int hd_drive_busy(void)
{
	unsigned int i;

	for (i = 0; i < 10; i++)
	{
		if (READY_STAT == (in_byte(HD_STATUS) & (BUSY_STAT|READY_STAT)))
			break;
	}
	i = in_byte(HD_STATUS);
	i &= BUSY_STAT | READY_STAT | SEEK_STAT;
	if (i == (READY_STAT | SEEK_STAT) )
	{
		return(0);
	}
	return(1);
}
static void hd_reset_controller(void)
{
	int i;

	out_byte(HD_CMD, 4);
	for(i = 0; i < 100; i++) 
		_nop();
	out_byte(HD_CMD, g_hd_param.hd_ctl  );
	if (hd_drive_busy())
		printk("HD-controller still busy\n");
	if ((i = in_byte(HD_ERROR)) != 1)
		printk("HD-controller reset failed: %02x\n",i);
}
static void hd_reset_hd()
{
	hd_reset_controller();
	hd_out(1,1,0, 0,WIN_SPECIFY);
}
static int hd_win_result(void)
{
	int i=in_byte(HD_STATUS);

	if ((i & (BUSY_STAT | READY_STAT | WRERR_STAT | SEEK_STAT | ERR_STAT))
			== (READY_STAT | SEEK_STAT))
		return(0); /* ok */
	if (i&1) i=in_byte(HD_ERROR);
	return (1);
}

static void hd_do_task()
{
	char cEmpty;

	if (suspend_cmd_nr) /*有未决命令*/
	{
		return;
	}

	if (cycle_buf_IsEmpty(&cEmpty, pstHead))
	{
		panic("cycle_buf_IsEmpty() failed!\n");
	}
	if (cEmpty) /*没有需要执行的请求*/
	{
		return;
	}


	if (cycle_buf_pop(pstHead,  &current_hd_request) != 0)
	{
		panic("cycle_buf_pop() failed!\n");
	}
	if (current_hd_request.cmd == WIN_READ)
	{
		//printk("%s %d\n", __FILE__, __LINE__);
		do_hd = &read_intr;
		hd_read(current_hd_request.abs_sect);
		++suspend_cmd_nr;
	}
	else if (current_hd_request.cmd == WIN_WRITE)
	{
		/*
		printk("wo kao              \n");
		printk("[cmd %d, buf 0x%x, pid %d, abs_sect %d]\n", current_hd_request.cmd, current_hd_request.buf, 
			current_hd_request.pid, current_hd_request.abs_sect);
			*/
		do_hd = &write_intr;
		hd_write(current_hd_request.abs_sect, (const unsigned char * )(current_hd_request.buf->data));
		++suspend_cmd_nr;
	}
	else
	{
		panic("invalid hd cmd value!");
	}
}
int hd_add_request(TBuffer* buf, uint32_t abs_sect, int cmd, int pid)
{
	THdRequest req;
    int iret;

	if (cmd != WIN_READ && cmd != WIN_WRITE)
	{
		return -1;
	}
	if (buf == NULL)
	{
		return -2;
	}
	req.cmd = cmd;
	req.buf = buf;
	req.pid = pid;
	req.abs_sect = abs_sect;

	//printk("[cmd %d, buf 0x%x, pid %d, abs_sect %d]\n", req.cmd, req.buf, req.pid, req.abs_sect);

	if ( (iret = cycle_buf_push(pstHead, &req)) != 0)
	{
//        printk("pstHead=%u\n", pstHead);
		return -1000 + iret;
	}
	
	hd_do_task();
	return 0;
}


static void read_intr()
{
	int i;
	unsigned char * p = NULL;
	int pid;
	
	//printk("read intr...\n");

	--suspend_cmd_nr;
	if (hd_win_result())
	{
		panic("hd read failed!\n");
	}

	for (i = 0, p = (unsigned char*)current_hd_request.buf->data; 
			i < 256; ++i, p += 2)
	{
		*(uint16_t*)p = in_word(HD_DATA);
	}
	current_hd_request.buf->flags = current_hd_request.buf->flags | BUFFER_FLAG_UPTODATE;
	pid = current_hd_request.pid;
	g_procs[pid].status = PROC_STATUS_RUNNING;


	hd_do_task(); /*继续下一个请求*/
}
static void write_intr()
{
	int pid;

	//printk("write intr...\n");
	--suspend_cmd_nr;
	if (hd_win_result())
	{
		panic("hd write failed!\n");
	}

	pid = current_hd_request.pid;
	g_procs[pid].status = PROC_STATUS_RUNNING;
	current_hd_request.buf->flags = current_hd_request.buf->flags & (~BUFFER_FLAG_DIRTY); /*清 dirty标志*/
	current_hd_request.buf->flags = current_hd_request.buf->flags | BUFFER_FLAG_UPTODATE;

	hd_do_task(); /*继续下一个请求*/

}

static int hd_read(uint32_t abs_sector)
{

	uint16_t cyl; uint8_t head; uint8_t sector;
	if (hd_abs_sect_to_phys_sect(abs_sector,
				g_hd_param.hd_spt,
				g_hd_param.hd_head,
				&cyl,
				&head,
				&sector) != 0)
	{
		panic("hd_abs_sect_to_phys_sect() failed!\n");
	}

	out_byte(HD_CMD, g_hd_param.hd_ctl);

	out_byte(HD_NSECTOR, 1);
	out_byte(HD_SECTOR, sector);
	out_byte(HD_LCYL, cyl&0x0f);
	out_byte(HD_HCYL, (cyl >> 8) & 0x3);
	out_byte(HD_CURRENT, (head&0x0f)|0xa0);
	out_byte(HD_STATUS, WIN_READ);

	return 0;
}
/*
 * 同步方式读取某块磁盘扇区，用于系统刚刚启动的时候读取文件系统信息
 */
int hd_read_sync(uint32_t abs_sector, unsigned char * buf)
{
	int i;
	int ready = 0;
	unsigned char * p = NULL;
	uint8_t status;

	uint16_t cyl; uint8_t head; uint8_t sector;
	if (hd_abs_sect_to_phys_sect(abs_sector,
				g_hd_param.hd_spt,
				g_hd_param.hd_head,
				&cyl,
				&head,
				&sector) != 0)
	{
		panic("hd_abs_sect_to_phys_sect() failed!\n");
	}

	out_byte(HD_CMD, g_hd_param.hd_ctl);

	out_byte(HD_NSECTOR, 1);
	out_byte(HD_SECTOR, sector);
	out_byte(HD_LCYL, cyl&0x0f);
	out_byte(HD_HCYL, (cyl >> 8) & 0x3);
	out_byte(HD_CURRENT, (head&0x0f)|0xa0);
	out_byte(HD_STATUS, WIN_READ);

	for (i = 0; i < 10000; ++i)
	{
		status = in_byte(HD_STATUS) ;
		//printk("in hd_read, status=0x%x\n", status);
		if (status & 0x08)
		{
			ready = 1;
			break;
		}
	}
	if (!ready)
	{
		return in_byte(HD_STATUS);
	}
	for (i = 0, p = buf; i < 256; ++i, p += 2)
	{
		*(uint16_t*)p = in_word(HD_DATA);
	}
	return 0;
}

static int hd_write(uint32_t abs_sector, const unsigned char * buf)
{	
	int i;
	int ready = 0;
	const unsigned char * p = NULL;
	uint8_t status;

	uint16_t cyl; uint8_t head; uint8_t sector;
	if (hd_abs_sect_to_phys_sect(abs_sector,
				g_hd_param.hd_spt,
				g_hd_param.hd_head,
				&cyl,
				&head,
				&sector) != 0)
	{
		panic("hd_abs_sect_to_phys_sect() failed!\n");
	}

	out_byte(HD_CMD, g_hd_param.hd_ctl);

	out_byte(HD_NSECTOR, 1);
	out_byte(HD_SECTOR, sector);
	out_byte(HD_LCYL, cyl&0x0f);
	out_byte(HD_HCYL, (cyl >> 8) & 0x3);
	out_byte(HD_CURRENT, (head&0x0f)|0xa0);
	out_byte(HD_STATUS, WIN_WRITE);


	for (i = 0; i < 10000; ++i)
	{
		status = in_byte(HD_STATUS) ;
		//printk("in hd_write, status=0x%x\n", status);
		if (status & 0x08)
		{
			ready = 1;
			break;
		}
	}
	if (!ready)
	{
		return status;
	}
	for (i = 0, p = buf; i < 256; ++i, p += 2)
	{
		out_word(HD_DATA, *(uint16_t*)p );
	}
	return 0;
}
void hd_init()
{
	uint8_t mask;
	
	//suspend_cmd_nr = 0;

	mask = in_byte(INT_M_CTLMASK);
	mask = mask & 0xfb;
	out_byte(INT_M_CTLMASK, mask);

	mask = in_byte(INT_S_CTLMASK);
	mask = mask & 0xbf;
	out_byte(INT_S_CTLMASK, mask);

	if (cycle_buf_MemAttach(request_buf, sizeof(request_buf), sizeof(THdRequest),
				&pstHead, 1))
	{
		panic("cycle_buf_MemAttach() failed!\n");
	}
    //printk("pstHead=%p\n", pstHead);
}
