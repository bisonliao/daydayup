#ifndef _FDISK_H_INCLUDE_
#define _FDISK_H_INCLUDE_

#define WIN_READ        0x20 
#define WIN_WRITE       0x30
#define MAX_LDT_ENT_NR      2   

#define PROC_STATUS_RUNNING         0x01    /*准备就绪使用cpu*/
#define PROC_STATUS_SLEEPING        0x02    /*正在睡眠一段时间*/
#define PROC_STATUS_WAITING         0x03    /*等待IO完成*/


/*文件名长度*/
#define NAME_LEN            12

/*文件类型*/
#define FILE_TYPE_REGULAR   0x1 /*一般文件*/
#define FILE_TYPE_DIR       0x2 /*目录*/

/* 高速缓冲区标志*/
#define BUFFER_FLAG_DIRTY   (0x01u)
#define BUFFER_FLAG_UPTODATE  (0x02u)
#define BUFFER_FLAG_LOCKED  (0x04u)

/* inode标志*/
#define INODE_FLAG_DIRTY   (0x01u)
#define INODE_FLAG_UPTODATE  (0x02u)
#define INODE_FLAG_LOCKED  (0x04u)

/*文件打开标志*/
#define O_APPEND        0x01

/*  
 * -----------------------------------------------------------------------------
 * |引导块 | 超级块| 2个inode位图块| 32个位图块| 911个块用于inode | 数据块...
 * -----------------------------------------------------------------------------
 */
#define NSECT_FOR_INODE_BMP     2   /*用于inode位图的扇区数目*/
#define NSECT_FOR_SECT_BMP      32  /*用于扇区位图的扇区数目*/
#define MAX_INODE_NR            (NSECT_FOR_INODE_BMP*8*512)     /*最大的inode个数*/
#define MAX_SECT_NR             (NSECT_FOR_SECT_BMP * 8 * 512)  /*最大的扇区个数*/
#define INODE_NR_PER_SECT       (512/sizeof(struct m_inode))     /*每个扇区保存的inode个数*/
#define DIR_ENTRY_NR_PER_BLOCK       (512/sizeof(struct dir_entry))     /*每个扇区保存的dir entry个数*/
#define NSECT_FOR_INODE         ( (MAX_INODE_NR +INODE_NR_PER_SECT-1)/ INODE_NR_PER_SECT)   /*用于保存inode的扇区的个数*/
#define FIRST_SECT_NO_FOR_DATA  (1+1+NSECT_FOR_INODE_BMP+NSECT_FOR_SECT_BMP+NSECT_FOR_INODE)
#define MAX_FILE_SIZE           (6*512+2*128*512+128*128*2*512)     /*最大文件大小,=16911360 bytes*/


#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define MAX_FILE_TABLE      100     /*文件表项数*/
#define MAX_FD_NR           10      /*每个进程打开的文件最大个数*/


typedef uint16_t                        Selector;

#pragma pack(1)

typedef struct {
	uint32_t	backlink;
	uint32_t	esp0;		/* stack pointer to use during interrupt */
	uint32_t	ss0;		/*   "   segment  "  "    "        "     */
	uint32_t	esp1;
	uint32_t	ss1;
	uint32_t	esp2;
	uint32_t	ss2;
	uint32_t	cr3;
	uint32_t	eip;
	uint32_t	eflags;
	uint32_t	eax;
	uint32_t	ecx;
	uint32_t	edx;
	uint32_t	ebx;
	uint32_t	esp;
	uint32_t	ebp;
	uint32_t	esi;
	uint32_t	edi;
	uint32_t	es;
	uint32_t	cs;
	uint32_t	ss;
	uint32_t	ds;
	uint32_t	fs;
	uint32_t	gs;
	uint32_t	ldt;
	uint16_t	trap;
	uint16_t	iobase;	/* I/O位图基址大于或等于TSS段界限，就表示没有I/O许可位图 */
}TSS;
typedef struct
{
	uint16_t    dr_lower_limit;
	uint16_t    dr_lower_base1;
	uint8_t     dr_lower_base2;
	uint16_t    dr_attributes;
	uint8_t     dr_higher_base;
} TDescriptor;

typedef struct {
	volatile Selector            tss_sel;        
	volatile Selector            ldt_sel;        
	volatile TDescriptor         ldts[ MAX_LDT_ENT_NR ];     /* 2个就够了*/
	TSS					tss;
	volatile uint16_t            pid;            
	volatile uint64_t			alarm; /*超时时间，当ticks大于等于该值，唤醒该进程*/
	volatile uint8_t				status;	/*进程状态*/
	volatile uint8_t				counter; /*时间片轮转计数器*/
	volatile uint8_t				nice; /*优先级*/
	volatile uint32_t			cwd_inode; /*当前工作目录的inode*/
	volatile uint32_t			root_inode; /*根目录的inode*/
	volatile uint16_t			fd[ MAX_FD_NR ]; /*打开文件句柄，是file table数组下标*/
} TProcess; 	/*进程控制块*/


/*高速缓冲区*/
typedef struct
{
	volatile uint32_t    abs_sect;
	volatile uint8_t     flags;   /* BUFFER_FLAG_DIRTY, BUFFER_FLAG_UPDATE, BUFFER_FLAG_LOCKED etc. */
	volatile uint16_t    locker_owner_pid;  /*如果被上锁，记录拥有该锁的进程编号*/
	volatile TProcess*   wait;    /* waiting process list*/

	volatile uint8_t     data[512];
} TBuffer;

struct m_inode {
	volatile uint32_t i_size;   /*文件/目录大小*/
	volatile uint16_t i_entry_nr;   /*如果是目录，记录有效目录项的个数*/
	volatile uint32_t i_ctime;  /*文件状态修改时间*/
	volatile uint32_t i_mtime;  /*文件内容修改时间*/
	volatile uint32_t i_atime;  /*访问时间*/ 
	volatile uint8_t  i_type;   /* 文件还是目录*/
	volatile uint8_t  i_count;  /* 引用次数,本来只需要存在在内存中，算了，一起吧*/
	volatile uint8_t  i_link;   /* 有多少个目录项指向该inode*/
	volatile uint32_t i_zone[10];   /*扇区指针*/ /*0-5直接块 6-7一次间接块 8-9两次间接块 最大文件约16M*/
};  

/*目录项 一定能够要能整除512*/
struct dir_entry {
	volatile uint32_t   inode;
	char                name[NAME_LEN];
};
/*file table里的冬冬*/
struct file {
	volatile uint8_t f_flags; /* O_APPEND O_CREAT等标志*/
	volatile uint8_t f_count;   /*引用计数 父子进程导致大于1, 0表示没有使用*/
	volatile uint32_t f_inode; /*inode编号*/
	volatile uint32_t f_pos;            /*读写偏移*/
};
#pragma pack()

int hd_add_request(TBuffer * buf, uint32_t abs_sect, int cmd, int pid);
void hd_init();
int hd_read_sync(uint32_t abs_sector, unsigned char * buf);
int hd_write_sync(uint32_t abs_sector, unsigned char * buf);

extern TProcess * g_current ;
extern TProcess g_proc;


#define _cli() {;}
#define sleep_on(a) {;}
#define schedule() {;}
#define wake_up(a) {;}
#define current_time() time(NULL)

void panic2(const char * fmt, ...);

#endif
