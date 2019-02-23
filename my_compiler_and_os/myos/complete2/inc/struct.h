#ifndef STRUCT_H_INCLUDED
#define STRUCT_H_INCLUDED

#include "types.h"
#include "const_def.h"



#pragma pack(1)
typedef struct
{
	uint16_t   	hd_cyl;
	uint8_t	   	hd_head;
	uint16_t	hd_reserved1;
	uint16_t	hd_wpcom;
	uint8_t		hd_reserved2;
	uint8_t		hd_ctl;
	uint8_t		hd_reserved3;
	uint8_t		hd_reserved4;
	uint8_t		hd_reserved5;
	uint16_t	hd_lzone;
	uint8_t		hd_spt; /*sects nr per track */
	uint8_t		hd_reserved6;
} THdParam;
typedef struct 
{
	uint16_t 	dr_lower_limit;
	uint16_t 	dr_lower_base1;
	uint8_t		dr_lower_base2;
	uint16_t	dr_attributes;
	uint8_t		dr_higher_base;
} TDescriptor;
typedef struct 
{
	uint16_t	gt_offset_low;	/* Offset Low */
	uint16_t	gt_selector;	/* Selector */
	uint16_t	gt_attr;		/* attribute */
	uint16_t	gt_offset_high;	/* Offset High */
}	TGate;
typedef struct 
{
	uint16_t	gr_len;	
	uint32_t	gr_base;	
} TGdtr48;
typedef struct 
{
	uint16_t	it_len;	
	uint32_t	it_base;	
} TIdtr48;


typedef struct {   
	uint32_t    gs;     
	uint32_t    fs;     
	uint32_t    es;     
	uint32_t    ds;     
	uint32_t    edi;        
	uint32_t    esi;        
	uint32_t    ebp;        
	uint32_t	temp; /* no used ? */
	uint32_t    ebx;        
	uint32_t    edx;        
	uint32_t    ecx;        
	uint32_t    eax;        
	uint32_t    retaddr;    

	uint32_t    eip;        
	uint32_t    cs;     
	uint32_t    eflags;     
	uint32_t    esp;        
	uint32_t    ss;     
}TRegContext; 	/*进程上下文保存的regs*/

typedef struct {
	uint32_t	backlink; 	/* 0 */
	uint32_t	esp0;		
	uint32_t	ss0;		
	uint32_t	esp1;
	uint32_t	ss1;
	uint32_t	esp2;	/* 5 */
	uint32_t	ss2;
	uint32_t	cr3;
	uint32_t	eip;
	uint32_t	eflags;
	uint32_t	eax;	/* 10 */
	uint32_t	ecx;
	uint32_t	edx;
	uint32_t	ebx;
	uint32_t	esp;
	uint32_t	ebp;	/* 15 */
	uint32_t	esi;
	uint32_t	edi;
	uint32_t	es;
	uint32_t	cs;
	uint32_t	ss;		/* 20 */
	uint32_t	ds;
	uint32_t	fs;
	uint32_t	gs;
	uint32_t	ldt;	/* 24 */
	uint16_t	trap;
	uint16_t	iobase;	/* I/O位图基址大于或等于TSS段界限，就表示没有I/O许可位图 */
}TSS;
typedef struct {
    uint64_t  st0;
    uint64_t  st1;
    uint64_t  st2;
    uint64_t  st3;
    uint64_t  st4;
    uint64_t  st5;
    uint64_t  st6;
    uint64_t  st7;
    char fpu_env[28];
} fpu_status_struct;

typedef struct {
    volatile TSS					tss;
    volatile Selector            tss_sel;        
    volatile Selector            ldt_sel;        
    volatile TDescriptor         ldts[ MAX_LDT_ENT_NR ];     /* 2个就够了*/
    volatile uint16_t            pid;            
    volatile uint64_t			alarm; /*超时时间，当ticks大于等于该值，唤醒该进程*/
    volatile uint8_t				status;	/*进程状态*/
    volatile uint8_t				counter; /*时间片轮转计数器*/
    volatile uint8_t				nice; /*优先级*/
    volatile uint32_t			cwd_inode; /*当前工作目录的inode*/
    volatile uint32_t			root_inode; /*根目录的inode*/
    volatile uint16_t			fd[ MAX_FD_NR ]; /*打开文件句柄，是file table数组下标*/
    volatile uint16_t           ppid;       /*父进程*/
    volatile uint16_t           flag;       /*一些标志位，比如正在sleep等待子进程结束*/
    volatile int32_t            exit_code;  /*退出码*/
    volatile fpu_status_struct fpuss;
} TProcess; 	/*进程控制块*/


struct tm {
	uint32_t tm_sec;
	uint32_t tm_min;
	uint32_t tm_hour;
	uint32_t tm_mday;
	uint32_t tm_mon;
	uint32_t tm_year;
	uint32_t tm_wday;
	uint32_t tm_yday;
	uint32_t tm_isdst;
};      

/*高速缓冲区*/
typedef struct
{
	volatile uint32_t    abs_sect;
	volatile uint8_t     flags;   /* BUFFER_FLAG_DIRTY, BUFFER_FLAG_UPDATE, BUFFER_FLAG_LOCKED etc. */
	volatile uint16_t	 locker_owner_pid; 	/*如果被上锁，记录拥有该锁的进程编号*/
	volatile TProcess*   wait;    /* waiting process list*/

	volatile uint8_t     data[512];
} TBuffer;

struct m_inode {
	volatile uint32_t i_size;	/*文件/目录大小*/
	volatile uint16_t i_entry_nr;	/*如果是目录，记录有效目录项的个数*/
	volatile uint32_t i_ctime;	/*文件状态修改时间*/
	volatile uint32_t i_mtime;	/*文件内容修改时间*/
	volatile uint32_t i_atime;	/*访问时间*/
	volatile uint8_t  i_type;	/* 文件还是目录*/
	volatile uint8_t  i_count;	/* 引用次数,本来只需要存在在内存中，算了，一起吧*/
	volatile uint8_t  i_link;	/* 有多少个目录项指向该inode*/
	volatile uint32_t i_zone[10];	/*扇区指针*/ /*0-5直接块 6-7一次间接块 8-9两次间接块 最大文件约16M*/
};  


/*超级块*/
struct super_block {
};

/*目录项 一定能够要能整除512*/
struct dir_entry {
	volatile uint32_t 	inode;
	char 	 			name[NAME_LEN];
};
/*file table里的冬冬*/
struct file {
	volatile uint8_t f_flags; /* O_APPEND O_CREAT等标志*/
	volatile uint8_t f_count;	/*引用计数 父子进程导致大于1, 0表示没有使用*/
	volatile uint32_t f_inode; /*inode编号*/
	volatile uint32_t f_pos;			/*读写偏移*/
};

/*每个进程的paging的数据结构*/
/*
 * 一个进程内核态4M,用户态1M虚拟地址空间，所以用一个页目录表两个页表就搞定
 * 每个进程占用12K，这个数据结构放在内核里
 */
struct paging_t {
    uint32_t page_dir_ent[1024];  
    uint32_t page_tbl_ent[2048];
};


#pragma pack()

#endif
