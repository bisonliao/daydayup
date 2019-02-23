#ifndef CONST_DEF_H_INCLUDED
#define CONST_DEF_H_INCLUDED

/////////////////////////////////////
// memory 
/////////////////////////////////////
#define PD_START			0		/* 页目录表开始在物理地址0的地方*/
#define PT_END				20480	/* 1页 pd + 4页 pt 共计5页*/
#define PHYS_MEM_SZ			16777216	/* 16M内存*/

/**
 * 内核代码的原点(org) 
 * PT_END ~ KERNEL_CODE_START之间准备用于内核堆栈区
 */
#define KERNEL_ORG			51200	/* 1M以下用于kernel*/

#define BUFFER_ORG			1048576 /* 1M-2M用于buffer*/
#define BUFFER_END			(1048576*2-1) 

#define PROC_PAGING_ORG          (1048576*2)  /* 2M-3M的地方用于paging数据结构*/
#define PROC_PAGING_END          (1048576*3-1)

#define FIRST_PROC_ORG		(1048576*3) /*第一个进程空间开始的地方, 要保证是4096的整数倍*/
#define PROC_SPACE			 (4096*256) /*每个进程1.0M空间，要保证是4096的整数倍.这个值很诡异，1024*1024就不行*/

#define USER_SPACE_VADDR_HEAD      (1048576*4)  /*用户态的虚拟地址起始*/
#define USER_SPACE_VADDR_BOTTOM      (1048576*4+PROC_SPACE-1)  /*用户态的虚拟地址结束*/


#define MAX_GDT_ENT_NR		32			/* gdt entry max number */
#define MAX_IDT_ENT_NR		255			/* idt entry max number */
#define MAX_LDT_ENT_NR		2			/* ldt entry max number */
#define MAX_PROC_NR			8			/* user level process max number */

#define g_kernel_code_selector 8
#define g_kernel_data_selector  16
#define g_kernel_stack_selector  16
#define g_kernel_gs_selector  24
#define g_tmp_tss_selector	32
#define g_first_ldt_selector	40 /*第一个ldt选择子，选择子就是在GDT表里的一个偏移*/
#define g_first_tss_selector	(g_first_ldt_selector+8*MAX_PROC_NR)  /*第一个tss选择子，紧跟在MAX_PROC_NR个ldt选择子后面*/

#ifndef NULL
#define NULL 	( (void*)0 )
#endif

///////////////////////////////////////////////////
// descriptor flags here
///////////////////////////////////////////////////
#define DA_32   0x4000  // 32 位段
#define DA_DRW	0x92 	// 存在的可读写数据段属性值
#define DA_C	0x98 // 存在的只执行代码段属性值
#define DA_LDT	0x82   // 局部描述符表段类型值
#define DA_PAGE	0x8000   //段界限的粒度为4K
#define DA_386IGate 0x8e   // 386 中断门类型值
#define	DA_386TSS		0x89

///////////////////////////////////////////////////
// page entry flags here
///////////////////////////////////////////////////
#define	PG_P    0x01
#define PG_RWW  0x02
#define PG_USU  0x04

/* 8259A interrupt controller ports. */
#define	INT_M_CTL	0x20	/* I/O port for interrupt controller         <Master> */
#define	INT_M_CTLMASK	0x21	/* setting bits in this port disables ints   <Master> */
#define	INT_S_CTL	0xA0	/* I/O port for second interrupt controller  <Slave>  */
#define	INT_S_CTLMASK	0xA1	/* setting bits in this port disables ints   <Slave>  */
/* 中断向量 */
#define	INT_VECTOR_DIVIDE		0x0
#define	INT_VECTOR_DEBUG		0x1
#define	INT_VECTOR_NMI			0x2
#define	INT_VECTOR_BREAKPOINT		0x3
#define	INT_VECTOR_OVERFLOW		0x4
#define	INT_VECTOR_BOUNDS		0x5
#define	INT_VECTOR_INVAL_OP		0x6
#define	INT_VECTOR_COPROC_NOT		0x7
#define	INT_VECTOR_DOUBLE_FAULT		0x8
#define	INT_VECTOR_COPROC_SEG		0x9
#define	INT_VECTOR_INVAL_TSS		0xA
#define	INT_VECTOR_SEG_NOT		0xB
#define	INT_VECTOR_STACK_FAULT		0xC
#define	INT_VECTOR_PROTECTION		0xD
#define	INT_VECTOR_PAGE_FAULT		0xE
#define	INT_VECTOR_COPROC_ERR		0x10
#define	INT_VECTOR_IRQ0			0x20
#define	INT_VECTOR_IRQ8			0x28

/*系统调用使用的中断*/
#define INT_VECTOR_SYSCALL		0x80

/**
 * 汇编写函数的时候，会在使用寄存器的过程中
 * 修改寄存器的值，而该函数的caller如果被编译器
 * 优化，那么caller会认为调用某个函数前后一些
 * 寄存器的值不变，而用这些寄存器保存某个变量值,
 * 减少读内存的次数以加快执行速度
 * 所以汇编写函数的时候要注意保存\恢复
 * 通用寄存器的值。eax经常作为函数返回，不能做保存\恢复
 */
#define SAVE_REG    \
	pushl %ebx;\
	pushl %ecx;\
	pushl %edx;\
	pushl %esi;\
	pushl %edi;

#define RESTORE_REG \
	popl %edi;\
	popl %esi;\
	popl %edx;\
	popl %ecx;\
	popl %ebx;





/* 选择子类型值说明 */
/* 其中, SA_ : Selector Attribute */
#define SA_RPL_MASK 0xFFFC
#define SA_RPL0     0
#define SA_RPL1     1
#define SA_RPL2     2
#define SA_RPL3     3
#define SA_TI_MASK  0xFFFB
#define SA_TIG      0
#define SA_TIL      4

/* 权限 */
#define PRIVILEGE_KRNL  0
#define PRIVILEGE_TASK  1
#define PRIVILEGE_USER  3
/* RPL */
#define RPL_KRNL    SA_RPL0
#define RPL_TASK    SA_RPL1
#define RPL_USER    SA_RPL3

/*有关时钟*/
#define TIMER0	0x40
#define TIMER_MODE 0x43
#define RATE_GENERATOR	0x34
#define HZ		1000
#define TIMER_FREQ	1193182



/* VGA */
#define CRTC_ADDR_REG                   0x3D4   /* CRT Controller Registers - Address Register */
#define CRTC_DATA_REG                   0x3D5   /* CRT Controller Registers - Data Registers */
#define CRTC_DATA_IDX_START_ADDR_H      0xC     /* register index of video mem start address (MSB) */
#define CRTC_DATA_IDX_START_ADDR_L      0xD     /* register index of video mem start address (LSB) */
#define CRTC_DATA_IDX_CURSOR_H          0xE     /* register index of cursor position (MSB) */
#define CRTC_DATA_IDX_CURSOR_L          0xF     /* register index of cursor position (LSB) */

#define KEYBOARD_DATA_PORT  0x60
#define KEYBOARD_CMD_PORT  0x64

#define GS_START            0x0B8000    /*显卡内存开始地方, 184K */
#define GS_END              0x0C0000    /*显卡内存结束地方, 与GS_START相差32K */

/*每屏约占用4K内存，显存可存储8屏*/
#define SCR_WIDTH           80          /* 屏幕的宽度和高度*/
#define SCR_HEIGHT          25

/*系统调用号*/
#define SYSCALL_GET_TICKS_LO  	0
#define SYSCALL_GET_TICKS_HI	1
#define SYSCALL_SLEEP		2
#define SYSCALL_READ		3
#define SYSCALL_WRITE		4
#define SYSCALL_HD			5
#define SYSCALL_TIME		6
#define SYSCALL_SYNC		7
#define SYSCALL_TEST		8
#define SYSCALL_MKDIR		9
#define SYSCALL_RMDIR		10
#define SYSCALL_READDIR		11
#define SYSCALL_OPEN		12
#define SYSCALL_CLOSE		13
#define SYSCALL_MKFILE		14
#define SYSCALL_RMFILE		15
#define SYSCALL_LSEEK		16
#define SYSCALL_COUT		17
#define SYSCALL_FORK		18
#define SYSCALL_EXEC		19
#define SYSCALL_EXIT		20
#define SYSCALL_CIN		    21
#define SYSCALL_ACCESS	    22
#define SYSCALL_WAIT	    23

/* Hd controller regs. Ref: IBM AT Bios-listing */
#define HD_DATA     0x1f0   /* _CTL when writing */
#define HD_ERROR    0x1f1   /* see err-bits */
#define HD_NSECTOR  0x1f2   /* nr of sectors to read/write */
#define HD_SECTOR   0x1f3   /* starting sector */
#define HD_LCYL     0x1f4   /* starting cylinder */
#define HD_HCYL     0x1f5   /* high byte of starting cyl */
#define HD_CURRENT  0x1f6   /* 101dhhhh , d=drive, hhhh=head */
#define HD_STATUS   0x1f7   /* see status-bits */
#define HD_PRECOMP HD_ERROR /* same io address, read=error, write=precomp */
#define HD_COMMAND HD_STATUS    /* same io address, read=status, write=cmd */

#define HD_CMD      0x3f6

/* Bits of HD_STATUS */
#define ERR_STAT    0x01
#define INDEX_STAT  0x02
#define ECC_STAT    0x04    /* Corrected error */
#define DRQ_STAT    0x08
#define SEEK_STAT   0x10
#define WRERR_STAT  0x20
#define READY_STAT  0x40
#define BUSY_STAT   0x80

/* Values for HD_COMMAND */
#define WIN_RESTORE     0x10
#define WIN_READ        0x20 
#define WIN_WRITE       0x30
#define WIN_VERIFY      0x40
#define WIN_FORMAT      0x50
#define WIN_INIT        0x60
#define WIN_SEEK        0x70
#define WIN_DIAGNOSE        0x90
#define WIN_SPECIFY     0x91

/* Bits for HD_ERROR */
#define MARK_ERR    0x01    /* Bad address mark ? */
#define TRK0_ERR    0x02    /* couldn't find track 0 */
#define ABRT_ERR    0x04    /* ? */
#define ID_ERR      0x10    /* ? */
#define ECC_ERR     0x40    /* ? */
#define BBD_ERR     0x80    /* ? */



/*进程状态*/
#define PROC_STATUS_RUNNING			0x01	/*准备就绪使用cpu*/
#define PROC_STATUS_SLEEPING		0x02	/*正在睡眠一段时间*/
#define PROC_STATUS_WAITING			0x03	/*等待IO完成*/
#define PROC_STATUS_ZOMBIE			0x04	/*僵尸进程*/

#define KRNL_STACK_SZ		2048	/*每个进程的内核堆栈大小*/
#define MAX_FILE_TABLE 		100		/*文件表项数*/
#define MAX_FD_NR			10		/*每个进程打开的文件最大个数*/

/*文件名长度*/
#define NAME_LEN			12

/*文件类型*/
#define FILE_TYPE_REGULAR	0x1	/*一般文件*/
#define FILE_TYPE_DIR		0x2	/*目录*/

/* 高速缓冲区标志*/
#define BUFFER_FLAG_DIRTY   (0x01u)
#define BUFFER_FLAG_UPTODATE  (0x02u)
#define BUFFER_FLAG_LOCKED  (0x04u)

/* inode标志*/
#define INODE_FLAG_DIRTY   (0x01u)
#define INODE_FLAG_UPTODATE  (0x02u)
#define INODE_FLAG_LOCKED  (0x04u)

/*文件打开标志*/
#define O_APPEND		0x01

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
#define MAX_FILE_SIZE			(6*512+2*128*512+128*128*2*512) 	/*最大文件大小,=16911360 bytes*/


#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2


#define R_OK 1
#define W_OK 2
#define X_OK 4
#define F_OK 8


#define PROC_FLAG_WAIT_CHILD    0x1u



#define offset(T, m)  ( (unsigned int)(&(((T*)0)->m)))

#endif

