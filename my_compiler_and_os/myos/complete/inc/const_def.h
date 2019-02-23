#ifndef CONST_DEF_H_INCLUDED
#define CONST_DEF_H_INCLUDED

/////////////////////////////////////
// memory 
/////////////////////////////////////
#define PD_START			0		/* ҳĿ¼��ʼ�������ַ0�ĵط�*/
#define PT_END				20480	/* 1ҳ pd + 4ҳ pt ����5ҳ*/
#define PHYS_MEM_SZ			16777216	/* 16M�ڴ�*/
/**
 * �ں˴����ԭ��(org) 
 * PT_END ~ KERNEL_CODE_START֮��׼�������ں˶�ջ��
 */
#define KERNEL_ORG			51200	


#define MAX_GDT_ENT_NR		8			/* gdt entry max number */
#define MAX_IDT_ENT_NR		255			/* idt entry max number */
#define MAX_LDT_ENT_NR		2			/* ldt entry max number */
#define MAX_PROC_NR			4			/* user level process max number */

#ifndef NULL
#define NULL 	( (void*)0 )
#endif

///////////////////////////////////////////////////
// descriptor flags here
///////////////////////////////////////////////////
#define DA_32   0x4000  // 32 λ��
#define DA_DRW	0x92 	// ���ڵĿɶ�д���ݶ�����ֵ
#define DA_C	0x98 // ���ڵ�ִֻ�д��������ֵ
#define DA_LDT	0x82   // �ֲ��������������ֵ
#define DA_PAGE	0x8000   //�ν��޵�����Ϊ4K
#define DA_386IGate 0x8e   // 386 �ж�������ֵ
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
/* �ж����� */
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

/*ϵͳ����ʹ�õ��ж�*/
#define INT_VECTOR_SYSCALL		0x80

/**
 * ���д������ʱ�򣬻���ʹ�üĴ����Ĺ�����
 * �޸ļĴ�����ֵ�����ú�����caller�����������
 * �Ż�����ôcaller����Ϊ����ĳ������ǰ��һЩ
 * �Ĵ�����ֵ���䣬������Щ�Ĵ�������ĳ������ֵ,
 * ���ٶ��ڴ�Ĵ����Լӿ�ִ���ٶ�
 * ���Ի��д������ʱ��Ҫע�Ᵽ��\�ָ�
 * ͨ�üĴ�����ֵ��eax������Ϊ�������أ�����������\�ָ�
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





/* ѡ��������ֵ˵�� */
/* ����, SA_ : Selector Attribute */
#define SA_RPL_MASK 0xFFFC
#define SA_RPL0     0
#define SA_RPL1     1
#define SA_RPL2     2
#define SA_RPL3     3
#define SA_TI_MASK  0xFFFB
#define SA_TIG      0
#define SA_TIL      4

/* Ȩ�� */
#define PRIVILEGE_KRNL  0
#define PRIVILEGE_TASK  1
#define PRIVILEGE_USER  3
/* RPL */
#define RPL_KRNL    SA_RPL0
#define RPL_TASK    SA_RPL1
#define RPL_USER    SA_RPL3

/*�й�ʱ��*/
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

#define GS_START            0x0B8000    /*�Կ��ڴ濪ʼ�ط�, 184K */
#define GS_END              0x0C0000    /*�Կ��ڴ�����ط�, ��GS_START���32K */

/*ÿ��Լռ��4K�ڴ棬�Դ�ɴ洢8��*/
#define SCR_WIDTH           80          /* ��Ļ�Ŀ�Ⱥ͸߶�*/
#define SCR_HEIGHT          25

/*ϵͳ���ú�*/
#define SYSCALL_GET_TICKS_LO  	0
#define SYSCALL_GET_TICKS_HI	1
#define SYSCALL_SLEEP		2
#define SYSCALL_READ		3
#define SYSCALL_WRITE		4
#define SYSCALL_HD			5

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


/*����0��æ��־*/
#define TASK_BUSY_FLAG_KEYBOARD		0x01	/*�м���������Ҫ���*/

/*����״̬*/
#define PROC_STATUS_RUNNING			0x01	/*׼������ʹ��cpu*/
#define PROC_STATUS_SLEEPING		0x02	/*����˯��һ��ʱ��*/
#define PROC_STATUS_WAITING			0x03	/*�ȴ�IO���*/

#endif

