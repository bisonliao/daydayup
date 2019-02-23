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
}TRegContext; 	/*���������ı����regs*/

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
	uint16_t	iobase;	/* I/Oλͼ��ַ���ڻ����TSS�ν��ޣ��ͱ�ʾû��I/O���λͼ */
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
    volatile TDescriptor         ldts[ MAX_LDT_ENT_NR ];     /* 2���͹���*/
    volatile uint16_t            pid;            
    volatile uint64_t			alarm; /*��ʱʱ�䣬��ticks���ڵ��ڸ�ֵ�����Ѹý���*/
    volatile uint8_t				status;	/*����״̬*/
    volatile uint8_t				counter; /*ʱ��Ƭ��ת������*/
    volatile uint8_t				nice; /*���ȼ�*/
    volatile uint32_t			cwd_inode; /*��ǰ����Ŀ¼��inode*/
    volatile uint32_t			root_inode; /*��Ŀ¼��inode*/
    volatile uint16_t			fd[ MAX_FD_NR ]; /*���ļ��������file table�����±�*/
    volatile uint16_t           ppid;       /*������*/
    volatile uint16_t           flag;       /*һЩ��־λ����������sleep�ȴ��ӽ��̽���*/
    volatile int32_t            exit_code;  /*�˳���*/
    volatile fpu_status_struct fpuss;
} TProcess; 	/*���̿��ƿ�*/


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

/*���ٻ�����*/
typedef struct
{
	volatile uint32_t    abs_sect;
	volatile uint8_t     flags;   /* BUFFER_FLAG_DIRTY, BUFFER_FLAG_UPDATE, BUFFER_FLAG_LOCKED etc. */
	volatile uint16_t	 locker_owner_pid; 	/*�������������¼ӵ�и����Ľ��̱��*/
	volatile TProcess*   wait;    /* waiting process list*/

	volatile uint8_t     data[512];
} TBuffer;

struct m_inode {
	volatile uint32_t i_size;	/*�ļ�/Ŀ¼��С*/
	volatile uint16_t i_entry_nr;	/*�����Ŀ¼����¼��ЧĿ¼��ĸ���*/
	volatile uint32_t i_ctime;	/*�ļ�״̬�޸�ʱ��*/
	volatile uint32_t i_mtime;	/*�ļ������޸�ʱ��*/
	volatile uint32_t i_atime;	/*����ʱ��*/
	volatile uint8_t  i_type;	/* �ļ�����Ŀ¼*/
	volatile uint8_t  i_count;	/* ���ô���,����ֻ��Ҫ�������ڴ��У����ˣ�һ���*/
	volatile uint8_t  i_link;	/* �ж��ٸ�Ŀ¼��ָ���inode*/
	volatile uint32_t i_zone[10];	/*����ָ��*/ /*0-5ֱ�ӿ� 6-7һ�μ�ӿ� 8-9���μ�ӿ� ����ļ�Լ16M*/
};  


/*������*/
struct super_block {
};

/*Ŀ¼�� һ���ܹ�Ҫ������512*/
struct dir_entry {
	volatile uint32_t 	inode;
	char 	 			name[NAME_LEN];
};
/*file table��Ķ���*/
struct file {
	volatile uint8_t f_flags; /* O_APPEND O_CREAT�ȱ�־*/
	volatile uint8_t f_count;	/*���ü��� ���ӽ��̵��´���1, 0��ʾû��ʹ��*/
	volatile uint32_t f_inode; /*inode���*/
	volatile uint32_t f_pos;			/*��дƫ��*/
};

/*ÿ�����̵�paging�����ݽṹ*/
/*
 * һ�������ں�̬4M,�û�̬1M�����ַ�ռ䣬������һ��ҳĿ¼������ҳ��͸㶨
 * ÿ������ռ��12K��������ݽṹ�����ں���
 */
struct paging_t {
    uint32_t page_dir_ent[1024];  
    uint32_t page_tbl_ent[2048];
};


#pragma pack()

#endif
