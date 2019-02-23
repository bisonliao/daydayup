#ifndef _FDISK_H_INCLUDE_
#define _FDISK_H_INCLUDE_

#define WIN_READ        0x20 
#define WIN_WRITE       0x30
#define MAX_LDT_ENT_NR      2   

#define PROC_STATUS_RUNNING         0x01    /*׼������ʹ��cpu*/
#define PROC_STATUS_SLEEPING        0x02    /*����˯��һ��ʱ��*/
#define PROC_STATUS_WAITING         0x03    /*�ȴ�IO���*/


/*�ļ�������*/
#define NAME_LEN            12

/*�ļ�����*/
#define FILE_TYPE_REGULAR   0x1 /*һ���ļ�*/
#define FILE_TYPE_DIR       0x2 /*Ŀ¼*/

/* ���ٻ�������־*/
#define BUFFER_FLAG_DIRTY   (0x01u)
#define BUFFER_FLAG_UPTODATE  (0x02u)
#define BUFFER_FLAG_LOCKED  (0x04u)

/* inode��־*/
#define INODE_FLAG_DIRTY   (0x01u)
#define INODE_FLAG_UPTODATE  (0x02u)
#define INODE_FLAG_LOCKED  (0x04u)

/*�ļ��򿪱�־*/
#define O_APPEND        0x01

/*  
 * -----------------------------------------------------------------------------
 * |������ | ������| 2��inodeλͼ��| 32��λͼ��| 911��������inode | ���ݿ�...
 * -----------------------------------------------------------------------------
 */
#define NSECT_FOR_INODE_BMP     2   /*����inodeλͼ��������Ŀ*/
#define NSECT_FOR_SECT_BMP      32  /*��������λͼ��������Ŀ*/
#define MAX_INODE_NR            (NSECT_FOR_INODE_BMP*8*512)     /*����inode����*/
#define MAX_SECT_NR             (NSECT_FOR_SECT_BMP * 8 * 512)  /*������������*/
#define INODE_NR_PER_SECT       (512/sizeof(struct m_inode))     /*ÿ�����������inode����*/
#define DIR_ENTRY_NR_PER_BLOCK       (512/sizeof(struct dir_entry))     /*ÿ�����������dir entry����*/
#define NSECT_FOR_INODE         ( (MAX_INODE_NR +INODE_NR_PER_SECT-1)/ INODE_NR_PER_SECT)   /*���ڱ���inode�������ĸ���*/
#define FIRST_SECT_NO_FOR_DATA  (1+1+NSECT_FOR_INODE_BMP+NSECT_FOR_SECT_BMP+NSECT_FOR_INODE)
#define MAX_FILE_SIZE           (6*512+2*128*512+128*128*2*512)     /*����ļ���С,=16911360 bytes*/


#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define MAX_FILE_TABLE      100     /*�ļ�������*/
#define MAX_FD_NR           10      /*ÿ�����̴򿪵��ļ�������*/


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
	uint16_t	iobase;	/* I/Oλͼ��ַ���ڻ����TSS�ν��ޣ��ͱ�ʾû��I/O���λͼ */
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
	volatile TDescriptor         ldts[ MAX_LDT_ENT_NR ];     /* 2���͹���*/
	TSS					tss;
	volatile uint16_t            pid;            
	volatile uint64_t			alarm; /*��ʱʱ�䣬��ticks���ڵ��ڸ�ֵ�����Ѹý���*/
	volatile uint8_t				status;	/*����״̬*/
	volatile uint8_t				counter; /*ʱ��Ƭ��ת������*/
	volatile uint8_t				nice; /*���ȼ�*/
	volatile uint32_t			cwd_inode; /*��ǰ����Ŀ¼��inode*/
	volatile uint32_t			root_inode; /*��Ŀ¼��inode*/
	volatile uint16_t			fd[ MAX_FD_NR ]; /*���ļ��������file table�����±�*/
} TProcess; 	/*���̿��ƿ�*/


/*���ٻ�����*/
typedef struct
{
	volatile uint32_t    abs_sect;
	volatile uint8_t     flags;   /* BUFFER_FLAG_DIRTY, BUFFER_FLAG_UPDATE, BUFFER_FLAG_LOCKED etc. */
	volatile uint16_t    locker_owner_pid;  /*�������������¼ӵ�и����Ľ��̱��*/
	volatile TProcess*   wait;    /* waiting process list*/

	volatile uint8_t     data[512];
} TBuffer;

struct m_inode {
	volatile uint32_t i_size;   /*�ļ�/Ŀ¼��С*/
	volatile uint16_t i_entry_nr;   /*�����Ŀ¼����¼��ЧĿ¼��ĸ���*/
	volatile uint32_t i_ctime;  /*�ļ�״̬�޸�ʱ��*/
	volatile uint32_t i_mtime;  /*�ļ������޸�ʱ��*/
	volatile uint32_t i_atime;  /*����ʱ��*/ 
	volatile uint8_t  i_type;   /* �ļ�����Ŀ¼*/
	volatile uint8_t  i_count;  /* ���ô���,����ֻ��Ҫ�������ڴ��У����ˣ�һ���*/
	volatile uint8_t  i_link;   /* �ж��ٸ�Ŀ¼��ָ���inode*/
	volatile uint32_t i_zone[10];   /*����ָ��*/ /*0-5ֱ�ӿ� 6-7һ�μ�ӿ� 8-9���μ�ӿ� ����ļ�Լ16M*/
};  

/*Ŀ¼�� һ���ܹ�Ҫ������512*/
struct dir_entry {
	volatile uint32_t   inode;
	char                name[NAME_LEN];
};
/*file table��Ķ���*/
struct file {
	volatile uint8_t f_flags; /* O_APPEND O_CREAT�ȱ�־*/
	volatile uint8_t f_count;   /*���ü��� ���ӽ��̵��´���1, 0��ʾû��ʹ��*/
	volatile uint32_t f_inode; /*inode���*/
	volatile uint32_t f_pos;            /*��дƫ��*/
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
