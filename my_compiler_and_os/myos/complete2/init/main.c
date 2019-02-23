#include "const_def.h"

#include "global.h"
#include "time.h"
#include "api.h"
#include "fs.h"
#include "redefine.h"



static void init_intr_table();
/*����ʱ���жϵ�Ƶ��*/
static void init_8253();
static void create_proc_0();
static void init_8259A();
int create_first_user_proc(const char * pathname);


void c_start()
{
	unsigned int i;

	/*����Ļ*/
	for (i = 0; i < 25; ++i)
	{
		printk("                                                     \n");
	}

	_setup_paging(); /*���÷�ҳ*/

	init_8259A();   /*��ʼ���жϿ�����*/
	init_intr_table(); /*��ʼ���ж�������*/
	_lidt();    /*�����ж�������*/
	init_8253();    /*��ʼ��ʱ�ӿ�����*/
	time_init();    
	keyboard_init();
	/* 640K ~ 1M�ĵط��� bois/�Դ��õ�ROM*/
	fs_init();


	/*ͬ���ķ�ʽ��ȡ/bin/sh�ļ������ݣ���ʼ����һ���û�̬����*/
	create_first_user_proc("/bin/sh");

	hd_init();

	g_hd_sync_flag = 0; /*���ͬ��IO��־*/

    /*����������ʼ��*/
    /*
    if (LoadDriver() != 1)
    {
        panic("nic init failed!\n");
    }
    */
	/*�����ں�̬����0�����л����ý���*/
	create_proc_0();

}
/* 
 * ԭ��ͬcreate_proc_0��
 * ֻ��Ҫ�Ӵ����аѽ��̶����ڴ�
 */
int create_first_user_proc(const char * pathname)
{
	int iret;
	int pid;
    int argc;
    uint32_t stack_bottom;
    char * * argv;

	TProcess * p_proc = NULL;
    uint32_t user_space_org;


    pid = 3;
    user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;    /*���̿ռ�������ڴ濪ʼλ��*/
    /*����һ��λ��ջ�׶˵�argc/argv*/
    /*�����������������ƣ�Ҫͬʱ���os�������ַ�ͽ��̵������ַ*/
    char *pCur = NULL;
    {
        int arglen = strlen(pathname)+1;
        argc = 1;
        pCur = (char*)(user_space_org + PROC_SPACE - 1 - arglen - sizeof(char*)*(argc+1)); /*�����ַ*/
        /*ͨ��ֱ�ӷ��������ַ��д�����, ��Ϊ�ں˵�ҳ�����Ե�ֱַ��ӳ�䵽�����ַ*/
        argv = (char **)(pCur);
        pCur += sizeof(char*)*(argc+1);
        {
            memcpy(pCur, pathname, arglen); /*��ջ�Ϸźõ�һ��������pathname*/
            argv[0] = (pCur - user_space_org)+USER_SPACE_VADDR_HEAD;  /*����Ҫд�����ַ����Ϊ���������Լ�����*/
        }
        pCur -= sizeof(char*)*(argc+1);
        argv[1] = NULL; /*0����0�������ַ�������ַһֱ*/
    }
    /*���� argv�����Ѿ���֯���ˣ���ʼѹջ*/
    pCur -= 4;
    *(int *)pCur = (uint32_t)argv - user_space_org+USER_SPACE_VADDR_HEAD; /* char **argv ��һ�������ַ*/
    pCur -= 4;
    *(int *)pCur = argc; /* int argc */

    /*pCur -=4;*/ /*����Ϊʲô��Ҫ��4?
        ��Ϊc����������__start�����Ĵ����ʱ��
        ���չ�����Ϊ����__start����һ�� 0(%esp)�Ƿ��ص�ַ, 
        4(%esp)��argc, 8(%esp)��argv, ������û��һ��call __start�Ĺ���,ջ��û�б�ѹ�뷵�ص�ַ,
        ����������ص�ַռ�õĿռ仹��Ҫ�ճ���,��������ȷ�õ�argc,argv��������*/
    stack_bottom = pCur-user_space_org+USER_SPACE_VADDR_HEAD;/*�����ַ :)*/

    iret = load_proc_from_fs("/bin/sh", pid);
    if (iret)
    {
        return -1;
    }


	/*���ý��̿������ݽṹ*/
	p_proc = &g_procs[pid];
	p_proc->ldt_sel = g_first_ldt_selector + 8*pid;

    /*ldtֱ��ӳ��4Gȫ�ռ䣬�൱��������ʽ����*/
	init_descriptor( (TDescriptor*)&p_proc->ldts[0], 0, 0xfffff, 
			DA_C|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));
	init_descriptor( (TDescriptor*)&p_proc->ldts[1], 0, 0xfffff, 
			DA_DRW|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));

	init_descriptor( &g_gdt[ p_proc->ldt_sel >> 3 ], 
			(uint32_t)p_proc->ldts, 
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1, 
			DA_LDT);
	init_descriptor(&g_gdt[(g_first_tss_selector+8*pid) >> 3],
			(uint32_t)&p_proc->tss,
			sizeof(p_proc->tss) - 1,
			DA_386TSS);

    /*α����̵ı������������*/
	memset(&p_proc->tss, 0, sizeof(p_proc->tss));
	p_proc->pid			= pid;
	p_proc->tss.ss0		= g_kernel_data_selector; /*�����жϡ��쳣�����л��������ջ*/
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid);

    /*SA_TIL���Ա�ʾ��ѡ������һ��ldt��ѡ���ӣ���Ҫ�ο�ldtr�Ĵ�������������ҵ�ldt*/
	p_proc->tss.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.gs     = p_proc->tss.ds;
	p_proc->tss.ldt     = p_proc->ldt_sel;

    /* paging :) */
    if (setup_paging_for_proc(pid) != 0) { return -2;}
	p_proc->tss.cr3     = get_cr3_for_proc(pid);
    //_set_cr3(p_proc->tss.cr3);
	p_proc->tss.eip    = USER_SPACE_VADDR_HEAD;
	p_proc->tss.esp    = stack_bottom;
    /*
    printk("eip=0x%x, cs=0x%x, esp=0x%x, ss=0x%x\n",
        p_proc->tss.eip , p_proc->tss.cs, p_proc->tss.esp, p_proc->tss.ss);
    while (1);
    */
    /*ֻ�е���ǰ���еĴ���ģãУ�С�ڵ��ڣɣϣУ�ʱ��������ܷ��ʣɣϵ�ַ�ռ䡣*/
	p_proc->tss.eflags = 0x202;   // IF=1, IOPL=0, bit 2 is always 1.
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 35;
	p_proc->counter		= 0xff - p_proc->nice;
	p_proc->ppid		= 0;
	return 0;
}
#if 0 /*����һ�£��Ķ�̫���� :) */
static int create_first_user_proc(const char * pathname)
{
	int iret;
	int len;
	int pid;
    int argc;
    uint32_t stack_bottom;
    char * * argv;

	TProcess * p_proc = NULL;
    uint32_t user_space_org;


    pid = 1;
    user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;    /*���̿ռ�������ڴ濪ʼλ��*/
    /*����һ��λ��ջ�׶˵�argc/argv*/
    {
        int arglen = strlen(pathname)+1;
        argc = 1;
        stack_bottom = (PROC_SPACE - 1 - arglen - sizeof(char*)*(argc+1) ) &  0xfffffffc;
        char * pCur = (char*)(user_space_org + PROC_SPACE - 1 - arglen - sizeof(char*)*(argc+1));
        argv = (char **)(pCur);
        pCur += sizeof(char*)*(argc+1);
        {
            memcpy(pCur, pathname, arglen);
            argv[0] = pCur - user_space_org;
            pCur += len;
        }
        argv[1] = NULL;
    }
    *(int *)(user_space_org+stack_bottom-4) = (uint32_t)argv - user_space_org;
    *(int *)(user_space_org+stack_bottom-8) = argc;
    stack_bottom -= 12; /*����Ϊʲô��12������8�� 
        ��Ϊc����������__start�����Ĵ����ʱ��
        ���չ�����Ϊ����__start����һ�� 0(%esp)�Ƿ��ص�ַ, 
        4(%esp)��argc, 8(%esp)��argv, ������û��һ��call __start�Ĺ���,ջ��û�б�ѹ�뷵�ص�ַ,
        ����������ص�ַռ�õĿռ仹��Ҫ�ճ���,��������ȷ�õ�argc,argv��������*/

    iret = load_proc_from_fs("/bin/sh", pid);
    if (iret)
    {
        return -1;
    }


	/*���ý��̿������ݽṹ*/
	p_proc = &g_procs[pid];
	p_proc->ldt_sel = g_first_ldt_selector + 8*pid;

	init_descriptor( (TDescriptor*)&p_proc->ldts[0], user_space_org, PROC_SPACE, 
			DA_C|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));
	init_descriptor( (TDescriptor*)&p_proc->ldts[1], user_space_org, PROC_SPACE, 
			DA_DRW|DA_32|DA_PAGE| ( PRIVILEGE_USER << 5));

	init_descriptor( &g_gdt[ p_proc->ldt_sel >> 3 ], 
			(uint32_t)p_proc->ldts, 
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1, 
			DA_LDT);
	init_descriptor(&g_gdt[(g_first_tss_selector+8*pid) >> 3],
			(uint32_t)&p_proc->tss,
			sizeof(p_proc->tss) - 1,
			DA_386TSS);

    /*α����̵ı������������*/
	memset(&p_proc->tss, 0, sizeof(p_proc->tss));
	p_proc->pid			= pid;
	p_proc->tss.ss0		= g_kernel_data_selector; /*�����жϡ��쳣�����л��������ջ*/
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid);

    /*SA_TIL���Ա�ʾ��ѡ������һ��ldt��ѡ���ӣ���Ҫ�ο�ldtr�Ĵ�������������ҵ�ldt*/
	p_proc->tss.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->tss.gs     = p_proc->tss.ds;
	p_proc->tss.ldt     = p_proc->ldt_sel;
	p_proc->tss.cr3     = _get_cr3();
	p_proc->tss.eip    = 0;
	p_proc->tss.esp    = stack_bottom;
    /*ֻ�е���ǰ���еĴ���ģãУ�С�ڵ��ڣɣϣУ�ʱ��������ܷ��ʣɣϵ�ַ�ռ䡣*/
	p_proc->tss.eflags = 0x202;   // IF=1, IOPL=0, bit 2 is always 1.
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 35;
	p_proc->counter		= 0xff - p_proc->nice;
	p_proc->ppid		= 0;
	return 0;
}
#endif
static void init_8259A()
{
	out_byte(INT_M_CTL,	0x11);			/* Master 8259, ICW1.*/
	out_byte(INT_S_CTL,	0x11);			/* Slave  8259, ICW1.*/
	out_byte(INT_M_CTLMASK,	INT_VECTOR_IRQ0);	/* Master 8259, ICW2. ���� '��8259' ���ж���ڵ�ַΪ 0x20.*/
	out_byte(INT_S_CTLMASK,	INT_VECTOR_IRQ8);	/* Slave  8259, ICW2. ���� '��8259' ���ж���ڵ�ַΪ 0x28*/
	out_byte(INT_M_CTLMASK,	0x4);			/* Master 8259, ICW3. IR2 ��Ӧ '��8259'.*/
	out_byte(INT_S_CTLMASK,	0x2);			/* Slave  8259, ICW3. ��Ӧ '��8259' �� IR2.*/
	out_byte(INT_M_CTLMASK,	0x1);			/* Master 8259, ICW4.*/
	out_byte(INT_S_CTLMASK,	0x1);			/* Slave  8259, ICW4.*/

	out_byte(INT_M_CTLMASK,	0xFC);	/* Master 8259, OCW1. */
	out_byte(INT_S_CTLMASK,	0xFF);	/* Slave  8259, OCW1. */
}
/*����ʱ���жϵ�Ƶ��*/
static void init_8253()
{
	out_byte(TIMER_MODE, RATE_GENERATOR);
	out_byte(TIMER0, (uint8_t)(TIMER_FREQ/HZ)); /*д��8�ֽ�*/
	out_byte(TIMER0, (uint8_t)((TIMER_FREQ/HZ)>>8)) ;/*д��8�ֽ�*/
}

static void create_proc_0()
{
	TProcess * p_proc = NULL;
	TSS	tmp_tss;

	/***************����0����idle***************************/
	p_proc = &g_procs[0];
	p_proc->pid			= 0;
	p_proc->ldt_sel = g_first_ldt_selector;

    /*���ں˴���κ����ݶ���ȫ����������ı���������̵ľֲ���������ĸģ��͵õ��˸ý��̵�ldt*/
	memcpy((void*)&p_proc->ldts[0], &g_gdt[g_kernel_code_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[0].dr_attributes = (p_proc->ldts[0].dr_attributes) | ( PRIVILEGE_KRNL << 5); 
	memcpy((void*)&p_proc->ldts[1], &g_gdt[g_kernel_data_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[1].dr_attributes = (p_proc->ldts[1].dr_attributes ) | ( PRIVILEGE_KRNL << 5);   

    /*
     *  lldtָ����ص���һ����GDT�е�ѡ����, ��ѡ���������Ķ���ldt�����ڵ�һ���ڴ�
     *  lgdtָ����ص���һ����ַ���õ�ִַ��GDT�����ڵ�λ��
     */
    /*��ʼ��GDT�е�һ������ñ����������̱���ldts�ֶ���һ���ڴ�*/
	init_descriptor( &g_gdt[ p_proc->ldt_sel >> 3 ], 
			(uint32_t)p_proc->ldts,   /*ƫ��*/
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1,  /*��С*/
			DA_LDT);
    /*��ʼ��GDT�е�һ������ñ����������̱���tss�ֶ���һ���ڴ�*/
	init_descriptor(&g_gdt[(g_first_tss_selector+0) >> 3],
			(uint32_t)&p_proc->tss,
			sizeof(p_proc->tss) - 1,
			DA_386TSS);

    /*��ʼα�����0�ı����������*/
	memset(&p_proc->tss, 0, sizeof(p_proc->tss));
	p_proc->tss.ss0		= g_kernel_data_selector; 
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid); /*����ʹ�����ջ����Ϊû�з�����Ȩ����仯�����л���ջ*/
	p_proc->tss.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL ; /*SA_TIL���Ա�ʾ��ѡ������һ��ldt��ѡ���ӣ���Ҫ�ο�ldtr�Ĵ�������������ҵ�ldt*/
	p_proc->tss.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL ;
	p_proc->tss.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL ;
	p_proc->tss.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL ;
	p_proc->tss.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL ;
	p_proc->tss.gs     = p_proc->tss.ds;
	//printk("ds=0x%x\n", p_proc->tss.ds );
	p_proc->tss.ldt     = p_proc->ldt_sel;
	p_proc->tss.cr3     = _get_cr3();
	p_proc->tss.eip    = (uint32_t)do_task;
	p_proc->tss.esp    = (uint32_t) ((KERNEL_ORG-1) & 0xfffffffc);
    /*ֻ�е���ǰ���еĴ���ģãУ�С�ڵ��ڣɣϣУ�ʱ��������ܷ��ʣɣϵ�ַ�ռ䡣*/
	p_proc->tss.eflags = 0x1202;   // IF=1, IOPL=1, bit 2 is always 1.

	p_proc->tss.iobase	= sizeof(p_proc->tss);	/* û��I/O���λͼ */

	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice	= 15;
	p_proc->counter		= 0xff - p_proc->nice;
	p_proc->ppid		= 0;

#if 1
	/*����һ����ʱ��tss�����֣����ڱ��浱ǰ�ġ����̡���Ϣ
	 * ��Ȼ��Щ��Ϣ�Ժ��ò����ˣ����ǵ��еط����棬�����д�������ַ0������
	 * ����ط�����tmp_tss
	 */
	init_descriptor(&g_gdt[(g_tmp_tss_selector) >> 3],
			(uint32_t)&tmp_tss,
			sizeof(tmp_tss) - 1,
			DA_386TSS);
	_ltr(g_tmp_tss_selector);
#endif

    /*�����л�����ʼ���н���0
     * �����л������ǽ���ǰ���̵������ı�����tr�Ĵ���ָ���tss���У�
     * ���Ҵ������н��̵�tss���лָ������н��̵�������
     */
	g_current = &g_procs[0];
    _switch(g_current, NULL, g_first_tss_selector+8*0);
}
static void init_intr_table()
{
	int i;

	memset(g_idt, 0, sizeof(g_idt));

    //init_gate(TGate* pgate, Selector s, uint32_t offset, uint8_t reserve, uint8_t attr)
	init_gate(&g_idt[0], g_kernel_code_selector, (uint32_t)_EH_divide_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[1], g_kernel_code_selector, (uint32_t)_EH_debug_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[2], g_kernel_code_selector, (uint32_t)_EH_not_mask_intr, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[3], g_kernel_code_selector, (uint32_t)_EH_debug_break, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[4], g_kernel_code_selector, (uint32_t)_EH_over_flow, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[5], g_kernel_code_selector, (uint32_t)_EH_break_limit, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[6], g_kernel_code_selector, (uint32_t)_EH_undefined_op, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[7], g_kernel_code_selector, (uint32_t)_EH_no_coproc, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[8], g_kernel_code_selector, (uint32_t)_EH_double_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[9], g_kernel_code_selector, (uint32_t)_EH_coproc_break_limit, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[10], g_kernel_code_selector, (uint32_t)_EH_invalid_tss, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[11], g_kernel_code_selector, (uint32_t)_EH_no_seg, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[12], g_kernel_code_selector, (uint32_t)_EH_stack_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[13], g_kernel_code_selector, (uint32_t)_EH_general_protect_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[14], g_kernel_code_selector, (uint32_t)_EH_page_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[15], g_kernel_code_selector, (uint32_t)_EH_reserve15, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[16], g_kernel_code_selector, (uint32_t)_EH_float_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[17], g_kernel_code_selector, (uint32_t)_EH_align_check, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[18], g_kernel_code_selector, (uint32_t)_EH_machine_check, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[19], g_kernel_code_selector, (uint32_t)_EH_simd_float_error, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));



	init_gate(&g_idt[INT_VECTOR_IRQ0+0], g_kernel_code_selector, (uint32_t) _IH_irq00, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+1], g_kernel_code_selector, (uint32_t) _IH_irq01, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+2], g_kernel_code_selector, (uint32_t) _IH_irq02, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+3], g_kernel_code_selector, (uint32_t) _IH_irq03, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+4], g_kernel_code_selector, (uint32_t) _IH_irq04, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+5], g_kernel_code_selector, (uint32_t) _IH_irq05, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+6], g_kernel_code_selector, (uint32_t) _IH_irq06, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ0+7], g_kernel_code_selector, (uint32_t) _IH_irq07, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));

	init_gate(&g_idt[INT_VECTOR_IRQ8+0], g_kernel_code_selector, (uint32_t) _IH_irq08, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+1], g_kernel_code_selector, (uint32_t) _IH_irq09, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+2], g_kernel_code_selector, (uint32_t) _IH_irq10, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+3], g_kernel_code_selector, (uint32_t) _IH_irq11, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+4], g_kernel_code_selector, (uint32_t) _IH_irq12, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+5], g_kernel_code_selector, (uint32_t) _IH_irq13, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+6], g_kernel_code_selector, (uint32_t) _IH_irq14, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));
	init_gate(&g_idt[INT_VECTOR_IRQ8+7], g_kernel_code_selector, (uint32_t) _IH_irq15, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));

	init_gate(&g_idt[INT_VECTOR_SYSCALL], g_kernel_code_selector, (uint32_t)_IH_sys_call, 0, DA_386IGate|(PRIVILEGE_USER<<5));

	g_idtr48.it_len = sizeof(g_idt) - 1;
	g_idtr48.it_base = (uint32_t)g_idt;
}
