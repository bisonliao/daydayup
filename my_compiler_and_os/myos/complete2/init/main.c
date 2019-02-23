#include "const_def.h"

#include "global.h"
#include "time.h"
#include "api.h"
#include "fs.h"
#include "redefine.h"



static void init_intr_table();
/*设置时钟中断的频率*/
static void init_8253();
static void create_proc_0();
static void init_8259A();
int create_first_user_proc(const char * pathname);


void c_start()
{
	unsigned int i;

	/*清屏幕*/
	for (i = 0; i < 25; ++i)
	{
		printk("                                                     \n");
	}

	_setup_paging(); /*设置分页*/

	init_8259A();   /*初始化中断控制器*/
	init_intr_table(); /*初始化中断向量表*/
	_lidt();    /*加载中断向量表*/
	init_8253();    /*初始化时钟控制器*/
	time_init();    
	keyboard_init();
	/* 640K ~ 1M的地方是 bois/显存用的ROM*/
	fs_init();


	/*同步的方式读取/bin/sh文件的内容，初始化第一个用户态进程*/
	create_first_user_proc("/bin/sh");

	hd_init();

	g_hd_sync_flag = 0; /*清空同步IO标志*/

    /*网卡驱动初始化*/
    /*
    if (LoadDriver() != 1)
    {
        panic("nic init failed!\n");
    }
    */
	/*创建内核态进程0，并切换到该进程*/
	create_proc_0();

}
/* 
 * 原理同create_proc_0，
 * 只是要从磁盘中把进程读入内存
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
    user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;    /*进程空间的物理内存开始位置*/
    /*构造一下位于栈底端的argc/argv*/
    /*他妈这里贼复杂贼绕，要同时兼顾os的物理地址和进程的虚拟地址*/
    char *pCur = NULL;
    {
        int arglen = strlen(pathname)+1;
        argc = 1;
        pCur = (char*)(user_space_org + PROC_SPACE - 1 - arglen - sizeof(char*)*(argc+1)); /*物理地址*/
        /*通过直接访问物理地址，写入参数, 因为内核的页表将线性地址直接映射到物理地址*/
        argv = (char **)(pCur);
        pCur += sizeof(char*)*(argc+1);
        {
            memcpy(pCur, pathname, arglen); /*在栈上放好第一个参数，pathname*/
            argv[0] = (pCur - user_space_org)+USER_SPACE_VADDR_HEAD;  /*这里要写虚拟地址，因为将被进程自己访问*/
        }
        pCur -= sizeof(char*)*(argc+1);
        argv[1] = NULL; /*0还是0，物理地址和虚拟地址一直*/
    }
    /*至此 argv数组已经组织好了，开始压栈*/
    pCur -= 4;
    *(int *)pCur = (uint32_t)argv - user_space_org+USER_SPACE_VADDR_HEAD; /* char **argv 是一个虚拟地址*/
    pCur -= 4;
    *(int *)pCur = argc; /* int argc */

    /*pCur -=4;*/ /*这里为什么还要减4?
        因为c编译器生成__start函数的代码的时候，
        按照惯例认为进入__start的那一刻 0(%esp)是返回地址, 
        4(%esp)是argc, 8(%esp)是argv, 而由于没有一个call __start的过程,栈上没有被压入返回地址,
        但是这个返回地址占用的空间还是要空出来,否则不能正确得到argc,argv两个参数*/
    stack_bottom = pCur-user_space_org+USER_SPACE_VADDR_HEAD;/*虚拟地址 :)*/

    iret = load_proc_from_fs("/bin/sh", pid);
    if (iret)
    {
        return -1;
    }


	/*设置进程控制数据结构*/
	p_proc = &g_procs[pid];
	p_proc->ldt_sel = g_first_ldt_selector + 8*pid;

    /*ldt直接映射4G全空间，相当于跳过段式管理*/
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

    /*伪造进程的被保存的上下文*/
	memset(&p_proc->tss, 0, sizeof(p_proc->tss));
	p_proc->pid			= pid;
	p_proc->tss.ss0		= g_kernel_data_selector; /*发生中断、异常，将切换到这个堆栈*/
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid);

    /*SA_TIL属性表示该选择子是一个ldt的选择子，需要参考ldtr寄存器里的内容来找到ldt*/
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
    /*只有当当前运行的代码的ＣＰＬ小于等于ＩＯＰＬ时，程序才能访问ＩＯ地址空间。*/
	p_proc->tss.eflags = 0x202;   // IF=1, IOPL=0, bit 2 is always 1.
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 35;
	p_proc->counter		= 0xff - p_proc->nice;
	p_proc->ppid		= 0;
	return 0;
}
#if 0 /*备份一下，改动太大了 :) */
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
    user_space_org = FIRST_PROC_ORG+ (pid-1)*PROC_SPACE;    /*进程空间的物理内存开始位置*/
    /*构造一下位于栈底端的argc/argv*/
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
    stack_bottom -= 12; /*这里为什么是12而不是8？ 
        因为c编译器生成__start函数的代码的时候，
        按照惯例认为进入__start的那一刻 0(%esp)是返回地址, 
        4(%esp)是argc, 8(%esp)是argv, 而由于没有一个call __start的过程,栈上没有被压入返回地址,
        但是这个返回地址占用的空间还是要空出来,否则不能正确得到argc,argv两个参数*/

    iret = load_proc_from_fs("/bin/sh", pid);
    if (iret)
    {
        return -1;
    }


	/*设置进程控制数据结构*/
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

    /*伪造进程的被保存的上下文*/
	memset(&p_proc->tss, 0, sizeof(p_proc->tss));
	p_proc->pid			= pid;
	p_proc->tss.ss0		= g_kernel_data_selector; /*发生中断、异常，将切换到这个堆栈*/
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid);

    /*SA_TIL属性表示该选择子是一个ldt的选择子，需要参考ldtr寄存器里的内容来找到ldt*/
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
    /*只有当当前运行的代码的ＣＰＬ小于等于ＩＯＰＬ时，程序才能访问ＩＯ地址空间。*/
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
	out_byte(INT_M_CTLMASK,	INT_VECTOR_IRQ0);	/* Master 8259, ICW2. 设置 '主8259' 的中断入口地址为 0x20.*/
	out_byte(INT_S_CTLMASK,	INT_VECTOR_IRQ8);	/* Slave  8259, ICW2. 设置 '从8259' 的中断入口地址为 0x28*/
	out_byte(INT_M_CTLMASK,	0x4);			/* Master 8259, ICW3. IR2 对应 '从8259'.*/
	out_byte(INT_S_CTLMASK,	0x2);			/* Slave  8259, ICW3. 对应 '主8259' 的 IR2.*/
	out_byte(INT_M_CTLMASK,	0x1);			/* Master 8259, ICW4.*/
	out_byte(INT_S_CTLMASK,	0x1);			/* Slave  8259, ICW4.*/

	out_byte(INT_M_CTLMASK,	0xFC);	/* Master 8259, OCW1. */
	out_byte(INT_S_CTLMASK,	0xFF);	/* Slave  8259, OCW1. */
}
/*设置时钟中断的频率*/
static void init_8253()
{
	out_byte(TIMER_MODE, RATE_GENERATOR);
	out_byte(TIMER0, (uint8_t)(TIMER_FREQ/HZ)); /*写低8字节*/
	out_byte(TIMER0, (uint8_t)((TIMER_FREQ/HZ)>>8)) ;/*写高8字节*/
}

static void create_proc_0()
{
	TProcess * p_proc = NULL;
	TSS	tmp_tss;

	/***************进程0负责idle***************************/
	p_proc = &g_procs[0];
	p_proc->pid			= 0;
	p_proc->ldt_sel = g_first_ldt_selector;

    /*将内核代码段和数据段在全局描述表里的表项拷贝到进程的局部描述表里，改改，就得到了该进程的ldt*/
	memcpy((void*)&p_proc->ldts[0], &g_gdt[g_kernel_code_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[0].dr_attributes = (p_proc->ldts[0].dr_attributes) | ( PRIVILEGE_KRNL << 5); 
	memcpy((void*)&p_proc->ldts[1], &g_gdt[g_kernel_data_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[1].dr_attributes = (p_proc->ldts[1].dr_attributes ) | ( PRIVILEGE_KRNL << 5);   

    /*
     *  lldt指令，加载的是一个在GDT中的选择子, 该选择子描述的段是ldt表所在的一段内存
     *  lgdt指令，加载的是一个地址，该地址执行GDT表所在的位置
     */
    /*初始化GDT中的一个表项，该表项描述进程表中ldts字段这一段内存*/
	init_descriptor( &g_gdt[ p_proc->ldt_sel >> 3 ], 
			(uint32_t)p_proc->ldts,   /*偏移*/
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1,  /*大小*/
			DA_LDT);
    /*初始化GDT中的一个表项，该表项描述进程表中tss字段这一段内存*/
	init_descriptor(&g_gdt[(g_first_tss_selector+0) >> 3],
			(uint32_t)&p_proc->tss,
			sizeof(p_proc->tss) - 1,
			DA_386TSS);

    /*开始伪造进程0的保存的上下文*/
	memset(&p_proc->tss, 0, sizeof(p_proc->tss));
	p_proc->tss.ss0		= g_kernel_data_selector; 
	p_proc->tss.esp0		= (uint32_t)GET_KRNL_STACK_START(p_proc->pid); /*不会使用这个栈，因为没有发生特权级别变化，不切换堆栈*/
	p_proc->tss.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL ; /*SA_TIL属性表示该选择子是一个ldt的选择子，需要参考ldtr寄存器里的内容来找到ldt*/
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
    /*只有当当前运行的代码的ＣＰＬ小于等于ＩＯＰＬ时，程序才能访问ＩＯ地址空间。*/
	p_proc->tss.eflags = 0x1202;   // IF=1, IOPL=1, bit 2 is always 1.

	p_proc->tss.iobase	= sizeof(p_proc->tss);	/* 没有I/O许可位图 */

	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice	= 15;
	p_proc->counter		= 0xff - p_proc->nice;
	p_proc->ppid		= 0;

#if 1
	/*构建一个临时的tss描述字，用于保存当前的“进程”信息
	 * 虽然这些信息以后都用不着了，但是得有地方保存，否则会写坏物理地址0的内容
	 * 这个地方就是tmp_tss
	 */
	init_descriptor(&g_gdt[(g_tmp_tss_selector) >> 3],
			(uint32_t)&tmp_tss,
			sizeof(tmp_tss) - 1,
			DA_386TSS);
	_ltr(g_tmp_tss_selector);
#endif

    /*进程切换，开始运行进程0
     * 进程切换，就是将当前进程的上下文保存在tr寄存器指向的tss段中，
     * 并且从欲运行进程的tss段中恢复欲运行进程的上下文
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
