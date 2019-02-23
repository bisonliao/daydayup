#include "const_def.h"

#include "global.h"
#include "api.h"
#include "buffer.h"



static void init_intr_table();
/*设置时钟中断的频率*/
static void init_8253();
static void init_process();
static void init_8259A();

void c_start()
{
	int i;
	uint16_t cyl;
	uint8_t head, sector;
	static unsigned char buf[512] ;

	_setup_paging(); /*设置分页*/

	init_8259A();
	init_intr_table();
	_lidt();
	init_8253();
	hd_init();
	buffer_init(10000000, 2000000);

	/*清屏幕*/
	for (i = 0; i < 25; ++i)
	{
		printk("                                                     \n");
	}


	init_process();
	_move_to_process();
}
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

static void init_process()
{
	TProcess* p_proc = &g_procs[0];
	p_proc->ldt_sel = g_idt_selector;
	memcpy(&p_proc->ldts[0], &g_gdt[g_kernel_code_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[0].dr_attributes = (p_proc->ldts[0].dr_attributes) | ( PRIVILEGE_TASK << 5); 
	memcpy(&p_proc->ldts[1], &g_gdt[g_kernel_data_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[1].dr_attributes = (p_proc->ldts[1].dr_attributes ) | ( PRIVILEGE_TASK << 5);   
	p_proc->regs.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_TASK;
	p_proc->regs.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_TASK;
	p_proc->regs.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_TASK;
	p_proc->regs.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_TASK;
	p_proc->regs.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_TASK;
	p_proc->regs.gs     = p_proc->regs.ds;
	p_proc->regs.eip    = (uint32_t)do_task;
	p_proc->regs.esp    = (uint32_t) 2000000; //先简单的放到 2M的位置
	p_proc->regs.eflags = 0x1202;   // IF=0, IOPL=1, bit 2 is always 1.
	p_proc->pid			= 0;
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice	= 0;
	p_proc->counter		= 0xff - p_proc->nice;

	p_proc = &g_procs[1];
	p_proc->ldt_sel = g_idt_selector;
	memcpy(&p_proc->ldts[0], &g_gdt[g_kernel_code_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[0].dr_attributes = (p_proc->ldts[0].dr_attributes) | ( PRIVILEGE_USER << 5); 
	memcpy(&p_proc->ldts[1], &g_gdt[g_kernel_data_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[1].dr_attributes = (p_proc->ldts[1].dr_attributes ) | ( PRIVILEGE_USER << 5);   
	p_proc->regs.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.gs     = p_proc->regs.ds;
	p_proc->regs.eip    = (uint32_t)proc_A;
	p_proc->regs.esp    = (uint32_t) 3000000; //先简单的放到 3M的位置
	p_proc->regs.eflags = 0x202;   // IF=1, IOPL=0, bit 2 is always 1.
	p_proc->pid			= 1;
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 15;
	p_proc->counter		= 0xff - p_proc->nice;

	p_proc = &g_procs[2];
	p_proc->ldt_sel = g_idt_selector;
	memcpy(&p_proc->ldts[0], &g_gdt[g_kernel_code_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[0].dr_attributes = (p_proc->ldts[0].dr_attributes) | ( PRIVILEGE_USER << 5); 
	memcpy(&p_proc->ldts[1], &g_gdt[g_kernel_data_selector >> 3], sizeof(TDescriptor));
	p_proc->ldts[1].dr_attributes = (p_proc->ldts[1].dr_attributes ) | ( PRIVILEGE_USER << 5);   
	p_proc->regs.cs     = ((8 * 0) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.ds     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.es     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.fs     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.ss     = ((8 * 1) & SA_RPL_MASK & SA_TI_MASK) | SA_TIL | RPL_USER;
	p_proc->regs.gs     = p_proc->regs.ds;
	p_proc->regs.eip    = (uint32_t)proc_B;
	p_proc->regs.esp    = (uint32_t) 4000000; //先简单的放到 4M的位置
	p_proc->regs.eflags = 0x202;   // IF=1, IOPL=0, bit 2 is always 1.
	p_proc->pid			= 2;
	p_proc->status		= PROC_STATUS_RUNNING;
	p_proc->nice		= 15;
	p_proc->counter		= 0xff - p_proc->nice;

	p_proc = &g_procs[1];
	g_current    = p_proc;	/*先启动A进程*/

	/* 初始化ldt所在gdt的描述字*/
	init_descriptor( &g_gdt[ g_idt_selector >> 3 ], 
			(uint32_t)g_current->ldts, 
			sizeof(TDescriptor)*MAX_LDT_ENT_NR-1, 
			DA_LDT);


	/* tss 描述进程被中断后进入内核后，ss/esp应该为什么值*/	
	memset(&g_tss, 0, sizeof(g_tss));
	g_tss.ss0		= g_kernel_data_selector;
	g_tss.esp0		= (uint32_t)(((char*)g_current)+sizeof(TRegContext));

	init_descriptor(&g_gdt[g_tss_selector >> 3],
			(uint32_t)&g_tss,
			sizeof(g_tss) - 1,
			DA_386TSS);
	g_tss.iobase	= sizeof(g_tss);	/* 没有I/O许可位图 */

}
static void init_intr_table()
{
	int i;

	memset(g_idt, 0, sizeof(g_idt));

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
	init_gate(&g_idt[INT_VECTOR_IRQ8+7], g_kernel_code_selector, (uint32_t) _IH_irq07, 0, DA_386IGate|(PRIVILEGE_KRNL<<5));

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
	g_idtr48.it_base = g_idt;
}
