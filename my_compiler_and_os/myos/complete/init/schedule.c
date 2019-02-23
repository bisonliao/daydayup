#include "const_def.h"

#include "global.h"
#include "api.h"


/*进程调度函数*/
int schedule()
{
	int i = 0;
	static int j  = 0;
	uint64_t busy_flag;

	int iMax = -1;
	int	iNeedRun = 0; /*是否有需要运行的进程的标志*/

	static int proc1 = 0;
	static int proc2 = 0;

	g_current = NULL;

	/* process 0 优先考虑*/
	task_is_busy(&busy_flag);
	if (busy_flag) /*如果有必要调度task*/
	{
		g_current = &(g_procs[0]);
		goto modify_desc;
	}
	/*再考虑其他用户进程*/
	for (i = 1; i < MAX_PROC_NR; ++i)
	{
		if (g_procs[i].pid == 0)
		{
			continue;
		}
		if (g_procs[i].status == PROC_STATUS_SLEEPING &&
			g_ticks >= g_procs[i].alarm)
		{
			g_procs[i].alarm = 0;
			g_procs[i].status = PROC_STATUS_RUNNING;
		}
		if (g_procs[i].status == PROC_STATUS_RUNNING)
		{
			iNeedRun = 1;
		}
		if (g_procs[i].status == PROC_STATUS_RUNNING && g_procs[i].counter > 0)
		{
			if (iMax == -1 ||
				g_procs[i].counter > g_procs[iMax].counter)
			{
				iMax = i;
			}
		}
	}
	/*如果没有进程需要调度*/
	if (iNeedRun == 0)
	{
		return -1;
	}
	if (iMax != -1)
	{
		g_current = &(g_procs[iMax]);
		goto modify_desc;
	}
	/**
	 * 如果没有找到最大的counter值, iMax == -1 
	 * 重新设置各待运行进程的时间片
	 */
	for (i = 1; i < MAX_PROC_NR; ++i)
	{
		if (g_procs[i].pid == 0)
		{
			continue;
		}
		if (g_procs[i].status == PROC_STATUS_RUNNING)
		{
			g_procs[i].counter = 0xff - g_procs[i].nice;
			if (iMax == -1 ||
				g_procs[i].counter > g_procs[iMax].counter)
			{
				iMax = i;
			}
		}
	}
	g_current = &(g_procs[iMax]);
	goto modify_desc;

modify_desc:

#if 0
	if (iMax == 1)  proc1++;
	if (iMax == 2)  proc2++;
	if (proc1 && (proc1 % 200 == 0))
	{
		printk("%d, %d\n", proc1, proc2);
	}
#endif
    /* 初始化ldt所在gdt的描述字*/
    init_descriptor( &g_gdt[ g_idt_selector >> 3 ],
                    (uint32_t)g_current->ldts,
                    sizeof(TDescriptor)*MAX_LDT_ENT_NR-1,
                    DA_LDT);


    /* tss 描述进程被中断后进入内核后，ss/esp应该为什么值*/
    memset(&g_tss, 0, sizeof(g_tss));
    g_tss.ss0       = g_kernel_data_selector;
    g_tss.esp0      = (uint32_t)(((char*)g_current)+sizeof(TRegContext));

    init_descriptor(&g_gdt[g_tss_selector >> 3],
            (uint32_t)&g_tss,
            sizeof(g_tss) - 1,
            DA_386TSS);
    g_tss.iobase    = sizeof(g_tss);    /* 没有I/O许可位图 */

	return 0;
}
void handle_timer_interrupt()
{
	++g_ticks;	
	if (g_current != NULL && g_current->counter > 0)
	{
		--(g_current->counter);
	}
	return;
}
void handle_exception(uint32_t vecno, uint32_t errno , uint32_t eip, uint32_t cs, uint32_t eflags)
{
	static char desc[][32] = {
		"divide_error",
		"debug_error",
		"not_mask_intr",
		"debug_break",
		"over_flow",
		"break_limit",
		"undefined_op",
		"no_coproc",
		"double_error",
		"coproc_break_limit",
		"invalid_tss",
		"no_seg",
		"stack_error",
		"general_protect_error",
		"page_error",
		"reserve15",
		"float_error",
		"align_check",
		"machine_check",
		"simd_float_error"};
	int i;

	/*清屏幕*/
	for (i = 0; i < 25; ++i)
	{
		print_str("                                                     \n");
	}

	print_str("exception occurs!\n");
	print_str(desc[vecno]);
	print_str("\nerrno=0x");
	print_hex(errno);
	print_str("\neip=0x");
	print_hex(eip);
	print_str("\ncs=0x");
	print_hex(cs);
	print_str("\neflags=0x");
	print_hex(eflags);
	return;
}
