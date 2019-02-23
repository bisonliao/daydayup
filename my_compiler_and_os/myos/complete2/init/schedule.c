#include "const_def.h"
#include "global.h"
#include "api.h"

void sleep_on(TProcess **p)
{
/*
 * 通过每个进程的堆栈里的这个变量，将一条进程链表隐式的建立起来
 * 该进程链表就是所有等待某个事件的进程的集合，在这个事件发生时，
 * 这些进程将被依次唤醒
 *
 * 这个隐式链表里的进程处于不可被中断的睡眠状态，因为如果其中
 * 某个进程被粗暴的kill掉，这个链表就断了，内核中的数据结构就乱七八糟了
 */
	TProcess *tmp; 

	if (!p)
		return;
	if (g_current == &g_procs[0])
		panic("process[0] trying to sleep"); 
	tmp = *p;
	*p = (TProcess *)g_current;
	g_current->status = PROC_STATUS_WAITING;

	schedule(); /*运行其它进程*/

	/*当cpu执行到这里的时候， 表示当前进程被重新调度，继续运行，*/
	if (tmp)
		tmp->status = PROC_STATUS_RUNNING;/* 将隐式链表里前一个进程唤醒*/
}       
void wake_up(TProcess **p)
{
	if (p && *p) {
		(**p).status=PROC_STATUS_RUNNING;
		*p=NULL;
	}
}


/*进程调度函数*/
int schedule()
{
	int i = 0;

	volatile int iMax = -1;
	int	iNeedRun = 0; /*是否有需要运行的进程的标志*/

	TProcess * prev;  /*保存当前运行的进程控制块指针*/
	volatile int  prev_index;

	prev = (TProcess *)g_current;
	prev_index = (int)(prev - &g_procs[0]);


	for (i = 0; i < MAX_PROC_NR; ++i)
	{
		if (i != 0 && g_procs[i].pid == 0)
		{
			continue;
		}
		if (g_procs[i].status == PROC_STATUS_SLEEPING &&
                g_procs[i].alarm > 0 &&
				g_ticks >= g_procs[i].alarm)
		{
			g_procs[i].alarm = 0;
			g_procs[i].status = PROC_STATUS_RUNNING;
		}
		if (g_procs[i].status != PROC_STATUS_RUNNING)
		{
			continue;
		}
		//至此，status都是 RUNNING
		iNeedRun = 1;
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
		iMax = 0;
		goto modify_desc;
	}
	if (iMax != -1)
	{
		goto modify_desc;
	}
	/**
	 * 如果没有找到最大的counter值, iMax == -1 
	 * 重新设置各待运行进程的时间片
	 */
	for (i = 0; i < MAX_PROC_NR; ++i)
	{
		if (i !=0 && g_procs[i].pid == 0)
		{
			continue;
		}
		if (g_procs[i].status != PROC_STATUS_RUNNING)
		{
			continue;
		}
		//至此，status都是 RUNNING
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
	if (iMax == -1)
	{
		iMax = 0;
		goto modify_desc;
	}

modify_desc:
    /*
    if (iMax == 3) 
    {
        printk("to run proc#3\n");
        while (1) ;
    }
    */
	g_current = &(g_procs[iMax]); /*待运行的进程*/

    //printk("run to %s, %s, %d\n", __FUNCTION__, __FILE__, __LINE__);
	if (g_current != prev)
	{
		/*
		 * 将tss描述符的忙标志去掉,否则在switch作ltr的时候异常
		 *	也就是将tss的第6个字节的低4位的值改为9
		 * 其实直接重新init_descriptor也可以
		 */
		unsigned char * pc = (unsigned char*)&g_gdt[(g_first_tss_selector+8*iMax) >> 3];
		*(pc+5) = (*(pc+5)) & 0xf0u + 9;
		//printk("before switch , proc#%d's esp=%u\n", prev_index, _get_esp());
		_switch(g_current, prev, g_first_tss_selector+8*iMax);
		//printk("after switch , proc#%d's esp=%u\n", prev_index, _get_esp());
	}
	return 0;
}
void handle_timer_interrupt()
{
	static uint16_t acc = 0;
	++g_ticks;	
	++acc;
	if (acc >= HZ) /*积满一秒*/
	{
		acc = 0;
		++g_uptime;
	}
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

	/*清屏幕*/
	/*
	   for (i = 0; i < 25; ++i)
	   {
	   print_str("                                                     \n");
	   }
	 */
	/*
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
	 */
	printk("!!!exception occurs!\n%s\n", desc[vecno]);
	printk("!!!errno=%u, 0x%x\n", errno, errno);
	printk("!!!eip=%u, 0x%x\n", eip, eip);
	printk("!!!cs=%u, 0x%x\n", cs, cs);
	printk("!!!eflags=%u, 0x%x\n", eflags, eflags);
	return;
}
