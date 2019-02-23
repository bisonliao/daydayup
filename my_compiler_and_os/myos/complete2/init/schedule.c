#include "const_def.h"
#include "global.h"
#include "api.h"

void sleep_on(TProcess **p)
{
/*
 * ͨ��ÿ�����̵Ķ�ջ��������������һ������������ʽ�Ľ�������
 * �ý�������������еȴ�ĳ���¼��Ľ��̵ļ��ϣ�������¼�����ʱ��
 * ��Щ���̽������λ���
 *
 * �����ʽ������Ľ��̴��ڲ��ɱ��жϵ�˯��״̬����Ϊ�������
 * ĳ�����̱��ֱ���kill�����������Ͷ��ˣ��ں��е����ݽṹ�����߰�����
 */
	TProcess *tmp; 

	if (!p)
		return;
	if (g_current == &g_procs[0])
		panic("process[0] trying to sleep"); 
	tmp = *p;
	*p = (TProcess *)g_current;
	g_current->status = PROC_STATUS_WAITING;

	schedule(); /*������������*/

	/*��cpuִ�е������ʱ�� ��ʾ��ǰ���̱����µ��ȣ��������У�*/
	if (tmp)
		tmp->status = PROC_STATUS_RUNNING;/* ����ʽ������ǰһ�����̻���*/
}       
void wake_up(TProcess **p)
{
	if (p && *p) {
		(**p).status=PROC_STATUS_RUNNING;
		*p=NULL;
	}
}


/*���̵��Ⱥ���*/
int schedule()
{
	int i = 0;

	volatile int iMax = -1;
	int	iNeedRun = 0; /*�Ƿ�����Ҫ���еĽ��̵ı�־*/

	TProcess * prev;  /*���浱ǰ���еĽ��̿��ƿ�ָ��*/
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
		//���ˣ�status���� RUNNING
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
	/*���û�н�����Ҫ����*/
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
	 * ���û���ҵ�����counterֵ, iMax == -1 
	 * �������ø������н��̵�ʱ��Ƭ
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
		//���ˣ�status���� RUNNING
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
	g_current = &(g_procs[iMax]); /*�����еĽ���*/

    //printk("run to %s, %s, %d\n", __FUNCTION__, __FILE__, __LINE__);
	if (g_current != prev)
	{
		/*
		 * ��tss��������æ��־ȥ��,������switch��ltr��ʱ���쳣
		 *	Ҳ���ǽ�tss�ĵ�6���ֽڵĵ�4λ��ֵ��Ϊ9
		 * ��ʵֱ������init_descriptorҲ����
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
	if (acc >= HZ) /*����һ��*/
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

	/*����Ļ*/
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
