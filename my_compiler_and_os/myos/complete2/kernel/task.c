#include "const_def.h"
#include "fs.h"
#include "api.h"

#include "global.h"


/*回收父进程为pid#0的僵尸进程*/
static int do_reclaim_zombie();


/*专门负责系统内部任务的进程，比如key board buffer 的处理*/
void do_task()
{
	uint8_t busy_flag = 0;

	while (1)
	{
        SendPacket2();

		busy_flag = 0; /*默认是清闲*/

        if (do_reclaim_zombie() == 1)
        {
            busy_flag = 1; /*more work to do*/
        }

		if (keyboard_do_task() == 1)
		{
			busy_flag = 1; /* more work to do */
		}


		if (!busy_flag)
		{
			__asm__("hlt");
		}
	}
}


/*回收父进程为pid#0的僵尸进程*/
static int do_reclaim_zombie()
{
    int i;
    char flag = 0;
    int index;
    TProcess * p_proc = NULL;
    static int last_reclaim_index __attribute__((section(".data"))) = 1;
    for (i = 0; i <  10; ++i)
    {
        index = (last_reclaim_index + 1) % MAX_PROC_NR;
        if (index == 0) { index = 1;}
        last_reclaim_index = index;

        p_proc = &g_procs[ index ];

        if (p_proc->pid > 0 &&
            p_proc->status == PROC_STATUS_ZOMBIE &&
            p_proc->ppid == 0)
        {
            flag = 1;
            //printk("reclaim proc#%d...\n", p_proc->pid);
            memset(p_proc, 0, sizeof(TProcess));
        }
    }
    if (flag) 
    {
        return 1;
    }
    return 0; /*比较清闲*/
}

