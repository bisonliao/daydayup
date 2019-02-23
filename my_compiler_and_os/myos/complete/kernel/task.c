#include "const_def.h"

#include "api.h"
#include "global.h"

static volatile uint64_t iBusyFlag = 0;


/*专门负责系统内部任务的进程，比如key board buffer 的处理*/
void do_task()
{
	uint64_t busy_flag;
	
	while (1)
	{
		busy_flag = iBusyFlag;	
		if (busy_flag & TASK_BUSY_FLAG_KEYBOARD)
		{
			keyboard_do_task();
		}
		/**
		 * 没有任务需要做,主动放弃cpu
		 * do_task在执行过程中，是关闭了中断的
		 * 所以需要主动放弃
		 */
		if (iBusyFlag == 0) 
		{
			//printk("task gives up cpu\n");
			_sleep(0);
		}
	}
}

void task_set_busy(uint64_t busy)
{
	iBusyFlag = busy;
}
void  task_is_busy(uint64_t * pflag)
{
	*pflag = iBusyFlag;
}
