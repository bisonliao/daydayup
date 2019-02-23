#include "const_def.h"

#include "api.h"
#include "global.h"

static volatile uint64_t iBusyFlag = 0;


/*ר�Ÿ���ϵͳ�ڲ�����Ľ��̣�����key board buffer �Ĵ���*/
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
		 * û��������Ҫ��,��������cpu
		 * do_task��ִ�й����У��ǹر����жϵ�
		 * ������Ҫ��������
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
