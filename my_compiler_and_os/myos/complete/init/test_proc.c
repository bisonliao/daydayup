#include "const_def.h"

#include "api.h"
#include "global.h"



/*进程体A*/
void proc_A()
{
	unsigned int i = 0;
	char buf[512] ;
	while (1)
	{
		snprintf(buf, sizeof(buf), "buf value=%d", i++);
		_hd(0, buf, WIN_WRITE);
		_sleep(10);
	}
}
/*进程体B*/
void proc_B()
{
	unsigned int i = 0;
	char buf[512] ;
	while (1)
	{
		//snprintf(buf, sizeof(buf), "buf value=%d", i++);
		_hd(0, buf, WIN_READ) ;
		printk("buf=[%s]\n",  buf);
		_sleep(100);
	}
}
