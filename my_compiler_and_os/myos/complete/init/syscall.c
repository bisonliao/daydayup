#include "const_def.h"

#include "global.h"

typedef uint32_t (*SYSCALL_FUNC_PTR)(uint32_t,uint32_t,uint32_t,uint32_t);


uint32_t       execute_sys_call(uint32_t eax,
		uint32_t ebx,
		uint32_t ecx,
		uint32_t edx)
{
	SYSCALL_FUNC_PTR func;	
	uint32_t tmp;
	func = NULL;
	g_current->regs.eax = 0xffffffff; /*默认失败*/
	if (eax >= g_syscall_nr)
	{
		return -1;
	}

	func = (SYSCALL_FUNC_PTR)(g_syscall_entry[eax]);
	return  func(eax, ebx, ecx, edx);
	return 0;
}

uint32_t   sys_get_ticks_lo(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx)
{
		g_current->regs.eax = g_ticks & 0xffffffff;
		return 0;				
}
uint32_t   sys_get_ticks_hi(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx)
{
		g_current->regs.eax = (g_ticks >> 32) & 0xffffffff;
		return 0;				
}
uint32_t   sys_sleep(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx)
{
		g_current->alarm = g_ticks + ebx; /*从现在起 ebx个ticks后唤醒*/
		g_current->status = PROC_STATUS_SLEEPING;
		return 0;
}
uint32_t   sys_read(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx)
{
	g_current->regs.eax = 0;
	return 0;
}
/* ssize_t write(int fd, const void *buf, size_t count); */
uint32_t   sys_write(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx)
{
	uint32_t addr;
	const char * p = NULL;
	int i;

	if (ecx == 0 || edx < 1)
	{
		return 0;		
	}
	if (ebx != 1) /*暂时只支持输出到标准输出*/
	{
		return 0;		
	}
	user_addr_to_virtual_addr(g_current, ecx, &addr);
	p = (const char * )addr;
	for (i = 0; i < edx; ++i)
	{
		print_chr(p[i]);
	}
	g_current->regs.eax = edx;
	return 0;
}

/* ssize_t hd(uint32_t abs_sector, void * buf, uint32_t cmd); */
uint32_t   sys_hd(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx)
{
	uint32_t addr;
	if (ecx == 0 || edx != WIN_READ && edx != WIN_WRITE)
	{
		return 0;
	}
	user_addr_to_virtual_addr(g_current, ecx, &addr);
	//printk("user_addr_to_virtual_addr return addr=0x%x\n", (uint32_t)addr);
	if (hd_add_request( (char *)addr, ebx, edx, g_current->pid) )
	{
		return 0;
	}
	g_current->regs.eax = 0;
	g_current->status = PROC_STATUS_WAITING;
	return 0;
}
