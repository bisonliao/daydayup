#include "const_def.h"
#include "global.h"


uint8_t		g_scr_line		=  0;	
uint8_t		g_scr_colume	=  0;	
/* GDT全局变量*/
TDescriptor g_gdt[MAX_GDT_ENT_NR] ;
TGdtr48		g_gdtr48;
Selector	g_kernel_code_selector;
Selector	g_kernel_data_selector;
Selector	g_kernel_stack_selector;
Selector	g_kernel_gs_selector;
Selector	g_idt_selector; /*所有进程共用同一个idt选择子*/
Selector	g_tss_selector; /*所有进程共用同一个tss选择子*/

/* IDT 全局变量*/
TGate		g_idt[255];
TIdtr48		g_idtr48;

/*进程全局变量*/
TProcess    g_procs[MAX_PROC_NR];
TProcess 	* g_current = NULL;
TSS			g_tss;

/*系统调用*/
uint32_t	g_syscall_param_eax;
uint32_t	g_syscall_param_ebx;
uint32_t	g_syscall_param_ecx;
uint32_t	g_syscall_param_edx;
uint64_t	g_ticks = 0;

/*硬盘参数*/
THdParam	g_hd_param;
void (*do_hd)()	= NULL;
