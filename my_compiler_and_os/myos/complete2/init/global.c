#include "const_def.h"
#include "global.h"


volatile uint8_t		g_scr_line	 __attribute__ ((section ("data"))) 	=  0;	
volatile uint8_t		g_scr_colume	 __attribute__ ((section ("data"))) =  0;	
/* GDT全局变量*/
TDescriptor g_gdt[MAX_GDT_ENT_NR] ;
TGdtr48		g_gdtr48;

/*进程的内核栈*/
unsigned char g_krnl_stack[MAX_PROC_NR][KRNL_STACK_SZ];

/* IDT 全局变量*/
TGate		g_idt[255];
TIdtr48		g_idtr48;

/*进程全局变量*/
TProcess    g_procs[MAX_PROC_NR] __attribute__((section(".data")));
volatile TProcess 	* g_current = NULL;

/*文件表*/
struct file  g_file_table[MAX_FILE_TABLE];

/*系统调用*/
volatile uint32_t	g_syscall_param_eax;
volatile uint32_t	g_syscall_param_ebx;
volatile uint32_t	g_syscall_param_ecx;
volatile uint32_t	g_syscall_param_edx;
volatile uint32_t	g_syscall_param_ds;
volatile uint64_t	g_ticks  __attribute__ ((section ("data")))  = 0;
volatile uint8_t 	g_hd_sync_flag __attribute__ ((section ("data")))  = 1;

/*硬盘参数*/
THdParam	g_hd_param;
volatile void (*do_hd)()	= NULL;

/*系统启动时间(second)*/
volatile uint32_t g_startup_time;

/*系统连续运行时间(second)*/
volatile uint32_t g_uptime;

