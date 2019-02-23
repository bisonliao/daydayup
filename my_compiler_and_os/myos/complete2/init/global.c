#include "const_def.h"
#include "global.h"


volatile uint8_t		g_scr_line	 __attribute__ ((section ("data"))) 	=  0;	
volatile uint8_t		g_scr_colume	 __attribute__ ((section ("data"))) =  0;	
/* GDTȫ�ֱ���*/
TDescriptor g_gdt[MAX_GDT_ENT_NR] ;
TGdtr48		g_gdtr48;

/*���̵��ں�ջ*/
unsigned char g_krnl_stack[MAX_PROC_NR][KRNL_STACK_SZ];

/* IDT ȫ�ֱ���*/
TGate		g_idt[255];
TIdtr48		g_idtr48;

/*����ȫ�ֱ���*/
TProcess    g_procs[MAX_PROC_NR] __attribute__((section(".data")));
volatile TProcess 	* g_current = NULL;

/*�ļ���*/
struct file  g_file_table[MAX_FILE_TABLE];

/*ϵͳ����*/
volatile uint32_t	g_syscall_param_eax;
volatile uint32_t	g_syscall_param_ebx;
volatile uint32_t	g_syscall_param_ecx;
volatile uint32_t	g_syscall_param_edx;
volatile uint32_t	g_syscall_param_ds;
volatile uint64_t	g_ticks  __attribute__ ((section ("data")))  = 0;
volatile uint8_t 	g_hd_sync_flag __attribute__ ((section ("data")))  = 1;

/*Ӳ�̲���*/
THdParam	g_hd_param;
volatile void (*do_hd)()	= NULL;

/*ϵͳ����ʱ��(second)*/
volatile uint32_t g_startup_time;

/*ϵͳ��������ʱ��(second)*/
volatile uint32_t g_uptime;

