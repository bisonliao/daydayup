#include "const_def.h"
#include "global.h"


uint8_t		g_scr_line		=  0;	
uint8_t		g_scr_colume	=  0;	
/* GDTȫ�ֱ���*/
TDescriptor g_gdt[MAX_GDT_ENT_NR] ;
TGdtr48		g_gdtr48;
Selector	g_kernel_code_selector;
Selector	g_kernel_data_selector;
Selector	g_kernel_stack_selector;
Selector	g_kernel_gs_selector;
Selector	g_idt_selector; /*���н��̹���ͬһ��idtѡ����*/
Selector	g_tss_selector; /*���н��̹���ͬһ��tssѡ����*/

/* IDT ȫ�ֱ���*/
TGate		g_idt[255];
TIdtr48		g_idtr48;

/*����ȫ�ֱ���*/
TProcess    g_procs[MAX_PROC_NR];
TProcess 	* g_current = NULL;
TSS			g_tss;

/*ϵͳ����*/
uint32_t	g_syscall_param_eax;
uint32_t	g_syscall_param_ebx;
uint32_t	g_syscall_param_ecx;
uint32_t	g_syscall_param_edx;
uint64_t	g_ticks = 0;

/*Ӳ�̲���*/
THdParam	g_hd_param;
void (*do_hd)()	= NULL;
