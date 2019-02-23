#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED


#include "struct.h"
#include <stdarg.h>
#include "const_def.h"

extern TDescriptor		g_gdt[MAX_GDT_ENT_NR];		/* GDT */
extern TGdtr48     g_gdtr48;
extern Selector    g_kernel_code_selector;
extern Selector    g_kernel_data_selector;
extern Selector    g_kernel_stack_selector;
extern Selector    g_kernel_gs_selector;
extern Selector    g_idt_selector; 
extern Selector    g_tss_selector; 

extern uint8_t     g_scr_line      ;
extern uint8_t     g_scr_colume    ;

extern TProcess    g_procs[MAX_PROC_NR];	/*进程表*/
extern TProcess    * g_current ;
extern TSS         g_tss; 

extern TGate       g_idt[MAX_IDT_ENT_NR];
extern TIdtr48     g_idtr48;

extern uint32_t    g_syscall_param_eax; 	/*系统调用参数保存变量*/
extern uint32_t    g_syscall_param_ebx;
extern uint32_t    g_syscall_param_ecx;
extern uint32_t    g_syscall_param_edx;
extern uint64_t    g_ticks;			/*时钟中断累计次数*/
extern uint32_t		g_syscall_nr; /*系统调用号最大值*/
extern uint32_t 	g_syscall_entry[]; /*系统调用函数地址列表 注意和 uint32_t * g_syscall_entry的区别 !*/
extern THdParam		g_hd_param;
extern void (*do_hd)();



/**
 * 将进程proc的地址offset转化为内核空间的虚拟地址,保存在paddr
 * 成功返回0， 失败返回-1
 */
int user_addr_to_virtual_addr(TProcess * proc, uint32_t offset, uint32_t * paddr);
void init_descriptor(TDescriptor* pdes, uint32_t base, uint32_t limit, uint16_t attr);
void init_gate(TGate* pgate, Selector s, uint32_t offset, uint8_t reserve, uint8_t attr);
void handle_exception(uint32_t vecno, uint32_t errno, uint32_t eip, uint32_t cs, uint32_t eflags);
void handle_timer_interrupt();
void _setup_paging();
uint32_t       execute_sys_call(uint32_t eax,
		uint32_t ebx,
		uint32_t ecx,
		uint32_t edx);

void memset(void * p, uint8_t v, uint32_t sz);
void memcpy(void * dst, const void * src, uint32_t sz);
int memcmp(const void * a, const void * b, size_t sz);
void print_str(const char * str);
int int_to_hex(uint32_t v, char * buf, size_t bufsz);
void print_hex(uint32_t v);
void print_chr(char c);
uint32_t in_byte(int port);
uint32_t in_word(int port);
void out_byte(int port, int value);
void out_word(int port, int value);
void _lidt();
/*void _send_end_of_intr(uint32_t is_from_slave);*/
uint32_t _sleep(uint32_t v);
uint32_t select2index(Selector sel);
int schedule();
void _move_to_process();

/*供应用程序使用的两个系统调用*/


void _EH_divide_error();
void _EH_debug_error();
void _EH_not_mask_intr();
void _EH_debug_break();
void _EH_over_flow();
void _EH_break_limit();
void _EH_undefined_op();
void _EH_no_coproc();
void _EH_double_error();
void _EH_coproc_break_limit();
void _EH_invalid_tss();
void _EH_no_seg();
void _EH_stack_error();
void _EH_general_protect_error();
void _EH_page_error();
void _EH_reserve15();
void _EH_float_error();
void _EH_align_check();
void _EH_machine_check();
void _EH_simd_float_error();

void  _IH_irq00();
void  _IH_irq01();
void  _IH_irq02();
void  _IH_irq03();
void  _IH_irq04();
void  _IH_irq05();
void  _IH_irq06();
void  _IH_irq07();
void  _IH_irq08();
void  _IH_irq09();
void  _IH_irq10();
void  _IH_irq11();
void  _IH_irq12();
void  _IH_irq13();
void  _IH_irq14();
void  _IH_irq15();
void  _IH_sys_call();

size_t strlen(const char *s);
void panic(const char * fmt, ...);
void do_task();
void task_set_busy(uint64_t busy);
void task_is_busy(uint64_t * pflag);
		               

void keyboard_intr_handle();
int keyboard_do_task();

int vsprintf(char *buf, size_t bufsz, const char *fmt, va_list args);
int snprintf(char * buf, size_t bufsz, const char *fmt, ...);
int printk(const char * fmt, ...);
void _nop(void);

#define _sti()		({__asm("sti"); 0;})
#define _cli()		({__asm("cli"); 0;})
#endif
