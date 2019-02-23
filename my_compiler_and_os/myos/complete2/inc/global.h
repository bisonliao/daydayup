#ifndef GLOBAL_H_INCLUDED
#define GLOBAL_H_INCLUDED


#include "struct.h"
#include <stdarg.h>
#include "const_def.h"

extern TDescriptor		g_gdt[MAX_GDT_ENT_NR];		/* GDT */
extern TGdtr48     g_gdtr48;

extern volatile uint8_t     g_scr_line      ;
extern volatile uint8_t     g_scr_colume    ;

extern TProcess    g_procs[MAX_PROC_NR];	/*进程表*/
extern volatile TProcess    * g_current ;

extern TGate       g_idt[MAX_IDT_ENT_NR];
extern TIdtr48     g_idtr48;

extern volatile uint32_t    g_syscall_param_eax; 	/*系统调用参数保存变量*/
extern volatile uint32_t    g_syscall_param_ebx;
extern volatile uint32_t    g_syscall_param_ecx;
extern volatile uint32_t    g_syscall_param_edx;
extern volatile uint32_t    g_syscall_param_ds;
extern uint32_t		g_syscall_nr; /*系统调用号最大值*/
extern uint32_t 	g_syscall_entry[]; /*系统调用函数地址列表 注意和 uint32_t * g_syscall_entry的区别 !*/
extern THdParam		g_hd_param;
extern volatile void (*do_hd)();

extern unsigned char g_krnl_stack[MAX_PROC_NR][KRNL_STACK_SZ];
extern volatile uint32_t g_startup_time;
extern volatile uint32_t g_uptime;
extern volatile uint64_t    g_ticks;			/*时钟中断累计次数*/
extern struct file 	g_file_table[MAX_FILE_TABLE];
extern volatile uint8_t g_hd_sync_flag;



/**
 * 将进程proc的地址offset转化为内核空间的虚拟地址,保存在paddr
 * 成功返回0， 失败返回-1
 */
int user_space_vaddr_to_paddr(TProcess * proc, uint32_t offset, uint32_t * paddr, uint32_t ds);
void init_descriptor(TDescriptor* pdes, uint32_t base, uint32_t limit, uint16_t attr);
void init_gate(TGate* pgate, Selector s, uint32_t offset, uint8_t reserve, uint8_t attr);
void handle_exception(uint32_t vecno, uint32_t errno, uint32_t eip, uint32_t cs, uint32_t eflags);
void handle_timer_interrupt();
void _setup_paging();
uint32_t       execute_sys_call(uint32_t eax,
		uint32_t ebx,
		uint32_t ecx,
		uint32_t edx,
		uint32_t ds);

void print_str(const char * str); /*内核态适用*/
int int_to_hex(uint32_t v, char * buf, size_t bufsz); /*用户态和内核态适用*/
void print_hex(uint32_t v); /*内核态适用*/
void print_chr(char c); /*内核态适用*/
uint32_t in_byte(int port); /*内核态适用*/
uint32_t in_word(int port);  /*内核态适用*/
void out_byte(int port, int value);  /*内核态适用*/
void out_word(int port, int value); /*内核态适用*/
void _lidt();
/*void _send_end_of_intr(uint32_t is_from_slave);*/
uint32_t _sleep(uint32_t v);
uint32_t select2index(Selector sel);
int schedule();
void _move_to_process();
int load_proc_from_fs(const char * pathname, int pid);



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

void panic(const char * fmt, ...);
void do_task();
void task_set_busy(uint64_t busy);
void task_is_busy(uint64_t * pflag);
		               

void keyboard_intr_handle();
void keyboard_init();
int keyboard_do_task();

int printk(const char * fmt, ...);
void _nop(void);
void div_uint64(uint64_t a, uint64_t b, uint64_t * res);
void sleep_on(TProcess **p);
void wake_up(TProcess **p);
void _switch(TProcess* next, TProcess* prev, uint32_t tss_selector);
uint32_t _get_cr3();
uint32_t _set_cr3(uint32_t);
uint32_t _get_esp();
void _copy_krnl_stack(void * _dst, void * _src,
        uint32_t * _eip, uint32_t * _esp,
		uint32_t * ebx, uint32_t * ecx, uint32_t * edx, uint32_t * esi, uint32_t * edi,
		uint32_t * ebp, uint32_t * eflags, int i_distance);

#define _sti()		({__asm("sti"); 0;})
#define _cli()		({__asm("cli"); 0;})

#define GET_KRNL_STACK_START(pid) 	(&g_krnl_stack[(pid)][0] + KRNL_STACK_SZ)
#define GET_USER_STACK_START(pid) 	((FIRST_PROC_ORG+ (pid)*PROC_SPACE-1) & 0xfffffffc)

#endif
