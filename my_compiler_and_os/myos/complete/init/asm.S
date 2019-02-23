#include "const_def.h"
.text
/*设置分页表*/
.globl _setup_paging
.type _setup_paging, @function
_setup_paging: 
        pushl %ebp
        movl %esp,%ebp

		SAVE_REG

        #设置页目录
        movl $1024,%ecx
        mov  $4096 + PG_P + PG_USU + PG_RWW, %eax
        movl $0,%edi

SetupPaging.loop: 
        movl %eax,%es:(%edi)
        addl $4096,%eax
        addl $4,%edi
        loop SetupPaging.loop


        #设置4个页表
        movl $(1024*4),%ecx
		movl $(PG_P + PG_USU + PG_RWW), %eax
        movl $0,%edi
SetupPaging.loop2: 
        movl %eax,(%edi)
        addl $4096,%eax
        addl $4,%edi

        loop SetupPaging.loop2

		xorl	%eax, %eax
        movl %eax, %cr3
        movl %cr0, %eax
        orl $0x80000000,%eax
        movl %eax, %cr0

		RESTORE_REG

        movl %ebp,%esp
        popl %ebp
        ret
.globl out_byte
.type   out_byte, @function /* void out_byte(int port, int value);*/
out_byte:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	#define port 8(%ebp)
	#define value 12(%ebp)

	movl port , %edx
	movl value, %eax
	out  %al, %dx
	nop
	nop
	nop

	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret

	#undef  port
	#undef  value

.globl out_word
.type   out_word, @function /* void out_word(int port, int value);*/
out_word:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	#define port 8(%ebp)
	#define value 12(%ebp)

	movl port , %edx
	movl value, %eax
	outw  %ax, %dx
	nop
	nop
	nop

	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret

	#undef  port
	#undef  value

.globl in_byte
.type   in_byte, @function /* uint32_t in_byte(int port);*/
in_byte:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	#define port 8(%ebp)

	movl port , %edx
	xorl	%eax, %eax
	in %dx, %al
	nop
	nop
	nop
	
	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret
	#undef  port

.globl in_word
.type   in_word, @function /* uint32_t in_word(int port);*/
in_word:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	#define port 8(%ebp)

	movl port , %edx
	xorl	%eax, %eax
	inw %dx, %ax
	nop
	nop
	nop

	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret
	#undef  port

/*_show_flash 在屏幕固定位置显示一个转动的小动画*/
.type _show_flash, @function
_show_flash: 

	SAVE_REG

	/*保存次数*/
	movl intr_count, %eax
	incl %eax
	movl %eax, intr_count

	shrl $3, %eax        /*使动画看起来慢一点*/
	andl $3,%eax  /*对4求余*/

	movl %eax,%edx

	cmpl $0,%edx
	je      _IntrHandler.l0

	decl %edx
	cmpl $0,%edx
	je      _IntrHandler.l1

	decl %edx
	cmpl $0,%edx
	je      _IntrHandler.l2

	decl %edx
	cmpl $0,%edx
	je      _IntrHandler.l3

_IntrHandler.l0: 
	movb $'-', %al
	jmp _IntrHandler.end

_IntrHandler.l1: 
	movb $'\\', %al
	jmp _IntrHandler.end
_IntrHandler.l2: 
	movb $'|', %al
	jmp _IntrHandler.end
_IntrHandler.l3: 
	movb $'/', %al
	jmp _IntrHandler.end
_IntrHandler.end: 
	movb $0xc,%ah                    /*显示的颜色*/
	movl $(80*13+3)*2, %edi
	movw %ax,GS_START(%edi)          /*显示*/
	
	RESTORE_REG

	ret
.globl _nop
.type _nop, @function
_nop:
	nop
	nop
	nop
	ret


func_io_delay: 
	nop
	nop
	nop
	nop
	ret
.globl _lidt
.type _lidt, @function
_lidt:
	cli
	lidt g_idtr48
	ret

.text
.globl _EH_divide_error		/* 0 */
.globl _EH_debug_error		/* 1 */
.globl _EH_not_mask_intr	/* 2 */
.globl _EH_debug_break		/* 3 */
.globl _EH_over_flow		/* 4 */
.globl _EH_break_limit		/* 5 */
.globl _EH_undefined_op		/* 6 */
.globl _EH_no_coproc		/* 7 */
.globl _EH_double_error
.globl _EH_coproc_break_limit
.globl _EH_invalid_tss		/* 10 */
.globl _EH_no_seg
.globl _EH_stack_error
.globl _EH_general_protect_error	/* 13 */
.globl _EH_page_error
.globl _EH_reserve15		/* 15 */
.globl _EH_float_error
.globl _EH_align_check
.globl _EH_machine_check
.globl _EH_simd_float_error	/* 19 */


_EH_divide_error:
    pushl $0xffffffff
    pushl $0
    jmp ._EH_exception_call
_EH_debug_error:
    pushl $0xffffffff
    pushl $1
    jmp ._EH_exception_call
_EH_not_mask_intr:
    pushl $0xffffffff
    pushl $2
    jmp ._EH_exception_call
_EH_debug_break:
    pushl $0xffffffff
    pushl $3
    jmp ._EH_exception_call
_EH_over_flow:
    pushl $0xffffffff
    pushl $4
    jmp ._EH_exception_call
_EH_break_limit:
    pushl $0xffffffff
    pushl $5
    jmp ._EH_exception_call
_EH_undefined_op:
    pushl $0xffffffff
    pushl $6
    jmp ._EH_exception_call
_EH_no_coproc:
    pushl $0xffffffff
    pushl $7
    jmp ._EH_exception_call
_EH_double_error:
    pushl $8
    jmp ._EH_exception_call
_EH_coproc_break_limit:
    pushl $0xffffffff
    pushl $9
    jmp ._EH_exception_call
_EH_invalid_tss:
    pushl $10
    jmp ._EH_exception_call
_EH_no_seg:
    pushl $11
    jmp ._EH_exception_call
_EH_stack_error:
    pushl $12
    jmp ._EH_exception_call
_EH_general_protect_error:
    pushl $13
    jmp ._EH_exception_call
_EH_page_error:
    pushl $14
    jmp ._EH_exception_call
_EH_reserve15:
    pushl $0xffffffff
    pushl $15
    jmp ._EH_exception_call
_EH_float_error:
    pushl $0xffffffff
    pushl $16
    jmp ._EH_exception_call
_EH_align_check:
    pushl $17
    jmp ._EH_exception_call
_EH_machine_check:
    pushl $0xffffffff
    pushl $18
    jmp ._EH_exception_call
_EH_simd_float_error:
    pushl $0xffffffff
    pushl $19
    jmp ._EH_exception_call

._EH_exception_call:
	call handle_exception
	addl $8, %esp	

	jmp .		/*直接宕机了*/
	/*iretl*/

/**
 * 在执行INT指令时，实际完成了以下几条操作：
 * （1） 由于INT指令发生了不同优先级之间的控制转移，所以首先从TSS（任务状态段）中获取高优先级的核心堆栈信息（SS和ESP）；
 * （2） 把低优先级堆栈信息（SS和ESP）保留到高优先级堆栈（即核心栈）中；
 * （3） 把EFLAGS，外层CS，EIP推入高优先级堆栈（核心栈）中。
 * （4） 通过IDT加载CS，EIP（控制转移至中断处理函数）
*/

.globl _IH_irq00
.globl _IH_irq01
.globl _IH_irq02
.globl _IH_irq03
.globl _IH_irq04
.globl _IH_irq05
.globl _IH_irq06
.globl _IH_irq07
.globl _IH_irq08
.globl _IH_irq09
.globl _IH_irq10
.globl _IH_irq11
.globl _IH_irq12
.globl _IH_irq13
.globl _IH_irq14
.globl _IH_irq15
.globl _IH_sys_call

	/**
	 * 进程被中断后，此时esp/ss根据tss里的设置指向TProcess.regs
	 * 刚好把进程的上下文压入这个字段，用于后续恢复用
	 */
#define SAVE_PROC_CTX \
	subl $4, %esp ;\
	pushal;\
	pushl %ds;\
	pushl %es;\
	pushl %fs;\
	pushl %gs;\

	/**
	 * 进程被中断、保存进程上下文后
	 * 修改 esp/ss和其他各段寄存器
	 * 恢复内核运行的环境
	 */
#define RESTORE_KNL_CTX \
	movl $(KERNEL_ORG-1), %eax;\
	andl $0xfffffffc, %eax;\
	movl %eax, %esp;\
	movb $1, use_kernel_stack ;\
	movw g_kernel_data_selector, %ax;\
	mov %ax, %ds;\
	mov %ax, %es;\
	mov %ax, %fs;\
	mov %ax, %gs;\

#define EOI_MST \
	movb $0x20,%al; \
	out %al, $INT_M_CTL; \

#define EOI_SLV \
	movb $0x20,%al; \
	out %al, $INT_M_CTL; \
	out %al, $INT_S_CTL; \


/*************************************************************/
/*系统调用中断的处理*/
_IH_sys_call:
	
	SAVE_PROC_CTX

	/*保存系统调用的参数*/
	movl %eax, g_syscall_param_eax
	movl %ebx, g_syscall_param_ebx
	movl %ecx, g_syscall_param_ecx
	movl %edx, g_syscall_param_edx


	/*切换为内核堆栈*/
	RESTORE_KNL_CTX

	pushl g_syscall_param_edx
	pushl g_syscall_param_ecx
	pushl g_syscall_param_ebx
	pushl g_syscall_param_eax
	call execute_sys_call
	addl $16, %esp
	/*系统调用的返回值在%eax,已被保存到TProcess结构里*/

	/*重新调度进程*/
	call schedule
	cmpl $0, %eax
	je  _IH_sys_call.l1
	sti
	hlt

_IH_sys_call.l1:
	call _move_to_process	

/**
 * 因为中断可能发生在程序执行的任何位置
 * 为了防止中断处理过程改变原程序中的寄存器的值
 * 处理前保存所有的通用寄存器，处理后恢复
 */
/************************************************************************/
/*时钟中断处理，完成进程的调度*/
_IH_irq00:
	cmpb $0, use_kernel_stack
	jne  _IH_irq00.l2	/*已经是内核堆栈，就跳过保存/切换的过程*/

	SAVE_PROC_CTX

_IH_irq00.l2:
	/*切换为内核堆栈*/
	RESTORE_KNL_CTX

	/*进行 时钟处理*/
	call _show_flash

	call handle_timer_interrupt

	/*重新调度进程*/
	call schedule
	cmpl $0, %eax
	je  _IH_irq00.l1
	sti
	EOI_MST
	hlt

_IH_irq00.l1:
	EOI_MST
	call _move_to_process	
/********************************************************************************/
/* keyboard intr */
_IH_irq01:
	cmpb $0, use_kernel_stack
	jne  _IH_irq01.l2	/*已经是内核堆栈，就跳过保存/切换的过程*/

	SAVE_PROC_CTX

_IH_irq01.l2:
	/*切换为内核堆栈*/
	RESTORE_KNL_CTX

	call  keyboard_intr_handle
	/*重新调度进程*/
	call schedule
	cmpl $0, %eax
	je _IH_irq01.l1
	EOI_MST
	sti
	hlt


_IH_irq01.l1:
	EOI_MST
	call _move_to_process	
/*********************************************************/	
_IH_irq02:
	pushl $2
	jmp ._IH_intr_call
_IH_irq03:
	pushl $3
	jmp ._IH_intr_call
_IH_irq04:
	pushl $4
	jmp ._IH_intr_call
_IH_irq05:
	pushl $5
	jmp ._IH_intr_call
_IH_irq06:
	pushl $6
	jmp ._IH_intr_call
_IH_irq07:
	pushl $7
	jmp ._IH_intr_call
_IH_irq08:
	pushl $8
	jmp ._IH_intr_call
_IH_irq09:
	pushl $9
	jmp ._IH_intr_call
_IH_irq10:
	pushl $10
	jmp ._IH_intr_call
_IH_irq11:
	pushl $11
	jmp ._IH_intr_call
_IH_irq12:
	pushl $12
	jmp ._IH_intr_call
_IH_irq13:
	pushl $13
	jmp ._IH_intr_call
/*******************************************************************/
/* 硬盘中断 */
_IH_irq14:
	cmpb $0, use_kernel_stack
	jne  _IH_irq14.l2	/*已经是内核堆栈，就跳过保存/切换的过程*/

	SAVE_PROC_CTX

_IH_irq14.l2:
	RESTORE_KNL_CTX

	/*进行处理...*/
	movl do_hd, %eax
	cmpl $0, %eax
	je  _IH_irq14.do_non
	call *%eax

_IH_irq14.do_non:

	/*重新调度进程*/
	call schedule
	cmpl $0, %eax
	je  _IH_irq14.l1
	sti
	EOI_SLV
	hlt

_IH_irq14.l1:
	EOI_SLV
	call _move_to_process	
/*******************************************************************/
_IH_irq15:
	pushl $15
	jmp ._IH_intr_call

._IH_intr_call:
	addl $4, %esp
	iretl


.globl _move_to_process
.type _move_to_process, @function
_move_to_process:
    movl g_current, %esp
	movb $0, use_kernel_stack

	movw g_idt_selector, %ax
    lldt    %ax


	movw g_tss_selector, %ax
	ltr	 %ax
	
	popl %eax
	movw %ax, %gs

	popl %eax
    movw %ax,  %fs

	popl %eax
    movw %ax, %es

	popl %eax
    movw %ax, %ds

	popal

	add $4, %esp	/* 跳过 retaddr */
    iretl


.data
	intr_count:	.long 0
	use_kernel_stack: .byte 0

