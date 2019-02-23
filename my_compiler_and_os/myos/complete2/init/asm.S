#include "const_def.h"
#include "offset.h"
.text
/*设置分页表*/
/*******************************************************
 通过查找分页表，完成线性地址到物理地址的映射
 32位线性地址分为三部分，
   22-31位是在页目录里的下标，查找得到页表的物理地址，每个页目录有1024个表项，共4096字节大小
   12-21位势在页表里的下标, 查找得到页的物理地址， 每个页表有1024个表项，共4096字节大小
   0-11位是在页里的偏移, 最后得到在页里的物理地址，每个页4096字节大小

 下面的程序在物理地址[0,4096)设置页目录，只用开头的4个表项
 [4096,8192)用于第一个页表， 管理0-4M的物理内存
 [8192,4096*2)用于第二个页表，管理4M-8M的物理内存
 依次类推，总共管理16M物理内存
 经过该分页表转换的线性地址和物理地址在数值上是相等的。
*******************************************************/
.globl _setup_paging
.type _setup_paging, @function
_setup_paging: 
        pushl %ebp
        movl %esp,%ebp

		SAVE_REG

        /*设置页目录*/
        movl $4,%ecx  /*4个页目录表项*/
        movl  $(4096 + PG_P + PG_USU + PG_RWW), %eax  /*第一个页表的物理地址和属性*/
        movl $0,%edi        /*页目录的起始地址*/

SetupPaging.loop: 
        movl %eax,%es:(%edi)
        addl $4096,%eax /*下一个页表的物理地址，属性同第一个页表*/
        addl $4,%edi    /*下一个页目录项*/

		decl %ecx
		cmpl $0, %ecx
        jne SetupPaging.loop


        /*设置4个页表*/
        movl $(1024*4),%ecx /*页表表项的个数*/
		movl $(PG_P + PG_USU + PG_RWW), %eax    /*第一个页的物理地址(0)和属性*/
        movl $4096,%edi /*第一个页表表项的物理地址*/
SetupPaging.loop2: 
        movl %eax,%es:(%edi)    
        addl $4096,%eax /*下一个页的物理地址，属性同第一个页的物理地址*/
        addl $4,%edi    /*下一个页表表项*/

		decl %ecx
		cmpl $0, %ecx
        jne SetupPaging.loop2


		xorl	%eax, %eax
        movl %eax, %cr3  /* cr3控制寄存器指向页目录的物理地址, 0! */


        
.globl debug_label
debug_label:
        /***************************************
         * CR0的详细情况见intel_sysmtem_programming_guide.pdf的2.5章节：控制寄存器
         ***************************************/ 
        movl %cr0, %eax
        orl $0x80000000,%eax /*PG置1，以启动分页机制*/
        orl $0x3, %eax  /* PE MP置1 */
        andl $0xffffffdb, %eax  /*清 EM NE标志，有fpu硬件  EM=0 MP=1 NE=0*/
        movl %eax, %cr0 

        finit /*初始化fpu,使得可以进行浮点运算，如果硬件没有fpu，操作系统可以模拟出一个*/

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

	shrl $2, %eax        /*使动画看起来慢一点*/
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
	movl $(80*0+79)*2, %edi          /*第1行的第1列的位置*/
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


.globl _lidt
.type _lidt, @function
_lidt:
	cli
	lidt g_idtr48
	ret

/**
 * 各种异常各自的处理进入点
 * 最后都是调用handle_exception函数
 */
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
 *各种中断的服务程序进入点
 * 在执行INT指令时，实际完成了以下几条操作：
 * （1） 由于INT指令发生了不同优先级之间的控制转移，所以首先从TSS（任务状态段）中获取高优先级的核心堆栈信息（SS和ESP）；
 * （2） 把低优先级堆栈信息（SS和ESP）保留到高优先级堆栈（即核心栈）中；如果没有发生优先级切换，这一步没有
 * （3） 把EFLAGS，外层CS，EIP推入高优先级堆栈（核心栈）中。
 * （4） 通过IDT加载CS，EIP（控制转移至中断处理函数）, 所有中断向量表项里有这两项
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

#define SAVE_PROC_CTX \
	subl $4, %esp ;\
	pushal;\
	pushl %ds;\
	pushl %es;\
	pushl %fs;\
	pushl %gs;\

#define RESTORE_KNL_CTX

#define RESTORE_PROC_CTX \
	popl %eax;\
	movw %ax, %gs;\
	popl %eax;\
    movw %ax,  %fs;\
	popl %eax;\
    movw %ax, %es;\
	popl %eax;\
    movw %ax, %ds;\
	popal;\
	add $4, %esp	;\

#define EOI_MST \
	movb $0x20,%al; \
	out %al, $INT_M_CTL; \

#define EOI_SLV \
	movb $0x20,%al; \
	out %al, $INT_M_CTL; \
	out %al, $INT_S_CTL; \

.globl  _move_to_process
.type _move_to_process, @function
_move_to_process:
	jmp .

/*************************************************************/
/*系统调用中断的处理*/
_IH_sys_call:

    /*exec由于其特殊性，另外处理*/
	cmpl $SYSCALL_EXEC, %eax
	je _sys_call_exec
	
    /*fork由于其特殊性，另外处理*/
	cmpl $SYSCALL_FORK, %eax
	je _sys_call_fork
	
    /*exit由于其特殊性，另外处理*/
	cmpl $SYSCALL_EXIT, %eax
	je _sys_call_exit
	
	SAVE_PROC_CTX

	pushl %ebp
	movl %esp, %ebp


	pushl %eax /*保存eax, 因为这个时候eax/ebx/ecx/edx里有系统调用的参数*/

	/*
     *这个时候数据段的选择子和寄存器还是用户进程的
     * 只有代码段自动完成了切换
     * 保存用户级别的ds 到全局变量, 因为系统调用的处理过程中一般都需要访问用户进程的空间
     */
	xorl %eax, %eax
	movw %ds, %ax
	movl %eax, g_syscall_param_ds

	/*恢复内核态各选择子*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	popl %eax /*恢复eax*/
	/*保存系统调用的参数到全局变量*/
	movl %eax, g_syscall_param_eax
	movl %ebx, g_syscall_param_ebx
	movl %ecx, g_syscall_param_ecx
	movl %edx, g_syscall_param_edx


	pushl g_syscall_param_ds
	pushl g_syscall_param_edx
	pushl g_syscall_param_ecx
	pushl g_syscall_param_ebx
	pushl g_syscall_param_eax
	call execute_sys_call
	addl $20, %esp
	/*系统调用的返回值在%eax, 直接修改保存好的上下文里的eax*/
	movl %eax, 48(%ebp)


	/*重新调度进程*/
	call schedule

	movl %ebp, %esp
	popl %ebp

	RESTORE_PROC_CTX
	iretl


/*exit由于其特殊性，另外处理*/
_sys_call_exit:
	SAVE_PROC_CTX

	pushl %ebp
	movl %esp, %ebp

	pushl %eax /*保存eax, 因为这个时候eax/ebx/ecx/edx里有系统调用的参数*/

	/*
     *这个时候数据段的选择子和寄存器还是用户进程的
     * 只有代码段自动完成了切换
     * 保存用户级别的ds 到全局变量, 因为系统调用的处理过程中一般都需要访问用户进程的空间
     */
	xorl %eax, %eax
	movw %ds, %ax
	movl %eax, g_syscall_param_ds

	/*恢复内核态各选择子*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	call sys_exit

	/*重新调度进程*/
	call schedule
    /*这里永远不会到达，因为这个进程消亡了，不会被调度*/

	movl %ebp, %esp
	popl %ebp

	RESTORE_PROC_CTX
	iretl

    

/*fork由于其特殊性，另外处理*/
_sys_call_fork:

	SAVE_PROC_CTX /*esp因此向前移动52字节*/

	/*恢复内核态各选择子*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	movl %esp, %ebp

	subl $100, %esp /*留出 100字节作局部变量*/

	#define _eip -4(%ebp)
	#define _cs -8(%ebp)
	#define _eflags -12(%ebp)
	#define _esp -16(%ebp)
	#define _ss -20(%ebp)
	#define _ds -24(%ebp)
	#define _edi -28(%ebp)
	#define _esi -32(%ebp)
	#define _ebp -36(%ebp)
	#define _ebx -40(%ebp)
	#define _edx -44(%ebp)
	#define _ecx -48(%ebp)
	#define _eax -52(%ebp)

	/*把栈中保存的上下文取出来，作为参数传递给sys_fork*/
	movl 52(%ebp), %eax; movl %eax, _eip
	movl 56(%ebp), %eax; movl %eax, _cs
	movl 60(%ebp), %eax; movl %eax, _eflags
	movl 64(%ebp), %eax; movl %eax, _esp
	movl 68(%ebp), %eax; movl %eax, _ss

	movl 12(%ebp), %eax; movl %eax, _ds
	/*pushal = Push EAX, ECX, EDX, EBX, original ESP, EBP, ESI, and EDI.*/
	movl 16(%ebp), %eax; movl %eax, _edi
	movl 20(%ebp), %eax; movl %eax, _esi
	movl 24(%ebp), %eax; movl %eax, _ebp
	movl 32(%ebp), %eax; movl %eax, _ebx
	movl 36(%ebp), %eax; movl %eax, _edx
	movl 40(%ebp), %eax; movl %eax, _ecx
	movl 44(%ebp), %eax; movl %eax, _eax


	pushl _eax
	pushl _ebx
	pushl _ecx
	pushl _edx
	pushl _ebp
	pushl _esi
	pushl _edi

	pushl _ss
	pushl _esp
	pushl _eflags
	pushl _cs
	pushl _eip
	call sys_fork
	addl $48, %esp   /*修改参数后，记得也要修改这里的常数*/

	/*系统调用的返回值在%eax, 直接修改保存好的上下文里的eax*/
    movl %eax, 44(%ebp) /* 44可要数好了*/

	#undef _eip 
	#undef _cs 
	#undef _eflags 
	#undef _esp 
	#undef _ss 
	#undef _ds 
	#undef _edi 
	#undef _esi 
	#undef _ebp 
	#undef _ebx 
	#undef _edx 
	#undef _ecx 
	#undef _eax 

	addl $100, %esp 

	movl %ebp, %esp

	 RESTORE_PROC_CTX

	 iretl

/*exec由于其特殊性，另外处理*/
_sys_call_exec:

	SAVE_PROC_CTX /*esp因此向前移动52字节*/

	/*恢复内核态各选择子*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	movl %esp, %ebp

	subl $100, %esp /*留出 100字节作局部变量*/

	#define _eip -4(%ebp)
	#define _cs -8(%ebp)
	#define _eflags -12(%ebp)
	#define _esp -16(%ebp)
	#define _ss -20(%ebp)
	#define _ds -24(%ebp)
	#define _edi -28(%ebp)
	#define _esi -32(%ebp)
	#define _ebp -36(%ebp)
	#define _ebx -40(%ebp)
	#define _edx -44(%ebp)
	#define _ecx -48(%ebp)
	#define _eax -52(%ebp)

	/*把栈中保存的上下文取出来，作为参数传递给sys_exec*/
	movl 52(%ebp), %eax; movl %eax, _eip
	movl 56(%ebp), %eax; movl %eax, _cs
	movl 60(%ebp), %eax; movl %eax, _eflags
	movl 64(%ebp), %eax; movl %eax, _esp
	movl 68(%ebp), %eax; movl %eax, _ss
	movl 12(%ebp), %eax; movl %eax, _ds
	/*pushal = Push EAX, ECX, EDX, EBX, original ESP, EBP, ESI, and EDI.*/
	movl 16(%ebp), %eax; movl %eax, _edi
	movl 20(%ebp), %eax; movl %eax, _esi
	movl 24(%ebp), %eax; movl %eax, _ebp
	movl 32(%ebp), %eax; movl %eax, _ebx
	movl 36(%ebp), %eax; movl %eax, _edx
	movl 40(%ebp), %eax; movl %eax, _ecx
	movl 44(%ebp), %eax; movl %eax, _eax
/*
   int sys_exec(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds,
        uint32_t * eip_for_userspace,
        uint32_t * esp_for_userspace);
*/
    leal    64(%ebp), %eax /*系统调用进入的时候，esp保存在栈上的地址*/
    pushl %eax

    leal    52(%ebp), %eax /*系统调用进入的时候，eip保存在栈上的地址*/
    pushl %eax

	pushl _ds
	pushl _edx
	pushl _ecx
	pushl _ebx
	pushl _eax
	call sys_exec
	addl $28, %esp   /*修改参数后，记得也要修改这里的常数*/

	/*系统调用的返回值在%eax, 直接修改保存好的上下文里的eax*/
    movl %eax, 44(%ebp) /* 44可要数好了*/

	#undef _eip 
	#undef _cs 
	#undef _eflags 
	#undef _esp 
	#undef _ss 
	#undef _ds 
	#undef _edi 
	#undef _esi 
	#undef _ebp 
	#undef _ebx 
	#undef _edx 
	#undef _ecx 
	#undef _eax 

	addl $100, %esp 

	movl %ebp, %esp

	 RESTORE_PROC_CTX

	 iretl

/**
 * 因为中断可能发生在程序执行的任何位置
 * 为了防止中断处理过程改变原程序中的寄存器的值
 * 处理前保存所有的通用寄存器，处理后恢复
 */
/************************************************************************/
/*时钟中断处理，完成进程的调度*/
_IH_irq00:
	SAVE_PROC_CTX

	/*恢复内核态各选择子, cs eip自动从中断向量表里恢复*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	/*显示时钟小动画*/
	call _show_flash
	call handle_timer_interrupt;

	EOI_MST
	call schedule

	RESTORE_PROC_CTX
	iretl
/********************************************************************************/
/* keyboard intr */
_IH_irq01:
	SAVE_PROC_CTX

	/*恢复内核态各选择子*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	call keyboard_intr_handle

	EOI_MST

	RESTORE_PROC_CTX
	iretl

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
    call NewFunction
    iretl
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
	SAVE_PROC_CTX


	/*恢复内核态各选择子*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	/*进行处理... do_hd是一个函数指针，如果不为0， 调用它*/
	movl do_hd, %eax
	cmpl $0, %eax
	je  _IH_irq14.do_non
	call *%eax

_IH_irq14.do_non:
	EOI_SLV

	RESTORE_PROC_CTX
	iretl
/*******************************************************************/
_IH_irq15:
	pushl $15
	jmp ._IH_intr_call

._IH_intr_call:
	addl $4, %esp
	iretl



.globl _switch
.type _switch, @function
_switch:
	pushl %ebp
	movl %esp, %ebp




#ifdef SWITCH_BY_MYSELF
	SAVE_REG 
	/*
	 *采取 mov指令保存上下文、加载上下文的方式，适用于risc硬件体系
     * 原型： _switch(目标进程控制块指针, 当前进程控制块指针, 目标进程tss选择子)
	 */
	movl 8(%ebp), %ebx	 /*目标进程控制块指针*/
	movl 12(%ebp), %edx  /*当前进程控制块指针*/
	movl 16(%ebp), %eax  /*目标进程tss选择子*/
	ltr %ax
	/* ebx此刻保存了欲运行的进程的TProcess结构的指针*/
	/* edx此刻保存了当前进程的TProcess结构的指针*/
	cmpl $0, %edx
	je .non_ctx_save

	xchgl %edx, %esp     /*直接把当前进程tss结构所在的内存块作为栈，使用push指令来保存当前进程的上下文*/
	addl $(4*25), %esp 	/*从字段tss.ldt开始*/
	xorl %eax, %eax
	sldt %ax
	pushl %eax   /*当前进程的ldt选择子*/
	pushl %gs
	pushl %fs
	pushl %ds
	pushl %ss
	pushl %cs
	pushl %es
	pushl %edi
	pushl %esi
	pushl %ebp
	pushl %edx	/*保存 esp*/
	pushl %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	pushfl
	pushl $.current_continue	/*保存eip, 当前进程被切换回来后就从.current_continue处运行*/
	movl %cr3, %eax  /*保存cr3*/
	pushl %eax
	/*保存上下文完毕*/
	movl %edx, %esp	/*恢复esp*/
.non_ctx_save:
	/*开始加载新的上下文*/
	movl 96(%ebx), %eax  /*加载新进程的ldt*/
	lldt %ax
    movl 28(%ebx), %eax /*加载新进程的页表和页目录*/
    movl %eax, %cr3
	/*如果切换前后运行级别不变，那么iret不会自动恢复ss esp*/
	movl 76(%ebx), %eax
	andl $3, %eax
	cmpl $0, %eax      /*看cs保存的选择子的RPL是否为0 */
	je .level_not_change	/*级别没有发生变化*/

    /*伪造一个中断返回的现场*/
	/*为iret准备 ss esp eflags cs eip*/
	movl 80(%ebx), %eax
	pushl %eax	/* ss */
	movl 56(%ebx), %eax
	pushl %eax	/* esp */
	movl 36(%ebx), %eax
	pushl %eax	/* eflags */
	movl 76(%ebx), %eax
	pushl %eax	/* cs */
	movl 32(%ebx), %eax
	pushl %eax	/* eip */
	jmp .switch_continue_1
.level_not_change:
	/*直接切换到待运行进程的堆栈*/
	movl 80(%ebx), %eax
	movw %ax, %ss	/* ss */
	movl 56(%ebx), %eax
	movl %eax, %esp	/* esp */

	/*为iret准备eflags cs eip*/
	movl 36(%ebx), %eax
	pushl %eax	/* eflags */
	movl 76(%ebx), %eax
	pushl %eax	/* cs */
	movl 32(%ebx), %eax
	pushl %eax	/* eip */
	

.switch_continue_1:
	/*准备恢复ebx*/
	movl 52(%ebx), %eax
	pushl %eax

	/*恢复 es ds fs gs */
	xchgl %ebx, %esp /*直接把欲运行的进程tss结构所在的内存块作为栈，使用popl指令来恢复进程的上下文*/
	addl $(18*4), %esp
	popl %eax
	movw %ax, %es
	addl $8, %esp
	popl %eax
	movw %ax, %ds
	popl %eax
	movw %ax, %fs
	popl %eax
	movw %ax, %gs

	/* 从这里开始，不能访问非栈内存了*/

	/*恢复 eax ecx edx ebp esi edi*/
	subl $56, %esp
	popl %eax
	popl %ecx
	popl %edx
	addl $8, %esp
	popl %ebp
	popl %esi
	popl %edi

	movl %ebx, %esp
	popl %ebx	/*恢复ebx*/


	iretl	/*切换到另外一个进程,  恢复了cs eip eflags ss esp或者eflags cs eip*/
.current_continue:
	RESTORE_REG

#else
	/*
	 * 采取 ljmp tss选择子,偏移的方式切换进程(cpu硬件提供的方法)，当前运行的进程的上下文保存在tr指向的tss段
	 * 然后装载tss选择子指向的tss里保存的上下文，完成进程的切换
     * 原型： _switch(目标进程控制块指针, 当前进程控制块指针, 目标进程tss选择子)
     * 当使用浮点运算的时候，会出现no_coproc的异常，还没有解决
	 */
	movl 8(%ebp), %eax /*目标进程控制块指针 */
    movl pid_OFFSET_IN_TProcess(%eax), %eax  /*目标进程控制块内pid字段的值*/

	pushl %ecx
	pushl %edx

	subl $8, %esp /*在栈上分配一个自动变量*/
    /* (g_first_tss_selector + pid * 8) 就是要跳转的进程的tss段的选择子，写入自动变量*/
	movl $8, %ecx
	mull %ecx		/*EDX:EAX ← EAX * r/m32,    将eax乘8 */
	addl $g_first_tss_selector, %eax
	movw %ax, 4(%esp)
	movl $0, (%esp)
	ljmp (%esp)	/*狗日的，这个跳转来得真不容易*/
	/*其实就是6字节大小的一个内存，低四字节存放偏移,值无所谓，高2字节存放tss的selector*/


	addl $8, %esp
	popl %edx
	popl %ecx
	
#endif

	movl %ebp, %esp
	popl %ebp
	ret

.globl _ltr
.type _ltr, @function
_ltr:
	pushl %ebp
	movl %esp, %ebp

	movl 8(%ebp), %eax
	ltr %ax

	movl %ebp, %esp
	popl %ebp
	ret

.globl _set_cr3
.type _set_cr3, @function
_set_cr3:
    pushl %ebp
    movl %esp, %ebp

	movl 8(%ebp), %eax
    movl %eax, %cr3

    movl %ebp, %esp
    popl %ebp
	ret
.globl _get_cr3
.type _get_cr3, @function
_get_cr3:
	movl %cr3, %eax
	ret
.globl _get_esp
.type _get_esp, @function
_get_esp:
	movl %esp, %eax
	addl $4,  %eax /*跳过ret addr*/
	ret
.globl _get_eip
.type _get_eip, @function
_get_eip:
	xorl %eax, %eax
	movl $., %eax	
	ret

.data
	intr_count:	.long 0
	_seg: .short 0
	_offset:.long 0
