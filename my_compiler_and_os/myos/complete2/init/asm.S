#include "const_def.h"
#include "offset.h"
.text
/*���÷�ҳ��*/
/*******************************************************
 ͨ�����ҷ�ҳ����������Ե�ַ��������ַ��ӳ��
 32λ���Ե�ַ��Ϊ�����֣�
   22-31λ����ҳĿ¼����±꣬���ҵõ�ҳ����������ַ��ÿ��ҳĿ¼��1024�������4096�ֽڴ�С
   12-21λ����ҳ������±�, ���ҵõ�ҳ��������ַ�� ÿ��ҳ����1024�������4096�ֽڴ�С
   0-11λ����ҳ���ƫ��, ���õ���ҳ���������ַ��ÿ��ҳ4096�ֽڴ�С

 ����ĳ�����������ַ[0,4096)����ҳĿ¼��ֻ�ÿ�ͷ��4������
 [4096,8192)���ڵ�һ��ҳ���� ����0-4M�������ڴ�
 [8192,4096*2)���ڵڶ���ҳ��������4M-8M�������ڴ�
 �������ƣ��ܹ�����16M�����ڴ�
 �����÷�ҳ��ת�������Ե�ַ��������ַ����ֵ������ȵġ�
*******************************************************/
.globl _setup_paging
.type _setup_paging, @function
_setup_paging: 
        pushl %ebp
        movl %esp,%ebp

		SAVE_REG

        /*����ҳĿ¼*/
        movl $4,%ecx  /*4��ҳĿ¼����*/
        movl  $(4096 + PG_P + PG_USU + PG_RWW), %eax  /*��һ��ҳ����������ַ������*/
        movl $0,%edi        /*ҳĿ¼����ʼ��ַ*/

SetupPaging.loop: 
        movl %eax,%es:(%edi)
        addl $4096,%eax /*��һ��ҳ����������ַ������ͬ��һ��ҳ��*/
        addl $4,%edi    /*��һ��ҳĿ¼��*/

		decl %ecx
		cmpl $0, %ecx
        jne SetupPaging.loop


        /*����4��ҳ��*/
        movl $(1024*4),%ecx /*ҳ������ĸ���*/
		movl $(PG_P + PG_USU + PG_RWW), %eax    /*��һ��ҳ��������ַ(0)������*/
        movl $4096,%edi /*��һ��ҳ�������������ַ*/
SetupPaging.loop2: 
        movl %eax,%es:(%edi)    
        addl $4096,%eax /*��һ��ҳ��������ַ������ͬ��һ��ҳ��������ַ*/
        addl $4,%edi    /*��һ��ҳ������*/

		decl %ecx
		cmpl $0, %ecx
        jne SetupPaging.loop2


		xorl	%eax, %eax
        movl %eax, %cr3  /* cr3���ƼĴ���ָ��ҳĿ¼��������ַ, 0! */


        
.globl debug_label
debug_label:
        /***************************************
         * CR0����ϸ�����intel_sysmtem_programming_guide.pdf��2.5�½ڣ����ƼĴ���
         ***************************************/ 
        movl %cr0, %eax
        orl $0x80000000,%eax /*PG��1����������ҳ����*/
        orl $0x3, %eax  /* PE MP��1 */
        andl $0xffffffdb, %eax  /*�� EM NE��־����fpuӲ��  EM=0 MP=1 NE=0*/
        movl %eax, %cr0 

        finit /*��ʼ��fpu,ʹ�ÿ��Խ��и������㣬���Ӳ��û��fpu������ϵͳ����ģ���һ��*/

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

/*_show_flash ����Ļ�̶�λ����ʾһ��ת����С����*/
.type _show_flash, @function
_show_flash: 

	SAVE_REG

	/*�������*/
	movl intr_count, %eax
	incl %eax
	movl %eax, intr_count

	shrl $2, %eax        /*ʹ������������һ��*/
	andl $3,%eax  /*��4����*/

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
	movb $0xc,%ah                    /*��ʾ����ɫ*/
	movl $(80*0+79)*2, %edi          /*��1�еĵ�1�е�λ��*/
	movw %ax,GS_START(%edi)          /*��ʾ*/
	
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
 * �����쳣���ԵĴ��������
 * ����ǵ���handle_exception����
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

	jmp .		/*ֱ��崻���*/
	/*iretl*/

/**
 *�����жϵķ����������
 * ��ִ��INTָ��ʱ��ʵ����������¼���������
 * ��1�� ����INTָ����˲�ͬ���ȼ�֮��Ŀ���ת�ƣ��������ȴ�TSS������״̬�Σ��л�ȡ�����ȼ��ĺ��Ķ�ջ��Ϣ��SS��ESP����
 * ��2�� �ѵ����ȼ���ջ��Ϣ��SS��ESP�������������ȼ���ջ��������ջ���У����û�з������ȼ��л�����һ��û��
 * ��3�� ��EFLAGS�����CS��EIP��������ȼ���ջ������ջ���С�
 * ��4�� ͨ��IDT����CS��EIP������ת�����жϴ���������, �����ж�������������������
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
/*ϵͳ�����жϵĴ���*/
_IH_sys_call:

    /*exec�����������ԣ����⴦��*/
	cmpl $SYSCALL_EXEC, %eax
	je _sys_call_exec
	
    /*fork�����������ԣ����⴦��*/
	cmpl $SYSCALL_FORK, %eax
	je _sys_call_fork
	
    /*exit�����������ԣ����⴦��*/
	cmpl $SYSCALL_EXIT, %eax
	je _sys_call_exit
	
	SAVE_PROC_CTX

	pushl %ebp
	movl %esp, %ebp


	pushl %eax /*����eax, ��Ϊ���ʱ��eax/ebx/ecx/edx����ϵͳ���õĲ���*/

	/*
     *���ʱ�����ݶε�ѡ���ӺͼĴ��������û����̵�
     * ֻ�д�����Զ�������л�
     * �����û������ds ��ȫ�ֱ���, ��Ϊϵͳ���õĴ���������һ�㶼��Ҫ�����û����̵Ŀռ�
     */
	xorl %eax, %eax
	movw %ds, %ax
	movl %eax, g_syscall_param_ds

	/*�ָ��ں�̬��ѡ����*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	popl %eax /*�ָ�eax*/
	/*����ϵͳ���õĲ�����ȫ�ֱ���*/
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
	/*ϵͳ���õķ���ֵ��%eax, ֱ���޸ı���õ����������eax*/
	movl %eax, 48(%ebp)


	/*���µ��Ƚ���*/
	call schedule

	movl %ebp, %esp
	popl %ebp

	RESTORE_PROC_CTX
	iretl


/*exit�����������ԣ����⴦��*/
_sys_call_exit:
	SAVE_PROC_CTX

	pushl %ebp
	movl %esp, %ebp

	pushl %eax /*����eax, ��Ϊ���ʱ��eax/ebx/ecx/edx����ϵͳ���õĲ���*/

	/*
     *���ʱ�����ݶε�ѡ���ӺͼĴ��������û����̵�
     * ֻ�д�����Զ�������л�
     * �����û������ds ��ȫ�ֱ���, ��Ϊϵͳ���õĴ���������һ�㶼��Ҫ�����û����̵Ŀռ�
     */
	xorl %eax, %eax
	movw %ds, %ax
	movl %eax, g_syscall_param_ds

	/*�ָ��ں�̬��ѡ����*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	call sys_exit

	/*���µ��Ƚ���*/
	call schedule
    /*������Զ���ᵽ���Ϊ������������ˣ����ᱻ����*/

	movl %ebp, %esp
	popl %ebp

	RESTORE_PROC_CTX
	iretl

    

/*fork�����������ԣ����⴦��*/
_sys_call_fork:

	SAVE_PROC_CTX /*esp�����ǰ�ƶ�52�ֽ�*/

	/*�ָ��ں�̬��ѡ����*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	movl %esp, %ebp

	subl $100, %esp /*���� 100�ֽ����ֲ�����*/

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

	/*��ջ�б����������ȡ��������Ϊ�������ݸ�sys_fork*/
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
	addl $48, %esp   /*�޸Ĳ����󣬼ǵ�ҲҪ�޸�����ĳ���*/

	/*ϵͳ���õķ���ֵ��%eax, ֱ���޸ı���õ����������eax*/
    movl %eax, 44(%ebp) /* 44��Ҫ������*/

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

/*exec�����������ԣ����⴦��*/
_sys_call_exec:

	SAVE_PROC_CTX /*esp�����ǰ�ƶ�52�ֽ�*/

	/*�ָ��ں�̬��ѡ����*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	movl %esp, %ebp

	subl $100, %esp /*���� 100�ֽ����ֲ�����*/

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

	/*��ջ�б����������ȡ��������Ϊ�������ݸ�sys_exec*/
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
    leal    64(%ebp), %eax /*ϵͳ���ý����ʱ��esp������ջ�ϵĵ�ַ*/
    pushl %eax

    leal    52(%ebp), %eax /*ϵͳ���ý����ʱ��eip������ջ�ϵĵ�ַ*/
    pushl %eax

	pushl _ds
	pushl _edx
	pushl _ecx
	pushl _ebx
	pushl _eax
	call sys_exec
	addl $28, %esp   /*�޸Ĳ����󣬼ǵ�ҲҪ�޸�����ĳ���*/

	/*ϵͳ���õķ���ֵ��%eax, ֱ���޸ı���õ����������eax*/
    movl %eax, 44(%ebp) /* 44��Ҫ������*/

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
 * ��Ϊ�жϿ��ܷ����ڳ���ִ�е��κ�λ��
 * Ϊ�˷�ֹ�жϴ������̸ı�ԭ�����еļĴ�����ֵ
 * ����ǰ�������е�ͨ�üĴ�����������ָ�
 */
/************************************************************************/
/*ʱ���жϴ�������ɽ��̵ĵ���*/
_IH_irq00:
	SAVE_PROC_CTX

	/*�ָ��ں�̬��ѡ����, cs eip�Զ����ж���������ָ�*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	/*��ʾʱ��С����*/
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

	/*�ָ��ں�̬��ѡ����*/
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
/* Ӳ���ж� */
_IH_irq14:
	SAVE_PROC_CTX


	/*�ָ��ں�̬��ѡ����*/
	movw $g_kernel_data_selector, %ax;
	mov %ax, %ds;
	mov %ax, %es;
	mov %ax, %fs;
	mov %ax, %gs;

	/*���д���... do_hd��һ������ָ�룬�����Ϊ0�� ������*/
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
	 *��ȡ movָ��������ġ����������ĵķ�ʽ��������riscӲ����ϵ
     * ԭ�ͣ� _switch(Ŀ����̿��ƿ�ָ��, ��ǰ���̿��ƿ�ָ��, Ŀ�����tssѡ����)
	 */
	movl 8(%ebp), %ebx	 /*Ŀ����̿��ƿ�ָ��*/
	movl 12(%ebp), %edx  /*��ǰ���̿��ƿ�ָ��*/
	movl 16(%ebp), %eax  /*Ŀ�����tssѡ����*/
	ltr %ax
	/* ebx�˿̱����������еĽ��̵�TProcess�ṹ��ָ��*/
	/* edx�˿̱����˵�ǰ���̵�TProcess�ṹ��ָ��*/
	cmpl $0, %edx
	je .non_ctx_save

	xchgl %edx, %esp     /*ֱ�Ӱѵ�ǰ����tss�ṹ���ڵ��ڴ����Ϊջ��ʹ��pushָ�������浱ǰ���̵�������*/
	addl $(4*25), %esp 	/*���ֶ�tss.ldt��ʼ*/
	xorl %eax, %eax
	sldt %ax
	pushl %eax   /*��ǰ���̵�ldtѡ����*/
	pushl %gs
	pushl %fs
	pushl %ds
	pushl %ss
	pushl %cs
	pushl %es
	pushl %edi
	pushl %esi
	pushl %ebp
	pushl %edx	/*���� esp*/
	pushl %ebx
	pushl %edx
	pushl %ecx
	pushl %eax
	pushfl
	pushl $.current_continue	/*����eip, ��ǰ���̱��л�������ʹ�.current_continue������*/
	movl %cr3, %eax  /*����cr3*/
	pushl %eax
	/*�������������*/
	movl %edx, %esp	/*�ָ�esp*/
.non_ctx_save:
	/*��ʼ�����µ�������*/
	movl 96(%ebx), %eax  /*�����½��̵�ldt*/
	lldt %ax
    movl 28(%ebx), %eax /*�����½��̵�ҳ����ҳĿ¼*/
    movl %eax, %cr3
	/*����л�ǰ�����м��𲻱䣬��ôiret�����Զ��ָ�ss esp*/
	movl 76(%ebx), %eax
	andl $3, %eax
	cmpl $0, %eax      /*��cs�����ѡ���ӵ�RPL�Ƿ�Ϊ0 */
	je .level_not_change	/*����û�з����仯*/

    /*α��һ���жϷ��ص��ֳ�*/
	/*Ϊiret׼�� ss esp eflags cs eip*/
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
	/*ֱ���л��������н��̵Ķ�ջ*/
	movl 80(%ebx), %eax
	movw %ax, %ss	/* ss */
	movl 56(%ebx), %eax
	movl %eax, %esp	/* esp */

	/*Ϊiret׼��eflags cs eip*/
	movl 36(%ebx), %eax
	pushl %eax	/* eflags */
	movl 76(%ebx), %eax
	pushl %eax	/* cs */
	movl 32(%ebx), %eax
	pushl %eax	/* eip */
	

.switch_continue_1:
	/*׼���ָ�ebx*/
	movl 52(%ebx), %eax
	pushl %eax

	/*�ָ� es ds fs gs */
	xchgl %ebx, %esp /*ֱ�Ӱ������еĽ���tss�ṹ���ڵ��ڴ����Ϊջ��ʹ��poplָ�����ָ����̵�������*/
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

	/* �����￪ʼ�����ܷ��ʷ�ջ�ڴ���*/

	/*�ָ� eax ecx edx ebp esi edi*/
	subl $56, %esp
	popl %eax
	popl %ecx
	popl %edx
	addl $8, %esp
	popl %ebp
	popl %esi
	popl %edi

	movl %ebx, %esp
	popl %ebx	/*�ָ�ebx*/


	iretl	/*�л�������һ������,  �ָ���cs eip eflags ss esp����eflags cs eip*/
.current_continue:
	RESTORE_REG

#else
	/*
	 * ��ȡ ljmp tssѡ����,ƫ�Ƶķ�ʽ�л�����(cpuӲ���ṩ�ķ���)����ǰ���еĽ��̵������ı�����trָ���tss��
	 * Ȼ��װ��tssѡ����ָ���tss�ﱣ��������ģ���ɽ��̵��л�
     * ԭ�ͣ� _switch(Ŀ����̿��ƿ�ָ��, ��ǰ���̿��ƿ�ָ��, Ŀ�����tssѡ����)
     * ��ʹ�ø��������ʱ�򣬻����no_coproc���쳣����û�н��
	 */
	movl 8(%ebp), %eax /*Ŀ����̿��ƿ�ָ�� */
    movl pid_OFFSET_IN_TProcess(%eax), %eax  /*Ŀ����̿��ƿ���pid�ֶε�ֵ*/

	pushl %ecx
	pushl %edx

	subl $8, %esp /*��ջ�Ϸ���һ���Զ�����*/
    /* (g_first_tss_selector + pid * 8) ����Ҫ��ת�Ľ��̵�tss�ε�ѡ���ӣ�д���Զ�����*/
	movl $8, %ecx
	mull %ecx		/*EDX:EAX �� EAX * r/m32,    ��eax��8 */
	addl $g_first_tss_selector, %eax
	movw %ax, 4(%esp)
	movl $0, (%esp)
	ljmp (%esp)	/*���յģ������ת�����治����*/
	/*��ʵ����6�ֽڴ�С��һ���ڴ棬�����ֽڴ��ƫ��,ֵ����ν����2�ֽڴ��tss��selector*/


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
	addl $4,  %eax /*����ret addr*/
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