#include "const_def.h"


.text
/* uint32_t _get_ticks() */
.globl _get_ticks
.type _get_ticks, @function
_get_ticks:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	movl $0, %eax
	int $INT_VECTOR_SYSCALL

	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret
/* uint32_t _set_ticks(uint32_t v) */
.globl _set_ticks
.type _set_ticks, @function
_set_ticks:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $1, %eax
	movl 8(%ebp), %ebx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret
/* uint32_t _sleep(uint32_t v) */
.globl _sleep
.type _sleep, @function
_sleep:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	movl $2, %eax
	movl 8(%ebp), %ebx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG

	movl %ebp, %esp
	popl %ebp
	ret
/* int32_t _write(uint32_t fd,  void * p, size_t sz) */
.globl _write
.type _write, @function
_write:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_WRITE, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG 

	movl %ebp, %esp
	popl %ebp
	ret
/* int32_t _hd(uint32_t abs_sector, void * buf, uint32_t cmd);*/
.globl _hd
.type _hd, @function
_hd:
	pushl %ebp
	movl %esp, %ebp

	/**
	 * �����Ż���eax�ȼĴ���������ϵͳ����ǰ��ֱ����������һЩ�ֲ�����
	 * ������Ҫ��ջ������/�ָ�
	 * ��Ҳ�������ǣ���д����������ʱ����ò�Ҫֱ���޸ļĴ���������ʹ��ջ�ϵľֲ�����
	 * ��Ϊ�������Ӱ�쵽caller.
	 */
	SAVE_REG	

	movl $SYSCALL_HD, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret