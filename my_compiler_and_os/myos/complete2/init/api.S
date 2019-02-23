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
.globl _time
.type _time, @function
_time:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG

	movl $SYSCALL_TIME, %eax
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
/* int32_t _read(uint32_t fd,  void * p, size_t sz) */
.globl _read
.type _read, @function
_read:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_READ, %eax
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
	 * 程序被优化后，eax等寄存器经常在系统调用前后被直接用作保存一些局部变量
	 * 所以需要用栈来保存/恢复
	 * 这也告诉我们，编写其他函数的时候，最好不要直接修改寄存器，而是使用栈上的局部变量
	 * 因为否则可能影响到caller.
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
.globl _sync
.type _sync, @function
_sync:
	pushl %ebp
	movl %esp, %ebp

	/**
	 * 程序被优化后，eax等寄存器经常在系统调用前后被直接用作保存一些局部变量
	 * 所以需要用栈来保存/恢复
	 * 这也告诉我们，编写其他函数的时候，最好不要直接修改寄存器，而是使用栈上的局部变量
	 * 因为否则可能影响到caller.
	 */
	SAVE_REG	

	movl $SYSCALL_SYNC, %eax
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _test
.type _test, @function
_test:
	pushl %ebp
	movl %esp, %ebp

	/**
	 * 程序被优化后，eax等寄存器经常在系统调用前后被直接用作保存一些局部变量
	 * 所以需要用栈来保存/恢复
	 * 这也告诉我们，编写其他函数的时候，最好不要直接修改寄存器，而是使用栈上的局部变量
	 * 因为否则可能影响到caller.
	 */
	SAVE_REG	

	movl $SYSCALL_TEST, %eax
	movl 8(%ebp), %ebx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _mkdir
.type _mkdir, @function
_mkdir:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_MKDIR, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _rmdir
.type _rmdir, @function
_rmdir:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_RMDIR, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _readdir
.type _readdir, @function
_readdir:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_READDIR, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _open
.type _open, @function
_open:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_OPEN, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _close
.type _close, @function
_close:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_CLOSE, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _mkfile
.type _mkfile, @function
_mkfile:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_MKFILE, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _rmfile
.type _rmfile, @function
_rmfile:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_RMFILE, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _lseek
.type _lseek, @function
_lseek:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_LSEEK, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
.globl _cout
.type _cout, @function
_cout:
	pushl %ebp
	movl %esp, %ebp

	SAVE_REG	

	movl $SYSCALL_COUT, %eax
	movl 8(%ebp), %ebx
	movl 12(%ebp), %ecx
	movl 16(%ebp), %edx
	int $INT_VECTOR_SYSCALL

	RESTORE_REG	

	movl %ebp, %esp
	popl %ebp
	ret
