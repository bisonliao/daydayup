

.data

.globl g_syscall_nr
g_syscall_nr:
	.long 6

.globl g_syscall_entry
g_syscall_entry:
	.long sys_get_ticks_lo 	/* 0 */
	.long sys_get_ticks_hi
	.long sys_sleep
	.long sys_read
	.long sys_write		/* 5*/
	.long sys_hd
