

.data

.globl g_syscall_nr
g_syscall_nr:
	.long 24

.globl g_syscall_entry
g_syscall_entry:
	.long sys_get_ticks_lo 	/* 0 */
	.long sys_get_ticks_hi
	.long sys_sleep
	.long sys_read
	.long sys_write		/* 4*/
	.long sys_hd
	.long sys_time		/* 6 */
	.long sys_sync		/* 7 */
	.long sys_test	
	.long sys_mkdir		
	.long sys_rmdir		/* 10 */
	.long sys_readdir		
	.long sys_open		
	.long sys_close		
	.long sys_mkfile	/* 14 */		
	.long sys_rmfile	/* 15 */		
	.long sys_lseek		/* 16 */		
	.long sys_cout		/* 17 */
	.long sys_fork		/* 18 */
	.long sys_exec		/* 19 */
	.long sys_exit		/* 20 */
	.long sys_cin		/* 21 */
	.long sys_access	/* 22 */
	.long sys_wait	    /* 23 */
