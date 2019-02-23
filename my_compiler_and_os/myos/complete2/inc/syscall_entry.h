#ifndef _SYSCALL_ENTRY_H_INCLUDED_
#define _SYSCALL_ENTRY_H_INCLUDED_

uint32_t    sys_get_ticks_lo(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_get_ticks_hi(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_sleep(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_read(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_write(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_time(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_cin(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_cout(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_access(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_mkdir(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_mkfile(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_rmdir(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_readdir(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_rmfile(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_open(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_close(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t    sys_lseek(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t     sys_exit(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
uint32_t     sys_wait(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds);
int     sys_fork(uint32_t eip, uint32_t cs, uint32_t eflags, uint32_t esp, uint32_t ss,
        uint32_t edi, uint32_t esi, uint32_t ebp, uint32_t edx, uint32_t ecx, uint32_t ebx, uint32_t eax);
int     sys_exec(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx, uint32_t ds,
        uint32_t * eip_for_userspace,
        uint32_t * esp_for_userspace);
#endif
