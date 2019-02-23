#ifndef _SYSCALL_ENTRY_H_INCLUDED_
#define _SYSCALL_ENTRY_H_INCLUDED_

uint32_t   sys_get_ticks_lo(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx);
uint32_t   sys_get_ticks_hi(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx);
uint32_t   sys_sleep(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx);
uint32_t   sys_read(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx);
uint32_t   sys_write(uint32_t eax, uint32_t ebx, uint32_t ecx, uint32_t edx);

#endif
