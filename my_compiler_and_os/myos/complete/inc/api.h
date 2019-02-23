#ifndef API_H_INCLUDED
#define API_H_INCLUDED


#include "struct.h"
#include "const_def.h"
/*供应用程序使用的两个系统调用*/
uint32_t _set_ticks(uint32_t v);
uint32_t _get_ticks();
int32_t _write(uint32_t fd,  void * p, size_t sz);
int32_t _hd(uint32_t abs_sector, void * buf, uint32_t cmd);

void proc_A();
void proc_B();
#endif
