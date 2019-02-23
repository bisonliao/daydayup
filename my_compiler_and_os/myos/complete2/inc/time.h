#ifndef _TIME_H_INCLUDED_
#define _TIME_H_INCLUDED_

#include "types.h"
#include "struct.h"
#include "const_def.h"
#include "global.h"

void time_init(void);
uint32_t kernel_mktime(struct tm * tm);
uint32_t current_time(); /*当前距离 1970年的秒数*/
uint32_t up_time(); /*机器启动后运行的时间, 秒*/
void localtime(uint32_t sec, struct tm * t);


#endif
