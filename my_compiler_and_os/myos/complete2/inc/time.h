#ifndef _TIME_H_INCLUDED_
#define _TIME_H_INCLUDED_

#include "types.h"
#include "struct.h"
#include "const_def.h"
#include "global.h"

void time_init(void);
uint32_t kernel_mktime(struct tm * tm);
uint32_t current_time(); /*��ǰ���� 1970�������*/
uint32_t up_time(); /*�������������е�ʱ��, ��*/
void localtime(uint32_t sec, struct tm * t);


#endif
