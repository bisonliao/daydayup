#ifndef _HD_H_INCLUDED_
#define _HD_H_INCLUDED_

#include "struct.h"

int hd_add_request(char * buf, uint32_t abs_sect, int cmd, int pid);
void hd_init();

#endif
