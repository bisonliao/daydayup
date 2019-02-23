#ifndef _HD_H_INCLUDED_
#define _HD_H_INCLUDED_

#include "struct.h"

int hd_add_request(TBuffer * buf, uint32_t abs_sect, int cmd, int pid);
void hd_init();
/*
 *  * 同步方式读取某块磁盘扇区，用于系统刚刚启动的时候读取文件系统信息
 *   */
int hd_read_sync(uint32_t abs_sector, unsigned char * buf);

#endif
