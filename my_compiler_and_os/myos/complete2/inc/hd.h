#ifndef _HD_H_INCLUDED_
#define _HD_H_INCLUDED_

#include "struct.h"

int hd_add_request(TBuffer * buf, uint32_t abs_sect, int cmd, int pid);
void hd_init();
/*
 *  * ͬ����ʽ��ȡĳ���������������ϵͳ�ո�������ʱ���ȡ�ļ�ϵͳ��Ϣ
 *   */
int hd_read_sync(uint32_t abs_sector, unsigned char * buf);

#endif
