#ifndef _BTCP_RTT_H_INCLUDED_
#define _BTCP_RTT_H_INCLUDED_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <glib.h>
#include "tool.h"

struct btcp_rtt_send_record
{
    uint32_t ack_seq; // 发送一个报文，改报文可能对应的ack报文的ack_seq
    uint64_t sent_msec; //发送报文的时刻，毫秒精度
};

struct btcp_rtt_handler
{
    GList * send_record_list; //btcp_rtt_send_record 记录的列表
    uint32_t current_rtt_msec; //当前计算得到的rtt值，毫秒精度
};

int btcp_rtt_init(struct btcp_rtt_handler * rtt_handler);
int btcp_rtt_add_send_record(struct btcp_rtt_handler * rtt_handler, uint32_t ack_seq_expected);
int btcp_rtt_update_rtt(struct btcp_rtt_handler * rtt_handler, uint32_t ack_seq);
int btcp_rtt_destroy(struct btcp_rtt_handler * rtt_handler);




#endif