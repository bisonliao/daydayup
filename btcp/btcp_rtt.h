#ifndef _BTCP_RTT_H_INCLUDED_
#define _BTCP_RTT_H_INCLUDED_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <glib.h>
#include "tool.h"

/*
 * 用于统计 rtt的实现，基本思路是：
 * 引擎每发送一个报文（该报文是发送窗口的第一个报文），就记录下
 * 该报文的末尾sequence（对端ack过来的sequence）和发送时刻到队列send_record_list里
 * 引擎收到对端一个ack后，就尝试在队列里找到这个报文的发送时刻，从而得到一个rtt
 * 有了这些报文的rtt，就可以统计该链路的整体的rtt
 * 可见这个是一个近似值，因为丢包导致的重复发包、还有对端引擎的处理及时性，都可能影响rtt的准确性
 * 更合理的是使用icmp协议。
 * 暂时rtt的精确性不影响btcp的实现，所以就先保持当前不准确状态。
 */
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

// 引擎每发送一个报文（该报文是发送窗口的第一个报文），就记录下
int btcp_rtt_add_send_record(struct btcp_rtt_handler * rtt_handler, uint32_t ack_seq_expected);

// 引擎收到对端一个ack后，就尝试在队列里找到这个报文的发送时刻，从而得到一个rtt, 并更新current_rtt_msec
int btcp_rtt_update_rtt(struct btcp_rtt_handler * rtt_handler, uint32_t ack_seq);

int btcp_rtt_destroy(struct btcp_rtt_handler * rtt_handler);




#endif