#ifndef BTCP_SELECTIVE_ACK_BLOCKLIST_H_INCLUDED
#define BTCP_SELECTIVE_ACK_BLOCKLIST_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <glib.h>
#include "tool.h"

/*
 *   TCP 拥塞控制中的一个经典问题，称为 “重传冗余” 或 “不必要的重传”。当发送窗口较大且有一定丢包率时
 *   例如：假设发送窗口大小为3，起始seq=100，发送了 p1 p2 p3 三个包，其中p1途中丢失了，对端收到了p2 p3 ，
 *   会发送2个ack，ack_seq都会等于100，而不是对p1 p2 p3进行ack。
 *   
 *   发送端一段时间后就会发现p1, p2 p3都没有被ack而超时，这时候又重新发送p1 p2 p3。
 *   对端收到p1后立即移动接收窗口获得了完整的p1 p2 p3, sequence移动到p3报文的末尾字节
 *   位置，再接着收到重发的p2 p3，这时候的p2 p3是白白重复传输的。
 *
 *   优化方案可以使用SACK选项，即选择性ack，对端收到p2, p3后ack的时候，在首部选项字段里也带上对p2 p3的ack
 *   发送端收到sack信息后，不再重复发送
 * 
 * 该模块使用一个队列，保存对端发过来的selective ack信息。
 * 本端引擎在发送tcp段的时候，对于已经在队里里的被ack过的段进行剔除，避免重复发送
*/

struct btcp_sack_blocklist
{
    GList *blocklist;
};

int btcp_sack_blocklist_init(struct btcp_sack_blocklist *list);
int btcp_sack_blocklist_add_record(struct btcp_sack_blocklist *list , const struct btcp_range * range);
int btcp_sack_blocklist_destroy(struct btcp_sack_blocklist *list);




#endif