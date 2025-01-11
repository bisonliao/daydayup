#ifndef TOOL_H_INCLUDED
#define TOOL_H_INCLUDED

#include <stdint.h>
#include <glib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <glib.h>

/*
 *这是一个工具模块，包含了一些比较底层、通用的工具函数
 */


int btcp_set_socket_nonblock(int sockfd);
int btcp_is_readable(int sockfd, int to, char * bigbuffer, int buflen, struct sockaddr_in *client_addr);

/*
 * tcp的sequence是32bit的，很容易就用完并且要回绕
 * 下面这些函数就提供了对sequence回绕和增减的处理
 */
// 回绕到32bit正整数范围内
uint32_t btcp_sequence_round_in(uint64_t original);
// 将发生了回绕的32bit正整数展开到64bit值
uint64_t btcp_sequence_round_out(uint32_t original);
// 32bit sequence 增长steps，保持在32bit范围内
uint32_t btcp_sequence_step_forward(uint32_t original, uint32_t steps);
// 32bit sequence 减小steps，保持在32bit范围内
uint32_t btcp_sequence_step_back(uint32_t original, uint32_t steps);

//从系统启动开始的时间，不受系统时间更改影响，毫秒精度
uint64_t btcp_get_monotonic_msec(); 


/*
 * tcp发送窗口里的数据是一段一段发送出去的，每一段用range来描述
 * 那么在反复尝试发送数据的时候，由于发送窗口大小在不断变化、已经发送的数据有的丢包了有的在途
 * 所以需要一些函数来进行 range的运算
 * 下面这些函数就提供了对range的运算
 */

// 定义 range 结构体
struct btcp_range {
    uint64_t from;
    uint64_t to;
};
// 从一组 range 中减去另一组 range
int btcp_range_subtract( GList *a,  GList *b, GList **result);

// 打印 range 列表
void btcp_range_print_list(const GList *list);

// 释放 range 列表
void btcp_range_free_list(GList *list);

/* 
很重的操作，把列表里的有重叠或者连续的range都合并，例如：
a: [1, 6] [2, 3] [7, 10] [9, 12]
Combined result: [1, 12]
*/
// 把列表里的有重叠或者连续的range都合并
int btcp_range_list_combine(GList *a, GList **result) ;
// 两个range的相等比较
int btcp_range_equal(const void *, int, const void *, int);
// 两个range是否有重叠的判断
int btcp_range_overlap(const struct btcp_range *r1, const struct btcp_range *r2);


#endif