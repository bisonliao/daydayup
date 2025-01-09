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


int btcp_set_socket_nonblock(int sockfd);
int btcp_is_readable(int sockfd, int to, char * bigbuffer, int buflen, struct sockaddr_in *client_addr);
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

int btcp_range_cmp(const void *, int, const void *, int);


#endif