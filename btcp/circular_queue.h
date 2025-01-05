#ifndef CIRCULAR_QUEUE_INCLUDED
#define CIRCULAR_QUEUE_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// 定义循环队列结构体
typedef struct {
    unsigned char *data; // 队列存储数据的数组
    int capacity;        // 队列的容量（最大可存储的字节数）
    int size;            // 当前队列中已有的字节数
    int head;            // 队列头部索引
    int tail;            // 队列尾部索引
    uint32_t start_seq; // 记录head指向的字节对应的seq，发送缓冲用到，head总指向第一个有效字节的位置
    uint32_t end_seq; // 记录tail指向的字节对应的seq，接收缓冲用到， tail指向一个空位置
} btcp_circular_queue;

int btcp_fetch_data_from_queue(btcp_circular_queue *q, uint32_t from, uint32_t to, unsigned char* data); // 从队列里取一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_set_start_seq_queue(btcp_circular_queue *q, uint32_t start); //初始化start_seq，或者修改seq实现发送窗口后移


int btcp_put_data_into_queue(btcp_circular_queue *q, uint32_t from, uint32_t to, const unsigned char * data); // 往队列里存储一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_set_end_seq_queue(btcp_circular_queue *q, uint32_t end); //初始化end_seq，或者修改seq实现接收窗口后移

int btcp_init_queue(btcp_circular_queue *q, int initial_capacity);
void btcp_free_queue(btcp_circular_queue *q);
int btcp_resize_queue(btcp_circular_queue *q, int new_capacity);
int btcp_enqueue(btcp_circular_queue *q, const unsigned char *value, int length);
int btcp_dequeue(btcp_circular_queue *q, unsigned char *output, int length) ;
int btcp_get_queue_size(btcp_circular_queue *q);
int btcp_get_free_space(btcp_circular_queue *q);
int btcp_get_queue_capacity(btcp_circular_queue *q);

#endif