#ifndef CIRCULAR_QUEUE_INCLUDED
#define CIRCULAR_QUEUE_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义循环队列结构体
typedef struct {
    unsigned char *data; // 队列存储数据的数组
    int capacity;        // 队列的容量（最大可存储的字节数）
    int size;            // 当前队列中已有的字节数
    int head;            // 队列头部索引
    int tail;            // 队列尾部索引
} btcp_circular_queue;

int btcp_init_queue(btcp_circular_queue *q, int initial_capacity);
void btcp_free_queue(btcp_circular_queue *q);
int btcp_resize_queue(btcp_circular_queue *q, int new_capacity);
int btcp_enqueue(btcp_circular_queue *q, const unsigned char *value, int length);
int btcp_dequeue(btcp_circular_queue *q, unsigned char *output, int length) ;
int btcp_get_queue_size(btcp_circular_queue *q);
int btcp_get_free_space(btcp_circular_queue *q);
int btcp_get_queue_capacity(btcp_circular_queue *q);

#endif