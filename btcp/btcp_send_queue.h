#ifndef BTCP_SEND_QUEUE_H
#define BTCP_SEND_QUEUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

/*
 * 发送队列的实现，其基础是一个定长数组实现的循环队列
 */
// 循环队列结构体
struct btcp_send_queue {
    unsigned char *buffer;  // 动态分配的数据缓冲区
    int capacity;           // 队列容量
    int head;               // 队头指针
    int tail;               // 队尾指针
    int size;               // 当前队列大小

    // start_seq是发送窗口的开始sequence
    // 即head指向的字节对应的seq，head总指向第一个有效字节的位置。
    // 为了方便，将类型设置为uint64_t，但大小不超过UINT32_MAX
    uint32_t start_seq;     
    int64_t fin_seq; //有fin请求需要发送时，值介于[0, UINT32_MAX]，否则-1
};

// 初始化队列
bool btcp_send_queue_init(struct btcp_send_queue *queue, int capacity);

// 销毁队列
void btcp_send_queue_destroy(struct btcp_send_queue *queue);

// 检查队列是否为空
bool btcp_send_queue_is_empty(struct btcp_send_queue *queue);

// 返回空闲空间的字节数
int btcp_send_queue_get_available_space(struct btcp_send_queue *queue);

// 检查队列是否已满
bool btcp_send_queue_is_full(struct btcp_send_queue *queue);

// 入队操作（字节流）
size_t btcp_send_queue_enqueue(struct btcp_send_queue *queue, const unsigned char *data, size_t data_len);

// 出队操作（字节流）
size_t btcp_send_queue_dequeue(struct btcp_send_queue *queue, unsigned char *data, size_t data_len);

// 获取队列当前大小
int btcp_send_queue_size(struct btcp_send_queue *queue);

// 清空队列
void btcp_send_queue_clear(struct btcp_send_queue *queue);

// 从队列里取一段数据，数据的范围由其对应的seq对应[from, to]
// 该函数用于发送窗口比较大时候，一次性发送多个mss，那就会逐段的从队列中获取待发送数据
// 该函数调用，不会移动底层循环队列的 head /tail，仅仅拷贝数据，循环队列对此操作一无所知
int btcp_send_queue_fetch_data(struct btcp_send_queue *queue, uint64_t from, uint64_t to, unsigned char* data); 

//初始化start_seq，或者修改seq实现发送窗口后移
// 参数start是想要设置的起始seq，为了方便，将类型设置为uint64_t，但大小不超过UINT32_MAX
int btcp_send_queue_set_start_seq(struct btcp_send_queue *queue, uint64_t start); 

// 在末尾追加一个fin请求待发送, 只修改fin_seq字段，不会占用队列空间
// 在合适的时候，引擎会把这个fin 请求发给对端
int btcp_send_queue_push_fin(struct btcp_send_queue *queue);

#endif // BTCP_SEND_QUEUE_H