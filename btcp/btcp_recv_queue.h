#ifndef BTCP_RECV_QUEUE_H
#define BTCP_RECV_QUEUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <glib.h>

// 循环队列结构体
struct btcp_recv_queue {
    unsigned char *buffer;  // 动态分配的数据缓冲区
    int capacity;           // 队列容量
    int head;               // 队头指针
    int tail;               // 队尾指针
    int size;               // 当前队列大小
    // 记录tail指向的字节对应的seq，tail总指向第一个空闲位置。
    //为了方便，将类型设置为uint64_t，但大小不超过UINT32_MAX
    uint32_t expected_seq;   
    GList * rcvd_range_list; //在当前接收窗口下，已经收到的数据段
    int64_t fin_seq; //缓存收到的对端发过来的fin请求（可能相比数据报文后发先至先，所以要缓存到接收队列）
};

// 初始化队列
bool btcp_recv_queue_init(struct btcp_recv_queue *queue, int capacity);

// 销毁队列
void btcp_recv_queue_destroy(struct btcp_recv_queue *queue);

// 检查队列是否为空
bool btcp_recv_queue_is_empty(struct btcp_recv_queue *queue);

// 返回空闲空间的字节数
int btcp_recv_queue_get_available_space(struct btcp_recv_queue *queue);

// 检查队列是否已满
bool btcp_recv_queue_is_full(struct btcp_recv_queue *queue);

// 入队操作（字节流）
size_t btcp_recv_queue_enqueue(struct btcp_recv_queue *queue, const unsigned char *data, size_t data_len);

// 出队操作（字节流）
size_t btcp_recv_queue_dequeue(struct btcp_recv_queue *queue, unsigned char *data, size_t data_len);

// 获取队列当前大小
int btcp_recv_queue_size(struct btcp_recv_queue *queue);

// 清空队列
void btcp_recv_queue_clear(struct btcp_recv_queue *queue);

// 从队列里取一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_recv_queue_fetch_data(struct btcp_recv_queue *queue, uint64_t from, uint64_t to, unsigned char* data); 

//初始化 expected_seq，或者修改seq实现窗口后移
// 参数start是想要设置的起始seq，为了方便，将类型设置为uint64_t，但大小不超过UINT32_MAX
int btcp_recv_queue_set_expected_seq(struct btcp_recv_queue *queue, uint64_t start); 

// 向队列里保存一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_recv_queue_save_data(struct btcp_recv_queue *queue, uint64_t from, uint64_t to, 
            const unsigned char* data);

//尝试向前移动接收窗口
int btcp_recv_queue_try_move_wnd(struct btcp_recv_queue *queue);

//收到了对端发来的fin请求，缓存到接收队列
int btcp_recv_queue_save_fin_req(struct btcp_recv_queue *queue, uint32_t seq);

#endif // btcp_recv_QUEUE_H