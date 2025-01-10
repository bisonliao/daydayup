#include "btcp_send_queue.h"
#include <stdio.h>
#include <string.h>
#include "tool.h"



static int calc_step(struct btcp_send_queue *queue, uint64_t position, uint64_t *result)
{
    position = btcp_sequence_round_in(position) ;
    
    uint64_t step;
    if (position >= queue->start_seq) {step = position - queue->start_seq;}
    else {step = btcp_sequence_round_out(position)  - queue->start_seq ;}
    if (step > queue->size)
    {
        fprintf(stderr, "fatal error! %s %d\n", __FILE__, __LINE__);
        return -1;
    }
    *result = step;
    return 0;
}

// 初始化队列
bool btcp_send_queue_init(struct btcp_send_queue *queue, int capacity) {
    if (capacity <= 0) {
        return false;  // 无效容量
    }

    queue->buffer = (unsigned char *)malloc(capacity);
    if (queue->buffer == NULL) {
        return false;  // 内存分配失败
    }

    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
    queue->fin_seq = -1;
    return true;
}
int btcp_send_queue_push_fin(struct btcp_send_queue *queue)
{
    uint64_t fin_seq = queue->start_seq + queue->size;
    fin_seq = btcp_sequence_round_in(fin_seq);
    queue->fin_seq = fin_seq;
    g_info("set fin seq to %llu in send queue", fin_seq);
    return 0;
}

// 销毁队列
void btcp_send_queue_destroy(struct btcp_send_queue *queue) {
    if (queue->buffer != NULL) {
        free(queue->buffer);
        queue->buffer = NULL;
    }
    queue->capacity = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
    queue->fin_seq = -1;
}

// 检查队列是否为空
bool btcp_send_queue_is_empty(struct btcp_send_queue *queue) {
    return queue->size == 0;
}

// 检查队列是否已满
bool btcp_send_queue_is_full(struct btcp_send_queue *queue) {
    return queue->size == queue->capacity;
}
int btcp_send_queue_get_available_space(struct btcp_send_queue *queue){
    return queue->capacity - queue->size;
}

// 入队操作（字节流）
size_t btcp_send_queue_enqueue(struct btcp_send_queue *queue, const unsigned char *data, size_t data_len) {
    if (data_len == 0 || data == NULL) {
        return 0;  // 无效输入
    }
    //如果有fin请求要发送了，那应该没有数据需要放到发送队列里了
    g_assert(queue->fin_seq < 0); 

    size_t available_space = queue->capacity - queue->size;
    size_t bytes_to_enqueue = (data_len > available_space) ? available_space : data_len;

    for (size_t i = 0; i < bytes_to_enqueue; i++) {
        queue->buffer[queue->tail] = data[i];
        queue->tail = (queue->tail + 1) % queue->capacity;  // 循环队列
    }

    queue->size += bytes_to_enqueue;
    return bytes_to_enqueue;
}

// 出队操作（字节流）
size_t btcp_send_queue_dequeue(struct btcp_send_queue *queue, unsigned char *data, size_t data_len) {
    if (data_len == 0 || data == NULL) {
        return 0;  // 无效输入
    }

    size_t bytes_to_dequeue = (data_len > queue->size) ? queue->size : data_len;

    for (size_t i = 0; i < bytes_to_dequeue; i++) {
        data[i] = queue->buffer[queue->head];
        queue->head = (queue->head + 1) % queue->capacity;  // 循环队列
    }

    queue->size -= bytes_to_dequeue;
    return bytes_to_dequeue;
}

// 获取队列当前大小
int btcp_send_queue_size(struct btcp_send_queue *queue) {
    return queue->size;
}

// 清空队列
void btcp_send_queue_clear(struct btcp_send_queue *queue) {
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
}



// 从队列里取一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_send_queue_fetch_data(struct btcp_send_queue *queue, uint64_t from, uint64_t to, unsigned char* data)
{
    uint64_t step1, step2;
    if (calc_step(queue, from, &step1) != 0 ||
        calc_step(queue, to, &step2) != 0)
    {
        return -1;
    }
    
    uint64_t i;
    int j, k;
    for (i = step1, j = 0; i <= step2; ++i, ++j)
    {
        k = (queue->head + i) % queue->capacity;
        data[j] = queue->buffer[k];
    }
    return 0;
}

//初始化start_seq，或者修改seq实现发送窗口后移
int btcp_send_queue_set_start_seq(struct btcp_send_queue *queue, uint64_t start)
{
    if (btcp_send_queue_is_empty(queue))
    {
        queue->start_seq = btcp_sequence_round_in(start) ;
        return 0;
    }
    uint64_t step;
    if (calc_step(queue, start, &step))
    {
        return -1;
    }
    
    queue->head = (queue->head + step) % queue->capacity;
    queue->start_seq = btcp_sequence_round_in( queue->start_seq + step) ;
    queue->size -= step;
    return 0;
}

 
#if 0
int main() {
    struct btcp_send_queue queue;

    // 初始化队列，容量为 1024 字节
    if (!btcp_send_queue_init(&queue, 1024)) {
        fprintf(stderr, "Failed to initialize queue.\n");
        return 1;
    }
    uint64_t start = 3321;
    btcp_send_queue_set_start_seq(&queue, start);

    // 入队操作（字节流）
    char c;
    int i;
    for (i = 0; i < 1000; ++i)
    {
        c = 'a'+(i%26);
        btcp_send_queue_enqueue(&queue, &c, 1);
    }
    printf("size=%d\n", btcp_send_queue_size(&queue));
    btcp_send_queue_set_start_seq(&queue, start+20);
    printf("size=%d\n", btcp_send_queue_size(&queue));
    char msg[100];
    btcp_send_queue_dequeue(&queue, msg, sizeof(msg)-1);
    msg[99] = 0;
    printf("msg=[%s]\n", msg);
    
    // 销毁队列
    btcp_send_queue_destroy(&queue);

    return 0;
}
#endif