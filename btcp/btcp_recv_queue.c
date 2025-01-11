#include "btcp_recv_queue.h"
#include <stdio.h>
#include <string.h>
#include "tool.h"

/*
 * 接收队列的实现
 */

// 计算 range的起止位置 position 距离quene的tail的步长，且确保步长不要超过队列空闲可写空间的范围
static int calc_step(struct btcp_recv_queue *queue, uint64_t position, uint64_t *result)
{
    //如果超过 UINT32_MAX，就转换为小于UINT32_MAX的整数
    position = btcp_sequence_round_in(position); 
  
    uint64_t step;
    if (position >= queue->start_seq) 
    {
        step = position - queue->start_seq;
    }
    else 
    {
        step = btcp_sequence_round_out((uint32_t)position) - queue->start_seq ;
    }
    if ( (step+1) > btcp_recv_queue_get_available_space(queue))
    {
        g_warning("fatal error! %s %d, %llu, %d, %llu, %u\n", __FILE__, __LINE__,
                step, btcp_recv_queue_get_available_space(queue), position, queue->start_seq);
    
        return -1;
    }
    *result = step;
    return 0;
}

// 初始化队列
bool btcp_recv_queue_init(struct btcp_recv_queue *queue, int capacity) {
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
    queue->start_seq = 0;
    queue->rcvd_range_list = NULL;
    queue->fin_seq = -1;
    return true;
}

// 销毁队列
void btcp_recv_queue_destroy(struct btcp_recv_queue *queue) {
    if (queue->buffer != NULL) {
        free(queue->buffer);
        queue->buffer = NULL;
    }
    queue->capacity = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
    queue->fin_seq = -1;
    btcp_range_free_list(queue->rcvd_range_list);

}

// 检查队列是否为空
bool btcp_recv_queue_is_empty(struct btcp_recv_queue *queue) {
    return queue->size == 0;
}

// 检查队列是否已满
bool btcp_recv_queue_is_full(struct btcp_recv_queue *queue) {
    return queue->size == queue->capacity;
}
int btcp_recv_queue_get_available_space(struct btcp_recv_queue *queue){
    return queue->capacity - queue->size;
}

// 入队操作（字节流）
size_t btcp_recv_queue_enqueue(struct btcp_recv_queue *queue, const unsigned char *data, size_t data_len) {
    if (data_len == 0 || data == NULL) {
        return 0;  // 无效输入
    }

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
size_t btcp_recv_queue_dequeue(struct btcp_recv_queue *queue, unsigned char *data, size_t data_len) {
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
int btcp_recv_queue_size(struct btcp_recv_queue *queue) {
    return queue->size;
}

// 清空队列
void btcp_recv_queue_clear(struct btcp_recv_queue *queue) {
    queue->head = 0;
    queue->tail = 0;
    queue->size = 0;
}


#if 0
// 从队列里取一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_recv_queue_fetch_data(struct btcp_recv_queue *queue, uint64_t from, uint64_t to, unsigned char* data)
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
#endif

int btcp_recv_queue_save_fin_req(struct btcp_recv_queue *queue, uint32_t seq)
{
    queue->fin_seq = seq;
    return 0;
}
// 向队列里保存一段数据，数据的范围由其对应的seq对应[from, to]
int btcp_recv_queue_save_data(struct btcp_recv_queue *queue, uint64_t from, uint64_t to, 
            const unsigned char* data)
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
        k = (queue->tail + i) % queue->capacity;
        queue->buffer[k] = data[j];
    }
    struct btcp_range * r = (struct btcp_range *)malloc(sizeof(struct btcp_range));
    if (r == NULL)
    {
        return -1;
    }
    r->from = from;
    r->to = to;
    queue->rcvd_range_list = g_list_append(queue->rcvd_range_list, r);
    return 0;
}
int btcp_recv_queue_try_move_wnd(struct btcp_recv_queue *queue)
{

    GList *result = NULL;

#ifdef _P_
    printf("rcvd segment:\n");
    btcp_range_print_list(queue->rcvd_range_list);
    printf("wnd start seq:%u\n", queue->start_seq);
#endif

    btcp_range_list_combine(queue->rcvd_range_list, &result);
    if (result == NULL)
    {
        return 0;
    }
    btcp_range_free_list(queue->rcvd_range_list);
    queue->rcvd_range_list = result;

#ifdef _P_
    printf("after combined,rcvd segment:\n");
    btcp_range_print_list(queue->rcvd_range_list);
#endif

    //int len = g_list_length(queue->rcvd_range_list);

    GList *iter = queue->rcvd_range_list;
    
    while ( iter != NULL)
    {
        struct btcp_range *range = (struct btcp_range *)iter->data;
        if (range->from == queue->start_seq) //收到了一段紧接着 start_seq 的数据
        {
            //move wnd
            uint64_t steps =  range->to - range->from + 1;
            if ((steps + 1) > btcp_recv_queue_get_available_space(queue))
            {
                g_warning( "fatal error! %s %d", __FILE__, __LINE__);
                return -1;
            }
#ifdef _P_
            printf("move wnd, steps=%llu, [%llu, %llu]\n", steps, range->from, range->to);
            printf("before move, start_seq:%u, tail:%d, size:%d\n", queue->start_seq, 
                                                queue->tail,
                                                queue->size); 
#endif
            queue->start_seq = btcp_sequence_round_in(range->to+1) ;
            queue->tail = (queue->tail + steps) % (queue->capacity);
            queue->size += steps;
#ifdef _P_
            printf("after move, start_seq:%u, tail:%d, size:%d\n", queue->start_seq, 
                                                queue->tail,
                                                queue->size); 
#endif
            //删除当前元素，有点小技巧
            GList * next = iter->next;
            queue->rcvd_range_list = g_list_delete_link(queue->rcvd_range_list, iter); 
            free(range);
            iter = next;

            return steps;
            
        }
        else
        {
#ifdef _P_
            printf("GAP! [%llu, %llu]\n", range->from, range->to);
#endif
            iter = iter->next;
        }
    }
#ifdef _P_
    printf("after move,rcvd segment:\n");
    btcp_range_print_list(queue->rcvd_range_list);
#endif

#undef _P_

}

//初始化start_seq，或者修改seq实现窗口后移
int btcp_recv_queue_set_start_seq(struct btcp_recv_queue *queue, uint64_t start)
{
    if (btcp_recv_queue_is_empty(queue))
    {
        queue->start_seq = btcp_sequence_round_in(start) ;
        return 0;
    }
    uint64_t step;
    if (calc_step(queue, start, &step))
    {
        return -1;
    }
    
    queue->tail = (queue->tail + step) % queue->capacity;
    queue->start_seq = btcp_sequence_round_in( queue->start_seq + step) ;
    queue->size += step;
    return 0;
}

 
#if 0
int main() {
    struct btcp_recv_queue queue;

    // 初始化队列，容量为 1024 字节
    if (!btcp_recv_queue_init(&queue, 1024)) {
        fprintf(stderr, "Failed to initialize queue.\n");
        return 1;
    }
    uint64_t start = 3321;
    btcp_recv_queue_set_start_seq(&queue, start);

    // 入队操作（字节流）
    char c;
    int i;
    for (i = 0; i < 1000; ++i)
    {
        c = 'a'+(i%26);
        btcp_recv_queue_enqueue(&queue, &c, 1);
    }
    printf("size=%d\n", btcp_recv_queue_size(&queue));
    btcp_recv_queue_set_start_seq(&queue, start+20);
    printf("size=%d\n", btcp_recv_queue_size(&queue));
    char msg[100];
    btcp_recv_queue_dequeue(&queue, msg, sizeof(msg)-1);
    msg[99] = 0;
    printf("msg=[%s]\n", msg);
    
    // 销毁队列
    btcp_recv_queue_destroy(&queue);

    return 0;
}
#endif