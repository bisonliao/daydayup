#include "circular_queue.h"

// 初始化队列
int btcp_init_queue(btcp_circular_queue *q, int initial_capacity) {
    q->data = (unsigned char *)malloc(initial_capacity * sizeof(unsigned char));
    if (!q->data) {
        return -1; // 内存分配失败
    }
    q->capacity = initial_capacity;
    q->size = 0;
    q->head = 0;
    q->tail = 0;
    return 0;
}

// 释放队列
void btcp_free_queue(btcp_circular_queue *q) {
    free(q->data);
}

// 修改队列大小
int btcp_resize_queue(btcp_circular_queue *q, int new_capacity) {
    if (new_capacity < q->size) {
        return -1; // 新容量不足以容纳当前队列中的数据
    }

    unsigned char *new_data = (unsigned char *)malloc(new_capacity * sizeof(unsigned char));
    if (!new_data) {
        return -1; // 内存分配失败
    }

    for (int i = 0; i < q->size; i++) {
        new_data[i] = q->data[(q->head + i) % q->capacity];
    }

    free(q->data);
    q->data = new_data;
    q->capacity = new_capacity;
    q->head = 0;
    q->tail = q->size;

    return 0;
}

// 入队操作，将多个字节入队
int btcp_enqueue(btcp_circular_queue *q, const unsigned char *value, int length) {
    if (length > q->capacity - q->size) {
        return -1; // 队列没有足够的空闲空间
    }

    for (int i = 0; i < length; i++) {
        q->data[q->tail] = value[i];
        q->tail = (q->tail + 1) % q->capacity;
    }
    q->size += length;

    return 0;
}

// 出队操作，从队列中取出指定数量的字节
int btcp_dequeue(btcp_circular_queue *q, unsigned char *output, int length) {
    if (length > q->size) {
        return -1; // 队列中数据不足
    }

    for (int i = 0; i < length; i++) {
        output[i] = q->data[q->head];
        q->head = (q->head + 1) % q->capacity;
    }
    q->size -= length;

    return 0;
}

// 获取队列中的数据大小
int btcp_get_queue_size(btcp_circular_queue *q) {
    return q->size;
}

// 获取队列的空闲空间大小
int btcp_get_free_space(btcp_circular_queue *q) {
    return q->capacity - q->size;
}

// 获取队列的当前容量
int btcp_get_queue_capacity(btcp_circular_queue *q) {
    return q->capacity;
}

#if 0
// 打印队列中的数据（用于调试）
void btcp_print_queue(btcp_circular_queue *q) {
    printf("Queue elements: ");
    for (int i = 0; i < q->size; i++) {
        printf("%d ", q->data[(q->head + i) % q->capacity]);
    }
    printf("\n");
}


// 测试主程序
int main() {
    btcp_circular_queue q;

    // 初始化队列，容量为 10
    if (btcp_init_queue(&q, 10) != 0) {
        printf("Failed to initialize queue\n");
        return -1;
    }

    // 入队操作
    unsigned char input1[] = {1, 2, 3, 4, 5};
    btcp_enqueue(&q, input1, 5);
    btcp_print_queue(&q);

    unsigned char input2[] = {6, 7, 8};
    btcp_enqueue(&q, input2, 3);
    btcp_print_queue(&q);

    // 获取队列状态
    printf("Queue size: %d\n", btcp_get_queue_size(&q));       // 输出 8
    printf("Free space: %d\n", btcp_get_free_space(&q));       // 输出 2
    printf("Queue capacity: %d\n", btcp_get_queue_capacity(&q)); // 输出 10

    // 出队操作
    unsigned char output[5];
    btcp_dequeue(&q, output, 5);
    printf("Dequeued: ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    btcp_print_queue(&q);

    // 修改队列大小
    if (btcp_resize_queue(&q, 15) == 0) {
        printf("Resized the queue\n");
    }
    btcp_print_queue(&q);

    // 再次入队操作
    unsigned char input3[] = {9, 10, 11, 12, 13};
    btcp_enqueue(&q, input3, 5);
    btcp_print_queue(&q);

    // 释放队列内存
    btcp_free_queue(&q);

    return 0;
}

#endif