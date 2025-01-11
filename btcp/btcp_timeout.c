#include "btcp_timeout.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tool.h"

/*
 * 本模块实现一个定时器。主体是一个链表，记录了未来会超时的事件
 */

// 初始化超时控制器
void btcp_timer_init(struct btcp_timeout *handler) {
    handler->head = NULL;
}

// 销毁超时控制器
void btcp_timer_destroy(struct btcp_timeout *handler) {
    struct btcp_timer_event *current = handler->head;
    while (current != NULL) {
        struct btcp_timer_event *next = current->next;
        free(current->event_data);  // 释放事件数据
        free(current);              // 释放事件节点
        current = next;
    }
    handler->head = NULL;
}

// 检查是否有超时的事件
int btcp_timer_check(struct btcp_timeout *handler, void *event, int *len) {
    time_t current_time = time(NULL);  // 获取当前时间

    struct btcp_timer_event *current = handler->head;
    if (current == NULL || current->expire_time > current_time) {
        return 0;  // 没有超时的事件
    }
    g_assert(current->event_len == *len);
    g_assert(current->event_data != NULL);
    
    
    // 返回超时的事件
    memcpy(event, current->event_data, current->event_len);
    *len = current->event_len;

    // 从链表中移除该事件
    handler->head = current->next;
    free(current->event_data);  // 释放事件数据
    free(current);

    return 1;
}

// 插入一个未来超时的事件
int btcp_timer_add_event(struct btcp_timeout *handler, int sec, const void *event, int len, int (*event_cmp)(const void *, int, const void *, int)) {
    time_t expire_time = time(NULL) + sec;  // 计算超时时间

    g_assert(len == sizeof(struct btcp_range));

    // 不管三七二十一，先尝试删除再插入，保证有序
    btcp_timer_remove_event(handler, event, len, event_cmp);
    
   
    // 创建新事件
    struct btcp_timer_event *new_event = (struct btcp_timer_event *)malloc(sizeof(struct btcp_timer_event));
    if (new_event == NULL) {
        return -1;  // 内存分配失败
    }
    new_event->expire_time = expire_time;
    new_event->event_data = malloc(len);
    if (new_event->event_data == NULL) {
        free(new_event);
        return -1;  // 内存分配失败
    }
    memcpy(new_event->event_data, event, len);  // 复制事件数据
    new_event->event_len = len;
    new_event->next = NULL;

    // 插入到链表中，按超时时间排序
    struct btcp_timer_event *current = handler->head;
    struct btcp_timer_event *prev = NULL;
    while (current != NULL && current->expire_time < expire_time) {
        prev = current;
        current = current->next;
    }

    if (prev == NULL) {
        // 插入到链表头部
        new_event->next = handler->head;
        handler->head = new_event;
    } else {
        // 插入到中间或尾部
        new_event->next = prev->next;
        prev->next = new_event;
    }

    return 0;  // 返回 0 表示成功插入新事件
}
int  btcp_timer_get_all_event(struct btcp_timeout *handler, GList **result)
{
    
    GList * tmp_result = NULL;
    struct btcp_timer_event *current = handler->head;
    
    for (; current != NULL;current = current->next ) 
    {
        struct btcp_range * e = malloc(current->event_len);
        if (e == NULL) 
        {
            return -1;    
        }
        memcpy(e, current->event_data, current->event_len);
        if (e->from > e->to) //可能因为tcp sequence是32bit而发生了回绕，恢复成更好理解的值
        {
            e->to = btcp_sequence_round_out(e->to) ;
        }
        tmp_result = g_list_append(tmp_result, e);
    }
    *result = tmp_result;
    return 0;
}

// 删除指定事件
int btcp_timer_remove_event(struct btcp_timeout *handler, const void *event, int len, int (*event_cmp)(const void *, int, const void *, int)) {
    struct btcp_timer_event *current = handler->head;
    struct btcp_timer_event *prev = NULL;

    while (current != NULL) {
        if (event_cmp(current->event_data, current->event_len, event, len) == 0) {
            // 找到匹配的事件，从链表中移除
            if (prev == NULL) {
                handler->head = current->next;
            } else {
                prev->next = current->next;
            }

            // 释放事件数据和节点
            free(current->event_data);
            free(current);
            return 0;  // 返回 0 表示成功删除
        }

        prev = current;
        current = current->next;
    }

    return -1;  // 返回 -1 表示未找到匹配的事件
}
const void* btcp_timer_find_event(struct btcp_timeout *handler, const void *event, int len, 
                        int (*event_cmp)(const void *, int, const void *, int))
{
    struct btcp_timer_event *current = handler->head;
    
    while (current != NULL) 
    {
        if (event_cmp(current->event_data, current->event_len, event, len) == 0) 
        {
            // 找到匹配的事件
            return current->event_data;

            current = current->next;
        }
    }
    return NULL;

}

int btcp_timer_remove_range(struct btcp_timeout *handler, const struct btcp_range * range)
{
    struct btcp_timer_event *current = handler->head;
    struct btcp_timer_event *prev = NULL;

    while (current != NULL) {
        struct btcp_range range_in_list = *(const struct btcp_range *)(current->event_data);
        if (range_in_list.to <range_in_list.from)
        {
            range_in_list.to = btcp_sequence_round_out(range_in_list.to );
        }
    // 这里删除是出过内存越界访问的bug的！
        if (range->from <= range_in_list.from && range->to >= range_in_list.to)
        {
            // 找到匹配的事件，从链表中移除
            struct btcp_timer_event * tmp = current->next;
            if (prev == NULL) {
                handler->head = current->next;
            } else {
                prev->next = current->next;
            }

            // 释放事件数据和节点
            free(current->event_data);
            free(current);

            current = tmp;
            continue;
        }

        prev = current;
        current = current->next;
    }

    return 0;
}
int btcp_timer_remove_by_from(struct btcp_timeout *handler, uint32_t from)
{
    struct btcp_timer_event *current = handler->head;
    struct btcp_timer_event *prev = NULL;

    while (current != NULL) {
        struct btcp_range range_in_list = *(const struct btcp_range *)(current->event_data);
        // 这里删除是出过内存越界访问的bug的！
        if (range_in_list.from == from)
        {
            // 找到匹配的事件，从链表中移除
            struct btcp_timer_event * tmp = current->next;
            if (prev == NULL) {
                handler->head = current->next;
            } else {
                prev->next = current->next;
            }

            // 释放事件数据和节点
            free(current->event_data);
            free(current);

            current = tmp;
            continue;
        }

        prev = current;
        current = current->next;
    }

    return 0;
}


// 事件比对函数
static int event_cmp(void *event1, int len1, void *event2, int len2) {
    if (len1 != len2) return -1;
    return memcmp(event1, event2, len1);
}

#if 0
int main() {
    struct btcp_timeout handler;
    btcp_timer_init(&handler);

    // 添加事件
    int event1_data = 100;
    btcp_timer_add_event(&handler, 15, &event1_data, sizeof(event1_data), event_cmp);

    int event2_data = 200;
    btcp_timer_add_event(&handler, 3, &event2_data, sizeof(event2_data), event_cmp);

    int event3_data = 300;
    btcp_timer_add_event(&handler, 5, &event3_data, sizeof(event3_data), event_cmp);

    int event4_data = 400;
    btcp_timer_add_event(&handler, 5, &event4_data, sizeof(event4_data), event_cmp);


    // 删除事件
    if (btcp_timer_remove_event(&handler, &event1_data, sizeof(event1_data), event_cmp) == 0) {
        printf("Event1 removed.\n");
    } else {
        printf("Event1 not found.\n");
    }

    // 检查超时事件
    for (int i = 0; i < 20; i++) {
        int event;
        int len = sizeof(int);
        int flag = false;
        while (btcp_timer_check(&handler, &event, &len) == 1) 
        {
            printf("Event timed out: %d\n", event);
            flag = true;
        } 
        if (!flag)
        {
            printf("no event timeout\n");
        }
        sleep(1);  // 每秒检查一次
    }

    // 销毁超时控制器
    btcp_timer_destroy(&handler);

    return 0;
}
#endif

