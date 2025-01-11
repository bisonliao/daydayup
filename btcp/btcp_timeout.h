#ifndef btcp_timer_H
#define btcp_timer_H

#include <stdbool.h>
#include <time.h>
#include <glib.h>
#include "tool.h"

/*
 * 本模块实现一个定时器。主体是一个链表，记录了未来会超时的事件
 * 
 * 发送tcp段后，要在定时器里记录，设定超时时间
 * 未来如果超时了，需要重新发送
 * 
 * 引擎每次尝试发送tcp段的时候，也需要查看是不是已经有发送了
 * 如果有发送过还没有超时，就先不发送，也是依赖定时器里的数据
 * 
 * 上层引擎在使用该定时器的时候，里面保存的event类型是struct btcp_range
 * 即一个tcp请求覆盖的用户数据起止sequence范围
 */


// 事件结构体
struct btcp_timer_event {
    time_t expire_time;  // 超时时间（绝对时间）
    void *event_data;    // 事件数据
    int event_len;       // 事件数据长度
    struct btcp_timer_event *next;  // 指向下一个事件
};

extern int btcp_errno;

// 超时控制器结构体
struct btcp_timeout {
    struct btcp_timer_event *head;  // 链表头指针
};

// 初始化超时控制器
void btcp_timer_init(struct btcp_timeout *handler);

// 销毁超时控制器
void btcp_timer_destroy(struct btcp_timeout *handler);

// 检查是否有超时的事件
int btcp_timer_check(struct btcp_timeout *handler, void *event, int *len);

// 插入一个未来超时的事件
int btcp_timer_add_event(struct btcp_timeout *handler, int sec, const void *event, int len, 
                    int (*event_cmp)(const void *, int, const void *, int));

// 删除指定事件
int btcp_timer_remove_event(struct btcp_timeout *handler, const void *event, int len, 
                        int (*event_cmp)(const void *, int, const void *, int));

//遍历得到所有事件，保存在result里
int  btcp_timer_get_all_event(struct btcp_timeout *handler, GList **result);

// 删除指定范围内的事件
int btcp_timer_remove_range(struct btcp_timeout *handler, const struct btcp_range * range);

// 删除指定开始seq的事件
int btcp_timer_remove_by_from(struct btcp_timeout *handler, uint32_t from);

// 搜索指定的事件
const void* btcp_timer_find_event(struct btcp_timeout *handler, const void *event, int len, 
                        int (*event_cmp)(const void *, int, const void *, int));


#endif // btcp_timer_H