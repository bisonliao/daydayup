#ifndef BTCP_RANGE_H
#define BTCP_RANGE_H

#include <glib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#endif // BTCP_RANGE_H