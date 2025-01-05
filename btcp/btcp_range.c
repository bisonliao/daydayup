#include "btcp_range.h"
#include <stdio.h>
#include <stdlib.h>

// 比较两个 range 的起始位置
static int btcp_range_compare(const struct btcp_range *r1, const struct btcp_range *r2) {
    if (r1->from < r2->from) return -1;
    if (r1->from > r2->from) return 1;
    return 0;
}

// 检查两个 range 是否重叠
static int btcp_range_overlap(const struct btcp_range *r1, const struct btcp_range *r2) {
    return (r1->from <= r2->to && r1->to >= r2->from);
}

// 从单个 range 中减去另一个 range
static GList *btcp_range_subtract_single(const struct btcp_range *a, const struct btcp_range *b) {
    GList *result = NULL;

    //printf("a:[%llu, %llu], b:[%llu, %llu]\n", a->from, a->to, b->from, b->to);

    if (!btcp_range_overlap(a, b)) {
        // 如果没有重叠，直接返回 a
        struct btcp_range *new_range = malloc(sizeof(struct btcp_range));
        new_range->from = a->from;
        new_range->to = a->to;
        result = g_list_append(result, new_range);
        return result;
    }

    // 处理重叠部分
    if (a->from < b->from) {
        // 左边剩余部分
        struct btcp_range *left = malloc(sizeof(struct btcp_range));
        left->from = a->from;
        left->to = b->from - 1;
        //printf("left:[%llu, %llu]\n", left->from, left->to);
        result = g_list_append(result, left);
    }

    if (a->to > b->to) {
        // 右边剩余部分
        struct btcp_range *right = malloc(sizeof(struct btcp_range));
        right->from = b->to + 1;
        right->to = a->to;
        //printf("right:[%llu, %llu]\n", right->from, right->to);
        result = g_list_append(result, right);
    }

    return result;
}


// 深度拷贝的时候用的元素拷贝函数
static gpointer copy_range(gconstpointer src, gpointer user_data) {
    const struct btcp_range *original = (const struct btcp_range *)src;
    struct btcp_range *a1 = malloc(sizeof(struct btcp_range));
    a1->from = original->from;
    a1->to = original->to;
    return a1;
}
// 从一组 range 中减去另一组 range
int btcp_range_subtract( GList *a,  GList *b, GList **result) {
    GList *tmp_result = NULL;
    GList *aa = NULL;
    aa = g_list_copy_deep(a, copy_range, NULL);
   

    // 遍历 b 中的每个 range
    for (const GList *b_iter = b; b_iter != NULL; b_iter = b_iter->next)
    {
        struct btcp_range *b_range = (struct btcp_range *)b_iter->data;
        GList * new_a = NULL;

        // 遍历 a 中的每个 range
        for (const GList *a_iter = aa; a_iter != NULL; a_iter = a_iter->next)
        {
            struct btcp_range *a_range = (struct btcp_range *)a_iter->data;

            if (btcp_range_overlap(a_range, b_range)) // 有重叠就减去
            {
                GList *subtracted = btcp_range_subtract_single(a_range, b_range);
                new_a = g_list_concat(new_a, subtracted); //subtracted这时候就变成了new_a的一部分，可以认为无效了，不要释放，也不要做其他操作
            }
            else // 没有重叠就原封未动的深度拷贝这个元素到结果list
            {
                struct btcp_range *a1 = malloc(sizeof(struct btcp_range));
                a1->from = a_range->from;
                a1->to = a_range->to;
                new_a = g_list_append(new_a, a1); 
            }
        }
        // 处理下一个b_range的时候，就用new_a替代aa。因为aa里的每个range都减去了b_range，得到一个新的aa了
        if (new_a != NULL)
        {
            btcp_range_free_list(aa);
            aa = new_a;
            printf("get new a: ");
            btcp_range_print_list(aa);
        }
    }

    *result = aa;
    return 0;
}
/*
// 比较两个 range 的起始位置
static int btcp_range_compare(const void *a, const void *b) {
    const struct btcp_range *r1 = (const struct btcp_range *)a;
    const struct btcp_range *r2 = (const struct btcp_range *)b;
    if (r1->from < r2->from) return -1;
    if (r1->from > r2->from) return 1;
    return 0;
}
*/

// 合并一组 range
int btcp_range_list_combine(GList *a, GList **result) {
    if (a == NULL) {
        *result = NULL;
        return 0;
    }

    // 将 GList 转换为数组并排序
    int count = g_list_length(a);
    struct btcp_range *ranges = malloc(count * sizeof(struct btcp_range));
    int i = 0;
    for (GList *iter = a; iter != NULL; iter = iter->next) {
        ranges[i++] = *(struct btcp_range *)iter->data;
    }
    qsort(ranges, count, sizeof(struct btcp_range), 
            (int (*)(const void *, const void *))btcp_range_compare);

    // 合并重叠的 range
    GList *combined = NULL;
    struct btcp_range current = ranges[0];

    for (i = 1; i < count; i++) {
        if (ranges[i].from <= current.to + 1) {
            // 如果当前 range 和下一个 range 重叠或相邻，合并它们
            if (ranges[i].to > current.to) {
                current.to = ranges[i].to;
            }
        } else {
            // 如果不重叠，将当前 range 添加到结果中
            struct btcp_range *new_range = malloc(sizeof(struct btcp_range));
            *new_range = current;
            combined = g_list_append(combined, new_range);

            // 更新当前 range
            current = ranges[i];
        }
    }

    // 添加最后一个 range
    struct btcp_range *new_range = malloc(sizeof(struct btcp_range));
    *new_range = current;
    combined = g_list_append(combined, new_range);

    // 释放临时数组
    free(ranges);

    // 返回结果
    *result = combined;
    return 0;
}


// 打印 range 列表
void btcp_range_print_list(const GList *list) {
    for (const GList *iter = list; iter != NULL; iter = iter->next) {
        struct btcp_range *range = (struct btcp_range *)iter->data;
        printf("[%llu, %llu] ", range->from, range->to);
    }
    printf("\n");
}

// 释放 range 列表
void btcp_range_free_list(GList *list) {
    for (const GList *iter = list; iter != NULL; iter = iter->next) {
        struct btcp_range *range = (struct btcp_range *)iter->data;
        free(range);
    }
    g_list_free(list);  // 释放链表
}

int btcp_range_cmp(void *a, int a_len, void *b, int b_len)
{
    struct btcp_range * aa = (struct btcp_range * )a;
    struct btcp_range * bb = (struct btcp_range * )b;
    if (aa->from == bb->from && aa->to == bb->to)
    {
        return 0;
    }
    return aa->from - bb->from;
}


#if 0
int main() {
    
    GList *a = NULL;
    GList *b = NULL;
    struct btcp_range *a1, *b1;

    a1 = malloc(sizeof(struct btcp_range));
    a1->from = 1;
    a1->to = 6;
    a = g_list_append(a, a1);

    a1 = malloc(sizeof(struct btcp_range));
    a1->from = 10;
    a1->to = 14;
    a = g_list_append(a, a1);

    a1 = malloc(sizeof(struct btcp_range));
    a1->from = 5;
    a1->to = 9;
    a = g_list_append(a, a1);
    ////////////////////////////////////
    b1 = malloc(sizeof(struct btcp_range));
    b1->from = 15;
    b1->to = 17;
    b = g_list_append(b, b1);

    b1 = malloc(sizeof(struct btcp_range));
    b1->from = 0;
    b1->to = 2;
    b = g_list_append(b, b1);

    b1 = malloc(sizeof(struct btcp_range));
    b1->from = 5;
    b1->to = 6;
    b = g_list_append(b, b1);

    // 计算 result = a - b
    GList *result = NULL;
   
    btcp_range_subtract(a, b, &result);
    

    // 打印结果
    printf("Result of a - b: ");
    btcp_range_print_list(result);
    btcp_range_free_list(result);
   
    // 释放 range 列表
    btcp_range_free_list(a);
    btcp_range_free_list(b);
    btcp_range_free_list(result);

    {

        // 创建 range 列表
        GList *a = NULL;
        struct btcp_range *r1 = malloc(sizeof(struct btcp_range));
        r1->from = 1;
        r1->to = 6;
        a = g_list_append(a, r1);

        struct btcp_range *r2 = malloc(sizeof(struct btcp_range));
        r2->from = 2;
        r2->to = 3;
        a = g_list_append(a, r2);

        struct btcp_range *r3 = malloc(sizeof(struct btcp_range));
        r3->from = 8;
        r3->to = 10;
        a = g_list_append(a, r3);

        struct btcp_range *r4 = malloc(sizeof(struct btcp_range));
        r4->from = 9;
        r4->to = 12;
        a = g_list_append(a, r4);

        // 打印原始 range 列表
        printf("Original ranges: ");
        btcp_range_print_list(a);

        // 合并 range 列表
        GList *result = NULL;
        btcp_range_list_combine(a, &result);

        // 打印合并后的 range 列表
        printf("Combined ranges: ");
        btcp_range_print_list(result);

        // 释放 range 列表
        btcp_range_free_list(a);
        btcp_range_free_list(result);
    }

    return 0;
}

#endif