#ifndef __bison_skiplist_h_included_
#define __bison_skiplist_h_included_

/***
 *  skip list 
 *  detail : https://zhuanlan.zhihu.com/p/347062710
 * */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

//typedef uint64_t skiplist_score_t;
typedef struct 
{
    unsigned char * ptr;
    size_t len;
} skiplist_buffer_t;

#define skiplist_total_level  (20)
#define DEBUG_SCORE_TYPE int



typedef struct skiplist_node
{   
    struct skiplist_node * prev;
    struct skiplist_node * next;
    struct skiplist_node * down;
    skiplist_buffer_t score;
    skiplist_buffer_t data;
} skiplist_node_t;

typedef int (*skiplist_cmpfunc_t)(const skiplist_buffer_t *a, const skiplist_buffer_t *b); // comparing function pointer

typedef struct
{
    skiplist_node_t * header[skiplist_total_level];
    skiplist_cmpfunc_t cmp;
    uint64_t count;
    uint64_t size;

} skiplist_handle_t;

int skiplist_compare_score_default(const skiplist_buffer_t *a, const skiplist_buffer_t *b);
int  skiplist_deep_copy_buffer2(const unsigned char * data, size_t len, skiplist_buffer_t * dst);
void skiplist_deep_copy_buffer(const skiplist_buffer_t* src, skiplist_buffer_t* dst);
void skiplist_free_buffer(skiplist_buffer_t *b);

int  skiplist_init(skiplist_handle_t * handle, skiplist_cmpfunc_t scorecmp);
void skiplist_release(skiplist_handle_t* handle);
int skiplist_search(const skiplist_handle_t * handle, skiplist_buffer_t score, skiplist_node_t * position[], int debugflag);
int skiplist_insert(skiplist_handle_t * handle, skiplist_buffer_t score, const skiplist_buffer_t * data);
int skiplist_delete(skiplist_handle_t * handle, skiplist_buffer_t score);

#endif

