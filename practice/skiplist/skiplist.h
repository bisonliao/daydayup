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

typedef uint64_t skiplist_score_t;
#define skiplist_total_level  (20)

typedef struct skiplist_node
{   
    struct skiplist_node * prev;
    struct skiplist_node * next;
    struct skiplist_node * down;
    skiplist_score_t score;
    unsigned char * data;
    size_t data_len;
} skiplist_node_t;

typedef struct
{
    skiplist_node_t * header[skiplist_total_level];

} skiplist_handle_t;

int  skiplist_init(skiplist_handle_t * handle);
void skiplist_release(skiplist_handle_t* handle);

int skiplist_search(const skiplist_handle_t * handle, skiplist_score_t score, skiplist_node_t * position[], int debugflag);
int skiplist_insert(const skiplist_handle_t * handle, skiplist_score_t score, const unsigned char * data, size_t datalen);
int skiplist_delete(const skiplist_handle_t * handle, skiplist_score_t score);

#endif

