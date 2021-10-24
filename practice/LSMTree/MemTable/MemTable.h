#ifndef __bison_memtable_h_included__
#define __bison_memtable_h_included__

#include "skiplist.h"
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

typedef struct MemTable
{
    skiplist_handle_t* activeTable;
    volatile skiplist_handle_t* immuTable;
    uint64_t maxSize;
    skiplist_cmpfunc_t cmp;
    pthread_t dumpThread;
    pthread_mutex_t mutex;

}memtable_handle_t ;

int memtable_init(memtable_handle_t * handle, uint64_t maxSize, skiplist_cmpfunc_t cmp);
int memtable_release(memtable_handle_t * handle);
int memtable_search(memtable_handle_t * handle, skiplist_buffer_t score, skiplist_buffer_t * data);
int memtable_insert(memtable_handle_t * handle, skiplist_buffer_t score, const skiplist_buffer_t * data);
int memtable_delete(memtable_handle_t * handle, skiplist_buffer_t score);





#endif