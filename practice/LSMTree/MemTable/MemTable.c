#include "MemTable.h"
#include <errno.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "SSTable.h"
#include "../skiplist/skiplist.h"

int memtable_switch_table(memtable_handle_t * handle);





static int mutex_lock_timeout(pthread_mutex_t * m,  int millisecs)
{
    int i;
    for (i = 0;; ++i)
    {
        int iret;
        iret = pthread_mutex_trylock(m);
        if (iret != 0 && EBUSY == iret)
        {
            if (i >= millisecs)
            {
                return -1;
            }
            usleep(1000); // sleep 1 milliseconds
            continue;
        }
        else if (iret != 0)
        {
            perror("pthread_mutex_trylock in mutex_lock_timeout():");
            return -1;
        }
        else if (iret == 0)
        {
            break;
        }
    }
    return 0;
}



static int dump2file(skiplist_handle_t * list,  int fileIndexStart)
{
    // dat file structure:
    //  "sstable" + [score_len(4B) + score_value(?) + data_len(4B) + data_value(?)]
    // idx file structure:
    //  indexCnt(4B) + [score_len(4B) + score_value(?)+ offsetInDatFile(4B)] + IndexRangeInfo
    //  IndexRangeInfo:
    //  minScoreLen(4B) + minScoreData(?) + maxScoreLen(4B) + maxScoreData(?) + rangeInfoLen(4B) 

    ssfile_ctx ssfile;
    ssfile_init(&ssfile, 0, fileIndexStart, O_APPEND);
    skiplist_node_t * pnode = list->header[0]->next;
    while (pnode != NULL)
    {
        ssfile_append(&ssfile, &pnode->score, &pnode->data);
        pnode = pnode->next;
    }
    ssfile_final(&ssfile, 1);

    return 0;

}

static void * dump(void * arg)
{
     memtable_handle_t * handle = (memtable_handle_t * )arg;
     
     int fileIndexStart = 0;
     pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
     pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
     while (1)
     {
        int iret;
        iret =  pthread_mutex_lock(&handle->mutex);
         if (iret != 0)
         {
             perror("mutext_lock in dump():");
             continue;
         }
         if (handle->immuTable == NULL) // dumped already or no data to dump
         {
             pthread_mutex_unlock(&handle->mutex);
             usleep(100);
             continue;
         }
        printf(">>%d, %lx\n", __LINE__, handle->immuTable);
         
         //dump starts
        //printf("we dump it into file...\n");
        fileIndexStart = sstable_getNextFileIndex(0, fileIndexStart);
        dump2file((skiplist_handle_t*)handle->immuTable,  fileIndexStart);

        // dump has been finished
        skiplist_release((skiplist_handle_t*)handle->immuTable);
        free((skiplist_handle_t*)handle->immuTable);
        handle->immuTable = NULL;
        pthread_mutex_unlock(&handle->mutex);

     }
   
}

int memtable_init(memtable_handle_t * handle, uint64_t maxSize, skiplist_cmpfunc_t cmp)
{
    handle->activeTable = (skiplist_handle_t*)malloc(sizeof(skiplist_handle_t));
    if (handle->activeTable == NULL)
    {
        return -1;
    }
    handle->immuTable = NULL;
    if (skiplist_init(handle->activeTable, cmp) != 0 )
    {
        free(handle->activeTable);
        handle->activeTable = NULL;
        return -2;
    }
    handle->maxSize = maxSize;
    handle->cmp = cmp;
    //printf("%d:%lx\n", __LINE__, handle);

    int iret;
    iret = pthread_mutex_init(&handle->mutex, NULL);
    if (iret != 0)
    {
        free(handle->activeTable);
        handle->activeTable = NULL;
        return -3;
    }
    #if 1
    iret = pthread_create(&handle->dumpThread, NULL, dump, handle);
    if (iret != 0)
    {
        free(handle->activeTable);
        handle->activeTable = NULL;
        pthread_mutex_destroy(&handle->mutex);
        return -4;
    }
    #endif
    return 0;
}
int memtable_release(memtable_handle_t * handle)
{
    
    
    pthread_cancel(handle->dumpThread);
    usleep(1000*5);

    if (handle->immuTable) 
    {
        int fileIndexStart = sstable_getNextFileIndex(0, 0);
        dump2file((skiplist_handle_t*)handle->immuTable, fileIndexStart);

        skiplist_release((skiplist_handle_t*)handle->immuTable);
        free((skiplist_handle_t*)handle->immuTable);
        handle->immuTable = NULL;
    }

    if (handle->activeTable) {
        int fileIndexStart = sstable_getNextFileIndex(0, 0);
        dump2file(handle->activeTable,  fileIndexStart);

        skiplist_release(handle->activeTable);
        free(handle->activeTable);
        handle->activeTable = NULL;
    }
   
    
    handle->maxSize = 0;
    pthread_cancel(handle->dumpThread);
    pthread_mutex_destroy(&handle->mutex);

    return 0;
}
int memtable_search(memtable_handle_t * handle, skiplist_buffer_t score, skiplist_buffer_t * data)
{
    skiplist_node_t * position[skiplist_total_level];
    if (handle->activeTable && skiplist_search(handle->activeTable, score, position, 0) == 1)
    {
        skiplist_deep_copy_buffer(&position[0]->data, data);
        return 1;
    }
    if (0 == mutex_lock_timeout(&handle->mutex, 100))
    {
        if (handle->immuTable && skiplist_search((skiplist_handle_t*)handle->immuTable, score, position, 0) == 1)
        {
            skiplist_deep_copy_buffer(&position[0]->data, data);
            pthread_mutex_unlock(&handle->mutex);
            return 1;
        }
        pthread_mutex_unlock(&handle->mutex);
        return 0;
    }
    else
    {
        return -1;
    }
    
}
int memtable_switch_table(memtable_handle_t * handle)
{
    int iret;
    iret = mutex_lock_timeout(&handle->mutex, 10);
    if (iret == 0 && handle->immuTable != NULL )
    {
        pthread_mutex_unlock(&handle->mutex);
        return -1;
    }
    if (0 != iret)
    {
        return -1;
    }

    // here dumping has finished, we can change immuTable
    skiplist_handle_t *tmp = (skiplist_handle_t *)malloc(sizeof(skiplist_handle_t));
    if (tmp == NULL || skiplist_init(tmp, handle->cmp) != 0)
    {
        pthread_mutex_unlock(&handle->mutex);
        printf("failed to create new skiplist");
        return -1;
    }
    handle->immuTable = handle->activeTable;
    handle->activeTable = tmp;
    pthread_mutex_unlock(&handle->mutex);
    return 0;
}

int memtable_insert(memtable_handle_t * handle, skiplist_buffer_t score, const skiplist_buffer_t * data)
{
   
    uint64_t totalSize = handle->activeTable->size + handle->activeTable->count * sizeof(skiplist_node_t)*2;
    if (totalSize > handle->maxSize) // need dump to file
    {
        //printf("I will sleep 30 secs\n");
        //sleep(30);
        if (memtable_switch_table(handle))
        {
            return -2;
        }
  
    }
    if (skiplist_insert(handle->activeTable, score, data) < 0)
    {
        return -1;
    }
    return 0;
}
int memtable_delete(memtable_handle_t * handle, skiplist_buffer_t score)
{
    //不支持删除，业务层面在data里记录自定义的墓碑态
    return  -1;
}