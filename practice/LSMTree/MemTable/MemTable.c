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

#define SAVE_PATH "/data/SSTable0/"

int memtable_switch_table(memtable_handle_t * handle);


static int get_next_file_index(const char * path, int start)
{
 
    int i;
    for (i = start; i< 1000000; ++i)
    {
        char filename[1024];
        snprintf(filename, sizeof(filename), "%s/sstable_%d.idx", path, i);
      
        if (access(filename, F_OK) == 0 )
        {
            continue;
        }
        else
        {
            break;
        }
    }
    return i;
  
}

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



static int dump2file(skiplist_handle_t * list, const char * path, int fileIndexStart)
{
#define BLOCK_SIZE (4*1024*1024)

    skiplist_node_t * pnode = list->header[0]->next;
    int fileindex = fileIndexStart;

// two huge block used for batch writting, they are important for efficiency
    static unsigned char index[BLOCK_SIZE];
    static unsigned char data[BLOCK_SIZE];
    int indexOffsetInBlock = 0;
    int dataOffsetInBlock = 0;
    char filename[1024];

    snprintf(filename, sizeof(filename), "%s/sstable_%d.idx.tmp", path, fileindex);
    int index_fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0666 );
    if (index_fd < 0)
    {
        perror("open index file in memtable_dump!");
        return -1;
    }

    snprintf(filename, sizeof(filename), "%s/sstable_%d.dat.tmp", path, fileindex);
    int data_fd = open(filename, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (data_fd < 0)
    {
        perror("open data file in memtable_dump!");
        return -1;
    }

    write(data_fd, "sstable", 7); // make the offset of the first record is not zero
    uint32_t offsetWritten = 7; // data written offset


    while (pnode != NULL)
    {

        int oneRecordLen = pnode->score.len + pnode->data.len + 2 * sizeof(uint32_t);
        if (oneRecordLen > BLOCK_SIZE)
        {
            close(index_fd);
            close(data_fd);
            printf("data of therecord is too big to save to file\n");
            return -1;
        }

        int oneIndexLen = pnode->score.len + sizeof(uint32_t) + sizeof(uint32_t);
        if (oneIndexLen > BLOCK_SIZE) 
        {
            close(index_fd);
            close(data_fd);
            printf("index of the record is too big to save to file\n");
            return -1;
        }

        if (oneRecordLen + dataOffsetInBlock > BLOCK_SIZE )//  a huge block almost full, then write
        {
            write(data_fd, data, dataOffsetInBlock);
            offsetWritten += dataOffsetInBlock;
            dataOffsetInBlock = 0;
            continue;
        }
        if (oneIndexLen + indexOffsetInBlock > BLOCK_SIZE) //  a huge block almost full, then write
        {
            write(index_fd, index, indexOffsetInBlock);
            indexOffsetInBlock = 0;
            continue;
        }
        
        // index
        *(uint32_t *)(index + indexOffsetInBlock) = htonl(pnode->score.len);
        indexOffsetInBlock += sizeof(uint32_t);
        memcpy(index + indexOffsetInBlock, pnode->score.ptr, pnode->score.len);
        indexOffsetInBlock += pnode->score.len;
        *(uint32_t *)(index + indexOffsetInBlock) = htonl(offsetWritten+dataOffsetInBlock);
        indexOffsetInBlock += sizeof(uint32_t);

        // data
        *(uint32_t *)(data + dataOffsetInBlock) = htonl(pnode->score.len);
        dataOffsetInBlock += sizeof(uint32_t);
        memcpy(data + dataOffsetInBlock, pnode->score.ptr, pnode->score.len);
        dataOffsetInBlock += pnode->score.len;

        *(uint32_t *)(data + dataOffsetInBlock) = htonl(pnode->data.len);
        dataOffsetInBlock += sizeof(uint32_t);
        memcpy(data + dataOffsetInBlock, pnode->data.ptr, pnode->data.len);
        dataOffsetInBlock += pnode->data.len;

        pnode = pnode->next;
    }
    if (dataOffsetInBlock > 0)
    {
        write(data_fd, data, dataOffsetInBlock);
        offsetWritten += dataOffsetInBlock;
        dataOffsetInBlock = 0;
    }
    if (indexOffsetInBlock > 0)
    {
        write(index_fd, index, indexOffsetInBlock);
        indexOffsetInBlock = 0;
    }
     

    close(data_fd);
    close(index_fd);

    char newfilename[1024];
    snprintf(filename, sizeof(filename), "%s/sstable_%d.idx.tmp", path, fileindex);
    snprintf(newfilename, sizeof(filename), "%s/sstable_%d.idx", path, fileindex);
    rename(filename, newfilename);
    snprintf(filename, sizeof(filename), "%s/sstable_%d.dat.tmp", path, fileindex);
    snprintf(newfilename, sizeof(filename), "%s/sstable_%d.dat", path, fileindex);
    rename(filename, newfilename);

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
        fileIndexStart = get_next_file_index(SAVE_PATH, fileIndexStart);
        dump2file((skiplist_handle_t*)handle->immuTable, SAVE_PATH, fileIndexStart);

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
        int fileIndexStart = get_next_file_index(SAVE_PATH, 0);
        dump2file((skiplist_handle_t*)handle->immuTable, SAVE_PATH, fileIndexStart);

        skiplist_release((skiplist_handle_t*)handle->immuTable);
        free((skiplist_handle_t*)handle->immuTable);
        handle->immuTable = NULL;
    }

    if (handle->activeTable) {
        int fileIndexStart = get_next_file_index(SAVE_PATH, 0);
        dump2file(handle->activeTable, SAVE_PATH, fileIndexStart);

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