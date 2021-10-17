#include "MemTable.h"
#include <errno.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>

#define SAVE_PATH "/data/SSTableSaved/"

static int get_next_file_index(const char * path)
{
    int i;
    for (i = 0; i< 1000000; ++i)
    {
        char filename[1024];
        snprintf(filename, sizeof(filename), "%s/sstable_%d", path, i);
      
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



int memtable_dump(memtable_handle_t * handle, const char * path, int fileIndexStart)
{
#define BLOCK_SIZE (4*1024*1024)

    skiplist_node_t * pnode = handle->immuTable->header[0]->next;
    int fileindex = fileIndexStart;

    while (pnode != NULL)
    {
        static unsigned char index[BLOCK_SIZE];
        static unsigned char data[BLOCK_SIZE];
        int indexOffset = 0;
        int dataOffset = 0;
        char filename[1024];
        memset(index, 0, sizeof(index));
        snprintf(filename, sizeof(filename), "%s/sstable_%d", path, fileindex++);
        int fd = open(filename, O_CREAT|O_RDWR);
        if (fd < 0)
        {
            perror("open in memtable_dump!");
            return -1;
        }
        write(fd, index, BLOCK_SIZE);
        int blockNum = 1;
        while (pnode != NULL)
        {
            int oneRecordLen = pnode->score.len + pnode->data.len + 2 * sizeof(uint32_t);
            if (dataOffset + oneRecordLen > BLOCK_SIZE) //write current data page ,and new  another data page
            {
                memset(data+dataOffset, 0, BLOCK_SIZE-dataOffset);
                write(fd, data, BLOCK_SIZE);
                dataOffset = 0;
                blockNum++;
            }

            int oneIndexLen = pnode->score.len + sizeof(uint32_t) + sizeof(uint32_t);
            if (indexOffset + oneIndexLen > BLOCK_SIZE) // index block full
            {
                break;
            }
            // index
            *(uint32_t*)(index+indexOffset) = htonl(pnode->score.len);
            indexOffset += sizeof(uint32_t);
            memcpy(index+indexOffset, pnode->score.ptr, pnode->score.len);
            indexOffset += pnode->score.len;
            *(uint32_t*)(index+indexOffset) = htonl(dataOffset+blockNum*BLOCK_SIZE);
            indexOffset += sizeof(uint32_t);

            // data
            *(uint32_t*)(data+dataOffset) = htonl(pnode->score.len);
            dataOffset += sizeof(uint32_t);
            memcpy(data+dataOffset, pnode->score.ptr, pnode->score.len);
            dataOffset += pnode->score.len;

            *(uint32_t*)(data+dataOffset) = htonl(pnode->data.len);
            dataOffset += sizeof(uint32_t);
            memcpy(data+dataOffset, pnode->data.ptr, pnode->data.len);
            dataOffset += pnode->data.len;

            pnode = pnode->next;

        }
        if (dataOffset > 0)
        {
            memset(data+dataOffset, 0, BLOCK_SIZE-dataOffset);
            write(fd, data, BLOCK_SIZE);
            dataOffset = 0;
        }
        memset(index+indexOffset, 0, BLOCK_SIZE-indexOffset);
        lseek(fd, 0, SEEK_SET);
        write(fd, index, BLOCK_SIZE);
        close(fd);
    }

    return 0;


}

static void * dump(void * arg)
{
     memtable_handle_t * handle = (memtable_handle_t * )arg;
 
     printf(">>%d, %lx\n", __LINE__, handle);
     pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
     pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
     while (1)
     {
        
         pthread_mutex_lock(&handle->mutex);
         if (handle->immuTable == NULL) // dumped already or no data to dump
         {
             pthread_mutex_unlock(&handle->mutex);
             sleep(1);
             continue;
         }
         printf(">>%d, %lx\n", __LINE__, handle);
         
         //dump starts
        printf("we dump it into file...\n");
        int fileIndexStart = get_next_file_index(SAVE_PATH);
        memtable_dump(handle, SAVE_PATH, fileIndexStart);

        // dump has been finished
        skiplist_release(handle->immuTable);
        free(handle->immuTable);
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
    printf("%d:%lx\n", __LINE__, handle);

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
    if (handle->activeTable) {
        skiplist_release(handle->activeTable);
        free(handle->activeTable);
        handle->activeTable = NULL;
    }
    if (handle->immuTable) 
    {
        skiplist_release(handle->immuTable);
        free(handle->immuTable);
        handle->immuTable = NULL;
    }
    
    handle->maxSize = 0;
    pthread_cancel(handle->dumpThread);
    pthread_mutex_destroy(&handle->mutex);

    return 0;
}
int memtable_search(const memtable_handle_t * handle, skiplist_buffer_t score, skiplist_buffer_t * data)
{
    skiplist_node_t * position[skiplist_total_level];
    if (handle->activeTable && skiplist_search(handle->activeTable, score, position, 0) == 1)
    {
        skiplist_deep_copy_buffer(&position[0]->data, data);
        return 1;
    }
    if (handle->immuTable && skiplist_search(handle->immuTable, score, position, 0) == 1)
    {
        skiplist_deep_copy_buffer(&position[0]->data, data);
        return 1;
    }
    return 0;
}

int memtable_insert(memtable_handle_t * handle, skiplist_buffer_t score, const skiplist_buffer_t * data)
{
    if (skiplist_insert(handle->activeTable, score, data) < 0)
    {
        return -1;
    }
    uint64_t totalSize = handle->activeTable->size + handle->activeTable->count * sizeof(skiplist_node_t)*2;
    if (totalSize > handle->maxSize)
    {
        int iret;
        iret = pthread_mutex_trylock(&handle->mutex);// dumping still if mutex is locked, then current thread will wait
        if (iret != 0 && EBUSY == iret)
        {
            printf("dump is still in progress...\n");
            return -2;
        }
        if (0 != iret)
        {
            perror("pthread_mutex_trylock in memtable_insert:");
            return -3;
        }

        // here dumping has finished, we can change immuTable
        skiplist_handle_t* tmp = (skiplist_handle_t*)malloc(sizeof(skiplist_handle_t));
        if (tmp == NULL || skiplist_init(tmp, handle->cmp) != 0)
        {
            pthread_mutex_unlock(&handle->mutex);
            return -1;
        }
        handle->immuTable = handle->activeTable;
        handle->activeTable = tmp;
        pthread_mutex_unlock(&handle->mutex); 
    }
    return 0;
}
int memtable_delete(memtable_handle_t * handle, skiplist_buffer_t score)
{
    //不支持删除，业务层面在data里记录自定义的墓碑态
    return  -1;
}
