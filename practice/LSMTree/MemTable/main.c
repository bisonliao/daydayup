#include "MemTable.h"
#include "SSTable.h"

int cmp(const skiplist_buffer_t *a, const skiplist_buffer_t *b)
{
    if (a->len == 0 && b->len > 0)
    {
        return -1;
    }
    else if (a->len > 0 && b->len == 0)
    {
        return 1;
    }
    else if (a->len == 0 && b->len == 0)
    {
        return 0;
    }
    int aa =*(int*)(a->ptr) ;
    int bb =*(int*)(b->ptr) ;
    if (aa < bb ) return -1;
    if (aa > bb ) return 1;
    return 0;
}

int main()
{
  #if 0
    memtable_handle_t handle;
    memtable_init(&handle, 1000000000, cmp);
    int i;
    for (i = 0; i < 100000000; ++i)
    {
        
        int value = i + 1;
        skiplist_buffer_t score, data;
        memset(&score, 0, sizeof(score));
        memset(&data, 0, sizeof(data));

        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &score);
        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &data);

        if (memtable_insert(&handle, score, &data))
        {
            printf("failed to insert\n");
        }

        if ( (i % 999997) == 7 && i > 1000000)
        {
            int value2 = value ;

            skiplist_buffer_t score2, data2;
            memset(&data2, 0, sizeof(data2));
            memset(&score2, 0, sizeof(score2));
            
            skiplist_deep_copy_buffer2((const unsigned char *)&value2, sizeof(value2), &score2);
            int iret = memtable_search(&handle, score2, &data2);
            if (iret != 1 || *(int*)data2.ptr != value2 || data2.len != sizeof(int))
            {
                printf("search failed! %d, %d, %d\n", value2, iret, data2.len);
            }
            else
            {
                printf("search success!%d\n", value2);
            }

        }
        
       
        skiplist_free_buffer(&score);
        skiplist_free_buffer(&data);
         
    }
    while (1)
    {
        sleep(1);
    }
    #else

    //sstable_printIdxFile("/data/SSTable0/sstable_000000000.idx");
    merged_file_t files[100];
    int maxNum = 100;
    sstable_score_cmp = cmp;
    sstable_getAllMergedFiles(0, files, &maxNum);
    int i;
    for (i = 0; i < maxNum; ++i)
    {
        printf("[%s][%d][%d]\n", files[i].filename, *(int*)(files[i].beginScore.ptr), *(int*)(files[i].endScore.ptr));
    }

    #endif
  
    return 0;
}