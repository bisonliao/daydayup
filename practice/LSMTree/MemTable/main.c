#include "MemTable.h"
#include "SSTable.h"
#include "../skiplist/skiplist.h"

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
    sstable_score_cmp = cmp;

  #if 0
    memtable_handle_t handle;
   
    memtable_init(&handle, 1000000000, cmp);
    int i;
    for (i = 0; i < 10000000; ++i)
    {
        
        int value = i + 1;
        int iret;
        skiplist_buffer_t score, data;
        memset(&score, 0, sizeof(score));
        memset(&data, 0, sizeof(data));

        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &score);
        value = value*5;
        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &data);

        if (iret = memtable_insert(&handle, score, &data))
        {
            printf("failed to insert, %d\n", iret);
        }
        

       
       
        skiplist_free_buffer(&score);
        skiplist_free_buffer(&data);
         
    }
    printf("\n");
    while (1)
    {
        sleep(1);
    }
    #else
    
    
    ssfile_ctx ssfile;
    ssfile_init(&ssfile, 1, 0, O_RDONLY);
   
    while (1)
    {
        skiplist_buffer_t score, data;
        int iret = ssfile_read(&ssfile, &score, &data);
        if (iret == 0)
        {
            break;
        }
        printf("%d, %d, %d, %d\n", score.len, *(int*)score.ptr, data.len, *(int*)data.ptr);
        skiplist_free_buffer(&score);
        skiplist_free_buffer(&data);

    }
    ssfile_final(&ssfile, 0);
    
  
   
    //fileMerge("sstable_000000001", 1);

    #endif
  
    return 0;
}