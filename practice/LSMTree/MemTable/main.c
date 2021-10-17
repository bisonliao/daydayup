#include "MemTable.h"

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
    memtable_handle_t handle;
    memtable_init(&handle, 100000000, cmp);
    int i;
    for (i = 0; i < 100000000; ++i)
    {
        
        int value = i + 1;
        skiplist_buffer_t score, data;
        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &score);
        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &data);

        memtable_insert(&handle, score, &data); 
        if ((i % 9973) == 7)
        {
            //sleep(1);
        }
       
        skiplist_free_buffer(&score);
        skiplist_free_buffer(&data);
         
    }
    while (1)
    {
        sleep(1);
    }
    return 0;
}