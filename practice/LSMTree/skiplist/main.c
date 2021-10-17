
#include "skiplist.h"

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
    skiplist_handle_t handle;
    int ret = skiplist_init(&handle, cmp);
    printf("init result:%d\n", ret);

    skiplist_buffer_t tofind[10];
    memset(tofind, 0, sizeof(tofind)); //very important
    int cnt = 0;
    int i;
    for (i = 0; i < 1000; ++i)
    {
        int value = (rand() % 997)+1;
        //int value = i + 1;
        skiplist_buffer_t score, data;
        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &score);
        skiplist_deep_copy_buffer2((const unsigned char *)&value, sizeof(value), &data);

        skiplist_insert(&handle, score, &data); 
        if (i > 700 && cnt < 10)
        {
            skiplist_deep_copy_buffer(&score, &tofind[cnt++]);
        }
        skiplist_free_buffer(&score);
        skiplist_free_buffer(&data);
         
    }
    printf("begin delete...\n");
    for (i = 0; i < 5; ++i)
    {
        printf("delete %d returns %d\n", *(int*)tofind[i].ptr, skiplist_delete(&handle, tofind[i]));
    }
    printf("begin search...\n");
    for (i = 0; i < 10; ++i)
    {
        skiplist_node_t * position[skiplist_total_level];
        printf("find %d returns %d\n", *(int*)tofind[i].ptr, skiplist_search(&handle, tofind[i], position, 1));
    }
    printf("search done\n");
    printf("element count:%lu, size:%lu\n", handle.count, handle.size);
    
    skiplist_release(&handle);
    return 0;

}