
#include "skiplist.h"

int main()
{
    skiplist_handle_t handle;
    int ret = skiplist_init(&handle);
    printf("init result:%d\n", ret);

    int tofind[10];
    int cnt = 0;
    for (int i = 0; i < 1000; ++i)
    {
        int data = (rand() % 997)+1;
        skiplist_insert(&handle, data, (const unsigned char *)&data, sizeof(data)); 
        if (i > 700 && cnt < 10)
        {
            tofind[cnt++] = data;
        }
         
    }
    printf("begin delete...\n");
    for (int i = 0; i < 5; ++i)
    {
        printf("delete %d returns %d\n", tofind[i], skiplist_delete(&handle, tofind[i]));
    }
    printf("begin search...\n");
    for (int i = 0; i < 10; ++i)
    {
        skiplist_node_t * position[skiplist_total_level];
        printf("find %d returns %d\n", tofind[i], skiplist_search(&handle, tofind[i], position, 1));
    }
    printf("search done\n");
    
    skiplist_release(&handle);
    return 0;

}