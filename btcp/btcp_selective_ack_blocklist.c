#include "btcp_selective_ack_blocklist.h"

int btcp_sack_blocklist_init(struct btcp_sack_blocklist *list)
{
    list->blocklist = NULL;
}

static void btcp_sack_blocklist_truncate(struct btcp_sack_blocklist * list)
{
   
    GList *iter = list->blocklist;
    
    int counter = 0;
    while ( iter != NULL)
    {
        struct btcp_range *rec = (struct btcp_range *)iter->data;
        if (counter > 100) 
        {
            
            //删除当前元素，有点小技巧
            GList * next = iter->next;
            list->blocklist = g_list_delete_link(list->blocklist, iter); 
            free(rec);
            iter = next;  //保证iter还有效
        }
        else
        {
            iter = iter->next;
            counter++;
        }
    }
    return;
}
int btcp_sack_blocklist_add_record(struct btcp_sack_blocklist *list, const struct btcp_range * range)
{
    struct btcp_range *rec = (struct btcp_range *)malloc(sizeof(struct btcp_range));
    if (rec == NULL)
    {
        return -1;
    }
    rec->from = range->from;
    rec->to = range->to;
    list->blocklist = g_list_insert(list->blocklist, rec, 0);

    btcp_sack_blocklist_truncate(list);
    return 0;
}
int btcp_sack_blocklist_destroy(struct btcp_sack_blocklist *list)
{
    for (const GList *iter = list->blocklist; iter != NULL; iter = iter->next) {
        struct range *rec = (struct range *)iter->data;
        free(rec);
    }
    g_list_free(list->blocklist);  // 释放链表
    list->blocklist = NULL;
    return 0;
}