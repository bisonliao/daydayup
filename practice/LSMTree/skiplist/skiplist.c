#include "skiplist.h"
static int get_level_randomly()
{
    int level = 0;
    int i;
    
    for (i = 1; i < skiplist_total_level;  ++i)
    {
        int r = rand() % 5; // in each level , we have 1/5 ratio to get a higher level index
        if (r == 3)
        {
            level++;
        }
        else
        {
            break;
        }
    }
    return level;
    
}
void skiplist_free_buffer(skiplist_buffer_t *b)
{
    if (b->ptr) {free(b->ptr);}        
    b->ptr = NULL;
    b->len = 0;
}

void skiplist_deep_copy_buffer(const skiplist_buffer_t* src, skiplist_buffer_t* dst)
{
    skiplist_buffer_t tmp;
    memset(&tmp, 0, sizeof(tmp));
    if (src->len > 0 && src->ptr != NULL)
    {
        tmp.len = src->len;
        tmp.ptr = (unsigned char *)malloc(tmp.len);
        memcpy(tmp.ptr, src->ptr, tmp.len);
    }
    
    skiplist_free_buffer(dst);
    *dst = tmp;
}


int  skiplist_deep_copy_buffer2(const unsigned char * data, size_t len, skiplist_buffer_t * dst)
{
    skiplist_buffer_t tmp;
    memset(&tmp, 0, sizeof(tmp));
    if (len > 0 && data != NULL)
    {
        tmp.len = len;
        tmp.ptr = (unsigned char *)malloc(tmp.len);
        memcpy(tmp.ptr, data, tmp.len);
    }
    if (dst->len > 0 && dst->ptr != NULL)
    {
        free(dst->ptr);
        dst->ptr = NULL;
        dst->len = 0;
    }
    *dst = tmp;
}

int  skiplist_init(skiplist_handle_t * handle, skiplist_cmpfunc_t scorecmp)
{
    int i;
    for (i = 0; i < skiplist_total_level; i++)
    {
        handle->header[i] = (skiplist_node_t *)malloc(sizeof(skiplist_node_t));
        if (handle->header[i] == NULL)
        {
            int j;
            for (j = 0; j < i; j++)
            {
                free(handle->header[j]);
                handle->header[j] = NULL;
            }
            return -1;
        }
        memset(handle->header[i], 0, sizeof(skiplist_node_t));

        if (i > 0)
        {
            handle->header[i]->down = handle->header[i-1];
        }

    }
    if (scorecmp == NULL)
    {
        handle->cmp = skiplist_compare_score_default;
    }
    else
    {
        handle->cmp = scorecmp;
    }
    handle->count = 0;
    handle->size = 0;
    
    return 0;
}
void skiplist_release(skiplist_handle_t* handle)
{
    int i;
    for (i = 0; i < skiplist_total_level; i++)
    {
        skiplist_node_t *pnode = handle->header[i];
        skiplist_node_t *next = NULL;
        while (pnode != NULL)
        {
            skiplist_free_buffer(&pnode->data);
            skiplist_free_buffer(&pnode->score);

            next = pnode->next;
            free(pnode);
            pnode = next;
        }
        handle->header[i] = NULL;
    }
    handle->count = 0;
    handle->size = 0;
}
int skiplist_compare_score_default(const skiplist_buffer_t *a, const skiplist_buffer_t *b)
{
    size_t i;
    for (i = 0; i < a->len && i < b->len; ++i)
    {
        if (a->ptr[i] < b->ptr[i])
        {
            return -1;
        }
        else if (a->ptr[i] > b->ptr[i])
        {
            return 1;
        }
    }
    if (a->len > b->len)
    {
        return 1;
    }
    else if (a->len < b->len)
    {
        return -1;
    }
    return 0;
}

int skiplist_search(const skiplist_handle_t * handle, skiplist_buffer_t score, skiplist_node_t * position[], int debugflag)
{
    int i;
    skiplist_node_t * start = handle->header[skiplist_total_level - 1]; // the start of each level
    int bingo = 0; // get only the position to insert it, not the exactly one
    if (debugflag) printf("++++++++++++++++++++++to find %lu:\n", *(int*)(score.ptr));
    for (i = skiplist_total_level - 1 ; i >= 0; i--)
    {
        skiplist_node_t * prev= start;
        skiplist_node_t * cur = start; 
       
        while (cur != NULL)
        {
            //if (cur->score < score)
            if (handle->cmp(&cur->score, &score) < 0)
            {
                if (debugflag)
                {
                    if (cur->score.len > 0) printf("[%d][%lu]-->", i, *(DEBUG_SCORE_TYPE*)(cur->score.ptr));
                    else         printf("[%d]header-->", i);
                }
                
                prev = cur;
                cur = cur->next;

                
            }
            //else if (cur->score >= score)
            else if (handle->cmp(&cur->score, &score) >= 0)
            {
                break;
            }
        }
        //if (cur == NULL ||  cur->score > score)
        if (cur == NULL || handle->cmp(&cur->score, &score) > 0)
        {
            start = prev->down;
            position[i] = prev;
            
            if (debugflag) printf("\n");
        }
        //else if (cur->score == score)
        else if (handle->cmp(&cur->score, &score) == 0)
        {
            if (debugflag)
            {
                if (cur->score.len > 0) printf("[%d][%lu]-->", i, *(DEBUG_SCORE_TYPE*)(cur->score.ptr));
                else         printf("[%d]header-->", i);
            }
            

            start = cur->down;
            position[i] = cur;
            bingo = 1; // catch that exact one
            
        }
        else
        {
            printf("panic! line:%d\n", __LINE__);
            return -1;
        }
    }
    if (debugflag) printf("\n");
    return bingo;
}
int skiplist_insert(skiplist_handle_t * handle, skiplist_buffer_t score, const skiplist_buffer_t * data)
{
    if (score.ptr == NULL)
    {
        return -1;
    }
    skiplist_node_t * position[skiplist_total_level];
    int iret = skiplist_search(handle, score,  position, 0);
    if (iret < 0)
    {
        printf("failed to search node for score:%lu\n", *(uint64_t*)(score.ptr));
        return -1;
    }
    
    if (iret == 1) //the score node exists already, update the data directly
    {
        skiplist_deep_copy_buffer(data, &position[0]->data);
        return 0;

    }
    // insert node in the level#0
    skiplist_node_t * newnode = (skiplist_node_t *)malloc(sizeof(skiplist_node_t));
    if (newnode == NULL)
    {
        printf("failed to allocate memory for new node!\n");
        return -1;
    }
    memset(newnode, 0, sizeof(skiplist_node_t));
    skiplist_deep_copy_buffer(data, &newnode->data);
    //newnode->score = score;
    skiplist_deep_copy_buffer(&score, &newnode->score);

    skiplist_node_t * oldnext = position[0]->next;
    position[0]->next = newnode;
    newnode->prev = position[0];
    newnode->next = oldnext;
    if (oldnext)
    {
        oldnext->prev = newnode;
    }

    //insert index node if necessary
    skiplist_node_t * prev = newnode;
    int maxlevel = get_level_randomly();
    if (maxlevel > 0)
    {
        //printf("this node#%lu will has %d level index\n", score, maxlevel );
    }
    
    int i;
    for (i = 1; i <= maxlevel; ++i)
    {
        skiplist_node_t * newindex = (skiplist_node_t *)malloc(sizeof(skiplist_node_t));
        if (newindex == NULL)
        {
            printf("failed to allocate memory for new index node!\n");
            return -1;
        }
        memset(newindex, 0, sizeof(skiplist_node_t));
        //newindex->score = score;
        skiplist_deep_copy_buffer(&score, &newindex->score);
       
        skiplist_node_t * oldnext = position[i]->next;
        position[i]->next = newindex;
        newindex->prev = position[i];
        newindex->next = oldnext;
        if (oldnext)
        {
            oldnext->prev = newindex;
        }

        newindex->down = prev;
        prev = newindex;
    }
    handle->count++;
    handle->size += (score.len + data->len);
    return 1;
}
int skiplist_delete(skiplist_handle_t * handle, skiplist_buffer_t score)
{
    if (score.ptr == NULL)
    {
        return -1;
    }
    skiplist_node_t * position[skiplist_total_level];
    int iret = skiplist_search(handle, score,  position, 0);
    if (iret < 0)
    {
        printf("failed to search node for score:%lu\n", *(uint64_t*)(score.ptr));
        return -1;
    }
    if (iret == 0)// the node with score does not exist
    {
        return 0;
    }
     //the score node exists already, delete it
    int i;
    uint64_t elesize = 0;
    for (i = 0; i < skiplist_total_level; ++i)
    {
        //if (position[i]->score == score)
        if (handle->cmp(&position[i]->score, &score) == 0)
        {
            if (i == 0)
            {
                elesize = position[i]->score.len + position[i]->data.len;
            }
            skiplist_node_t * todel = position[i];
            todel->prev->next = todel->next;
            if (todel->next) 
            {
                todel->next->prev = todel->prev;
            }

            skiplist_free_buffer(&todel->data);
            skiplist_free_buffer(&todel->score);
        }
        
    }
    if (handle->count > 0) { handle->count--;}
    if (handle->size >elesize) 
    {
        handle->size += elesize;
    } 
    return 1;
}
