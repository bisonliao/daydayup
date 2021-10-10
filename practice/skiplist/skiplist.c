#include "skiplist.h"

int  skiplist_init(skiplist_handle_t * handle)
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
    return 0;
}
void skiplist_release(skiplist_handle_t* handle)
{
    int i;
    for (i = 0; i < skiplist_total_level; i++)
    {
        skiplist_node_t *pnode = handle->header[i];
        skiplist_node_t *next = NULL;
        if (pnode != NULL)
        {
            if (pnode->data)
            {
                free(pnode->data);
            }
            next = pnode->next;
            free(pnode);
            pnode = next;
        }
    }
}

int skiplist_search(const skiplist_handle_t * handle, skiplist_score_t score, skiplist_node_t * position[], int debugflag)
{
    int i;
    skiplist_node_t * start = handle->header[skiplist_total_level - 1]; // the start of each level
    int bingo = 0; // get only the position to insert it, not the exactly one
    if (debugflag) printf("++++++++++++++++++++++to find %lu:\n", score);
    for (i = skiplist_total_level - 1 ; i >= 0; i--)
    {
        skiplist_node_t * prev= start;
        skiplist_node_t * cur = start; 
       
        while (cur != NULL)
        {
            if (cur->score < score)
            {
                if (debugflag) printf("[%d][%lu]-->", i, cur->score);
                prev = cur;
                cur = cur->next;

                
            }
            else if (cur->score >= score)
            {
                break;
            }
        }
        if (cur == NULL ||  cur->score > score)
        {
            start = prev->down;
            position[i] = prev;
            
            if (debugflag) printf("\n");
        }
        else if (cur->score == score)
        {
            if (debugflag) printf("[%d][%lu]", i, cur->score);

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

int skiplist_insert(const skiplist_handle_t * handle, skiplist_score_t score, const unsigned char * data, size_t datalen)
{
    if (score <= 0)
    {
        return -1;
    }
    skiplist_node_t * position[skiplist_total_level];
    int iret = skiplist_search(handle, score,  position, 0);
    if (iret < 0)
    {
        printf("failed to search node for score:%lu\n", score);
        return -1;
    }
    
    if (iret == 1) //the score node exists already, update the data directly
    {
        if (position[0]->data)
        {
            unsigned char * pdata = malloc(datalen);
            if (pdata == NULL)
            {
                printf("failed to allocate data memory!\n");
                return -1;
            }
            memcpy(pdata, data, datalen);
            free(position[0]->data);
            position[0]->data = pdata;
            position[0]->data_len = datalen;
            return 0;
        }
    }
    // insert node in the level#0
    skiplist_node_t * newnode = (skiplist_node_t *)malloc(sizeof(skiplist_node_t));
    if (newnode == NULL)
    {
        printf("failed to allocate memory for new node!\n");
        return -1;
    }
    memset(newnode, 0, sizeof(skiplist_node_t));
    newnode->data = malloc(datalen);
    if (newnode->data == NULL)
    {
        printf("failed to allocate data memory for new node!\n");
        return -1;
    }
    memcpy(newnode->data, data, datalen);
    newnode->score = score;

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
        newindex->score = score;
       
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

    return 1;
}

int skiplist_delete(const skiplist_handle_t * handle, skiplist_score_t score)
{
    if (score <= 0)
    {
        return -1;
    }
    skiplist_node_t * position[skiplist_total_level];
    int iret = skiplist_search(handle, score,  position, 0);
    if (iret < 0)
    {
        printf("failed to search node for score:%lu\n", score);
        return -1;
    }
    if (iret == 0)// the node with score does not exist
    {
        return 0;
    }
     //the score node exists already, delete it
    int i;
    for (i = 0; i < skiplist_total_level; ++i)
    {
        if (position[i]->score == score)
        {
            skiplist_node_t * todel = position[i];
            todel->prev->next = todel->next;
            if (todel->next) 
            {
                todel->next->prev = todel->prev;
            }

            if (todel->data)
            {
                free(todel->data);
            }
            free(todel);
        }
        
    }
    return 1;
}
