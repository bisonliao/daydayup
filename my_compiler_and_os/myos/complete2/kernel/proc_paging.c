/* 为每各用户态进程管理paging */
#include "const_def.h"

#include "global.h"

static struct paging_t * g_paging = (struct paging_t *)PROC_PAGING_ORG;

int proc_paging_init()
{
    int i;
    if ( sizeof(struct paging_t) * MAX_PROC_NR > (PROC_PAGING_END-PROC_PAGING_ORG) )
    {
        return -1;
    }
    for (i = 0; i < MAX_PROC_NR; ++i)
    {
        memset(&g_paging[i], 0, sizeof(struct paging_t) );
    }
    return 0;
}

uint32_t get_cr3_for_proc(int proc_id)
{
    if (proc_id < 1 || proc_id >= MAX_PROC_NR)
    {
        return 0;
    }
    return &g_paging[proc_id];
}
int setup_paging_for_proc(int proc_id)
{
    struct paging_t * pg = NULL;
    uint32_t user_space_org, i, j;
    if (proc_id < 1 || proc_id >= MAX_PROC_NR)
    {
        return -1;
    }
    user_space_org = FIRST_PROC_ORG+ (proc_id-1)*PROC_SPACE; /*用户态物理地址起始*/
    pg = &g_paging[proc_id]; /*页目录和页表结构体指针*/
    memset(pg, 0, sizeof(struct paging_t) );

    //设置内核态每一个页表项
    for (i = 0, j = 0; i < (1048576*4); j++, i+=4096)
    {
        pg->page_tbl_ent[j] = i  +PG_P + PG_RWW; /*可读可写，在物理内存里, 用户态不能访问*/
    }
    if (j != 1024) { panic("%s %d: wo kao!\n", __FILE__, __LINE__); }

    //设置用户态每一个页表项
    for (i = 0, j = 0; i < (PROC_SPACE+4097); j++, i+=4096)
    {
        pg->page_tbl_ent[1024+j] = (user_space_org+i) + PG_USU+PG_P +  PG_RWW; /*可读可写，在物理内存里，用户态可以访问*/
    }
    //设置两个页目录项
    pg->page_dir_ent[0] = (uint32_t)&(pg->page_tbl_ent[0]) + PG_P  + PG_RWW; /*这里一开始没有加uint32_t强制转换，搞得老子折腾了1整天 :( */
    pg->page_dir_ent[1] = (uint32_t)&(pg->page_tbl_ent[1024]) + PG_P + PG_USU + PG_RWW;


    return 0;
}
