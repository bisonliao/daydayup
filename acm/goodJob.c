/*
你打算利用空闲时间来做兼职工作赚些零花钱。

这里有?n?份兼职工作，每份工作预计从?startTime[i]?开始到?endTime[i]?结束，报酬为?profit[i]。

给你一份兼职工作表，包含开始时间?startTime，结束时间?endTime?和预计报酬?profit?三个数组，请你计算并返回可以获得的最大报酬。

注意，时间上出现重叠的 2 份工作不能同时进行。

如果你选择的工作在时间?X?结束，那么你可以立刻进行在时间?X?开始的下一份工作。
*/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>

int isOverlap(uint32_t start1, uint32_t start2, uint32_t end1, uint32_t end2)
{
   
    if (end1 <= start2 || end2 <= start1 )
    {
        return 0;
    }
    return 1;
}

int maxProfit(const uint32_t * startTime, const uint32_t * endTime, const int * profit, const uint32_t jobCount)
{
    int j;
    printf("call maxProfit:");
    for (j = 0; j < jobCount;j++)
    {
        printf("<%u, %u> ", startTime[j], endTime[j]);
    }
    printf("\n");
    if (jobCount == 1)
    {
        return profit[0];
    }
    if (jobCount == 0)
    {
        printf("invalid\n");
        return 0;
    }
    //取第一个工作，则时间与它冲突的都不能取
    uint32_t * start = (uint32_t *)malloc( (jobCount-1) * sizeof(uint32_t));
    uint32_t * end =   (uint32_t *)malloc( (jobCount-1) * sizeof(uint32_t));
    int * profit2 =    (uint32_t *)malloc( (jobCount-1) * sizeof(int));
    if (start == NULL || end == NULL || profit2 == NULL)
    {
        fprintf(stderr, "failed to malloc()\n");
        return -1;
    }
    int i;
    uint32_t cnt = 0;
    for (i = 1; i < jobCount; ++i)
    {
        
        if (!isOverlap(startTime[0], startTime[i], endTime[0], endTime[i])) // no overlap with job#0
        {
            start[cnt] = startTime[i];
            end[cnt] = endTime[i];
            profit2[cnt] = profit[i];
            cnt++;
        }
    }
    int value1 = profit[0];
    if (cnt > 0)
    {
        value1 += maxProfit(start, end, profit2, cnt);
    }
    
    free(start);
    free(end);
    free(profit2);
    
    


    //不取第一个工作，其他工作都取
    int value2 = maxProfit(startTime+1, endTime+1, profit+1, jobCount-1);
    

    if (value1 > value2)
    {
        return value1;
    }
    else
    {
        return value2;
    }

    
}
int main()
{
#if 0
    uint32_t startTime[] = {1,2,3,4,6};
    uint32_t endTime[] = {3,5,10,6,9};
    int profit[] = {20,20,100,70,60};
    uint32_t jobCount = 5;
    printf("%d\n", maxProfit(startTime, endTime, profit, jobCount));
#else
    uint32_t startTime[] = {1,2,3,3};
    uint32_t endTime[] = {3,4,5,6};
    int profit[] = {50,10,40,70};
    uint32_t jobCount = 4;
    printf("%d\n", maxProfit(startTime, endTime, profit, jobCount));
#endif

    return 0;
}
