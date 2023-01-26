/*
leetcode题：
给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。
candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是不同的。 
对于给定的输入，保证和为 target 的唯一组合数少于 150 个。

示例 1：
输入: candidates = [2,3,6,7], target = 7
输出: [[7],[2,2,3]]

示例 2：
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]

示例 3：
输入: candidates = [2], target = 1
输出: []

示例 4：
输入: candidates = [1], target = 1
输出: [[1]]

示例 5：
输入: candidates = [1], target = 2
输出: [[1,1]]
 
提示：
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
candidate 中的每个元素都是独一无二的。
1 <= target <= 500
================================================================================
我的思路是递归。
不断的把一个整数拆分为两个整数的和，然后分别求出这两个加数的解，然后join就得到了该整数的解
例如求解 整数 5的解，就是
1）求解 1的解，求解4的解，然后把1和4的解做join
2）求解 2的解，求解3的解，然后把2和3的解做join
3）把1）和2）的解做union

进一步的优化：
1） 会重复求解某个整数的解，使用缓存可以节约不必要的重复计算，也就是g_answerCache变量干的事情
2） 解会比较大，尤其是c语言的定长数组固定预留最大空间，所以动态分配到堆上，避免递归下快速耗尽宝贵的栈空间
3） 每求解得到一个整数的所有解，会有很多重复，把这些解要去重，也就是compressAnswers函数干的事情
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define ANSWER_MAX_LEN (1000) // 一个解的最大组成元素的个数
#define ANSWER_MAX_NUM (1000) // 一个target允许最多的解的个数
// answer[ANSWER_MAX_NUM][ANSWER_MAX_LEN]保存了一个target的所有解，一行一个解，该行的所有元素是构成该解的加数
#define ELEMENT_MAX (100) // 数组元素的上界

typedef struct 
{
    int answerCount;
    unsigned int (*answer)[ANSWER_MAX_LEN]; // 一个指针，二维数组的指针
} answer_t;



static answer_t  g_answerCach[ELEMENT_MAX] = { {0, NULL}}; // 缓存已经求解的答案，例如g_answerCache[3].answer如果不为NULL，就指向了target=3的所有解

static void freeAnswerCache()
{
    int i;
    for (i = 0; i < ELEMENT_MAX; ++i)
    {
        if (g_answerCach[i].answer != NULL)
        {
            free(g_answerCach[i].answer);
            g_answerCach[i].answer = NULL;
        }
    }
}

static int compareAnswer(const void * aa, const void * bb) // 比较两个单一的answer，有点像strcmp
{
    int i;
    const unsigned int *a = (const unsigned int*)aa;
    const unsigned int *b = (const unsigned int*)bb;
    for (i = 0; i < ANSWER_MAX_LEN; ++i)
    {
        if (a[i] == b[i])
        {
            continue;
        }
        if (a[i] > b[i])
        {
            return 1;
        }
        if (a[i] < b[i])
        {
            return -1;
        }
    }
    return 0;
}
static void compressAnswers(unsigned int (*answer)[ANSWER_MAX_LEN], int *count)
{
    if (*count < 1)
    {
        return;
    }
    qsort(answer, *count, sizeof(unsigned int)*ANSWER_MAX_LEN, compareAnswer);
    int i;
    int valid = 0;
    for (i = 1; i < *count; ++i)
    {
        if (compareAnswer(answer[i], answer[valid]) == 0) //第i个是重复的，忽略
        {
            continue;
        }
        else
        {
            ++valid;
            memcpy(answer+valid, answer+i, sizeof(unsigned int)*ANSWER_MAX_LEN);
        }
    }
    *count = valid+1;
}

static int compar(const void *a, const void * b )//比较两个无符号整数
{
    unsigned int aa = *(unsigned int*)a;
    unsigned int bb = *(unsigned int*)b;
    if (aa < bb)
    {
        return -1;
    }
    if (aa > bb)
    {
        return 1;
    }
    return 0;
}

static int generateData(unsigned int* input, int input_len, unsigned int * target )//生成测试数据
{
    
    if (input == NULL || target == NULL)
    {
        return -1;
    }
    srandom(time(NULL));
    int i;
    for (i = 0; i < input_len; ++i)
    {
        while (1)
        {
            unsigned int r = random() % ELEMENT_MAX;
            if (r > 0  )
            {
                if (i > 0) // already exsits
                {
                    if (NULL != bsearch(&r, input, i, sizeof(unsigned int), compar))
                    {
                        continue;
                    }
                }
                input[i] =  r;
                qsort(input, i+1, sizeof(unsigned int), compar);
                break;
            }
        }
        
    }
    qsort(input, input_len, sizeof(unsigned int), compar);
    *target = (random() % ELEMENT_MAX) * 4;
    return 0;
}
// 解题函数
static int findAnswer(const unsigned int* input, 
                    int input_len, 
                    unsigned int target, 
                    unsigned int answer[][ANSWER_MAX_LEN])
{
    int ansCount = 0;
    if (bsearch(&target, input, input_len, sizeof(unsigned int), compar) != NULL)
    {
        answer[ansCount][0] = target;
        ansCount++;
    }
    if (target <= 1)
    {
        return ansCount;
    }

    int i;
    for (i = 1; i <= target / 2; i++)
    {
        // 动态分配二维数组，用来保存两个加数的答案，即subAnswerLeft[ANSWER_MAX_NUM][ANSWER_MAX_LEN]
        const int totalSize = sizeof(unsigned int)*ANSWER_MAX_LEN*ANSWER_MAX_NUM;
        unsigned int (*subAnswerLeft)[ANSWER_MAX_LEN] = NULL;
        unsigned int (*subAnswerRight)[ANSWER_MAX_LEN] = NULL;
        
        int ansCntLeft;
        int ansCntRight;
        unsigned int targetLeft = i;
        unsigned int targetRight = target - i;

        if (g_answerCach[targetLeft].answer != NULL) // 已经缓存了答案
        {
            subAnswerLeft = g_answerCach[targetLeft].answer;
            ansCntLeft = g_answerCach[targetLeft].answerCount;
        }
        else
        {
            subAnswerLeft = (unsigned int (*)[ANSWER_MAX_LEN])malloc(totalSize);
            memset(subAnswerLeft, 0, totalSize);
            ansCntLeft  = findAnswer(input, input_len, targetLeft, subAnswerLeft);       
            if (ansCntLeft < 0)
            {
                free(subAnswerLeft);
                return -1;
            }
            if (targetLeft > 1)
                printf("find %d answer for target %u\n", ansCntLeft, targetLeft);
            //同时缓存起来
            g_answerCach[targetLeft].answerCount = ansCntLeft;
            g_answerCach[targetLeft].answer = subAnswerLeft;
        }
        if (g_answerCach[targetRight].answer != NULL) // 已经缓存了答案
        {
            subAnswerRight = g_answerCach[targetRight].answer;
            ansCntRight = g_answerCach[targetRight].answerCount;
        }
        else
        {
            subAnswerRight = (unsigned int(*)[ANSWER_MAX_LEN])malloc(totalSize);
            memset(subAnswerRight, 0, totalSize);
            ansCntRight = findAnswer(input, input_len, targetRight, subAnswerRight);
            if ( ansCntRight < 0)
            {
                free(subAnswerRight);
                return -1;
            }
            if (targetRight > 1)
                printf("find %d answer for target %u\n", ansCntRight, targetRight);
            //同时缓存起来
            g_answerCach[targetRight].answerCount = ansCntRight;
            g_answerCach[targetRight].answer = subAnswerRight;
        }
        
        if (ansCntLeft == 0 || ansCntRight == 0)
        {
            continue;
        }
        
        // join together
        int j, k;
        
        for (j = 0; j < ansCntLeft; ++j)
        {
            for (k = 0; k < ansCntRight; ++k)
            {
                if (ansCount >= ANSWER_MAX_NUM)
                {
                    printf("answer max num is not enought!\n");
                    return -1;
                }
                int answerLen = 0;
                int m;
                for (m = 0; m < ANSWER_MAX_LEN; ++m)
                {
                    if (subAnswerLeft[j][m] > 0)
                    {
                        if (answerLen >= ANSWER_MAX_LEN)
                        {
                            printf("answer max len is not enough!\n");
                            return -1;
                        }
                        answer[ansCount][answerLen] = subAnswerLeft[j][m];
                        answerLen++;
                    }
                    else
                    {
                        break;
                    }
                }
                for (m = 0; m < ANSWER_MAX_LEN; ++m)
                {
                    if (subAnswerRight[k][m] > 0)
                    {
                        if (answerLen >= ANSWER_MAX_LEN)
                        {
                            printf("answer max len is not enough!\n");
                            return -1;
                        }
                        answer[ansCount][answerLen] = subAnswerRight[k][m];
                        answerLen++;
                    }
                    else
                    {
                        break;
                    }
                }
                qsort(answer[ansCount], answerLen, sizeof(unsigned int), compar);
                ansCount++;
            }
        }
    }
    compressAnswers(answer, &ansCount);
    
    return ansCount;
}


int main()
{
    unsigned int input[10] = { 2, 3, 4, 5, 6 ,  7, 8, 9, 10, 11};
    unsigned int target = 16;

    //generateData(input, sizeof(input)/sizeof(input[0]), &target);
    int i;
    printf("input:");
    for (i = 0; i < sizeof(input) / sizeof(input[0]); ++i)
    {
        printf("%u ", input[i]);
    }
    printf("\n");
    printf("target: %u\n", target);

    unsigned int answer[ANSWER_MAX_NUM][ANSWER_MAX_LEN];
    memset(answer, 0, sizeof(answer));
    int answerNumber = findAnswer(input, sizeof(input) / sizeof(input[0]), target, answer);
    printf("find %d answer for target %u\n", answerNumber, target);
    for (i = 0; i < answerNumber; ++i)
    {
        for (int j=0 ; j < ANSWER_MAX_LEN; ++j)
        {
            if (answer[i][j] > 0)
            {
                printf("%u ", answer[i][j]);
            }
            else
            {
                break;
            }
        }
        printf("\n");
    }
    freeAnswerCache();
    return 0;


}
