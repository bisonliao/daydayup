/*
leetcode�⣺
����һ�����ظ�Ԫ�ص����������� candidates ��һ�������� target ���ҳ� candidates �����п���ʹ���ֺ�ΪĿ���� target ��Ψһ��ϡ�
candidates �е����ֿ����������ظ���ѡȡ���������һ����ѡ����������ͬ������������ǲ�ͬ�ġ� 
���ڸ��������룬��֤��Ϊ target ��Ψһ��������� 150 ����

ʾ�� 1��
����: candidates = [2,3,6,7], target = 7
���: [[7],[2,2,3]]

ʾ�� 2��
����: candidates = [2,3,5], target = 8
���: [[2,2,2,2],[2,3,3],[3,5]]

ʾ�� 3��
����: candidates = [2], target = 1
���: []

ʾ�� 4��
����: candidates = [1], target = 1
���: [[1]]

ʾ�� 5��
����: candidates = [1], target = 2
���: [[1,1]]
 
��ʾ��
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
candidate �е�ÿ��Ԫ�ض��Ƕ�һ�޶��ġ�
1 <= target <= 500
================================================================================
�ҵ�˼·�ǵݹ顣
���ϵİ�һ���������Ϊ���������ĺͣ�Ȼ��ֱ���������������Ľ⣬Ȼ��join�͵õ��˸������Ľ�
������� ���� 5�Ľ⣬����
1����� 1�Ľ⣬���4�Ľ⣬Ȼ���1��4�Ľ���join
2����� 2�Ľ⣬���3�Ľ⣬Ȼ���2��3�Ľ���join
3����1����2���Ľ���union

��һ�����Ż���
1�� ���ظ����ĳ�������Ľ⣬ʹ�û�����Խ�Լ����Ҫ���ظ����㣬Ҳ����g_answerCache�����ɵ�����
2�� ���Ƚϴ�������c���ԵĶ�������̶�Ԥ�����ռ䣬���Զ�̬���䵽���ϣ�����ݹ��¿��ٺľ������ջ�ռ�
3�� ÿ���õ�һ�����������н⣬���кܶ��ظ�������Щ��Ҫȥ�أ�Ҳ����compressAnswers�����ɵ�����
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define ANSWER_MAX_LEN (1000) // һ�����������Ԫ�صĸ���
#define ANSWER_MAX_NUM (1000) // һ��target�������Ľ�ĸ���
// answer[ANSWER_MAX_NUM][ANSWER_MAX_LEN]������һ��target�����н⣬һ��һ���⣬���е�����Ԫ���ǹ��ɸý�ļ���
#define ELEMENT_MAX (100) // ����Ԫ�ص��Ͻ�

typedef struct 
{
    int answerCount;
    unsigned int (*answer)[ANSWER_MAX_LEN]; // һ��ָ�룬��ά�����ָ��
} answer_t;



static answer_t  g_answerCach[ELEMENT_MAX] = { {0, NULL}}; // �����Ѿ����Ĵ𰸣�����g_answerCache[3].answer�����ΪNULL����ָ����target=3�����н�

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

static int compareAnswer(const void * aa, const void * bb) // �Ƚ�������һ��answer���е���strcmp
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
        if (compareAnswer(answer[i], answer[valid]) == 0) //��i�����ظ��ģ�����
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

static int compar(const void *a, const void * b )//�Ƚ������޷�������
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

static int generateData(unsigned int* input, int input_len, unsigned int * target )//���ɲ�������
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
// ���⺯��
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
        // ��̬�����ά���飬�����������������Ĵ𰸣���subAnswerLeft[ANSWER_MAX_NUM][ANSWER_MAX_LEN]
        const int totalSize = sizeof(unsigned int)*ANSWER_MAX_LEN*ANSWER_MAX_NUM;
        unsigned int (*subAnswerLeft)[ANSWER_MAX_LEN] = NULL;
        unsigned int (*subAnswerRight)[ANSWER_MAX_LEN] = NULL;
        
        int ansCntLeft;
        int ansCntRight;
        unsigned int targetLeft = i;
        unsigned int targetRight = target - i;

        if (g_answerCach[targetLeft].answer != NULL) // �Ѿ������˴�
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
            //ͬʱ��������
            g_answerCach[targetLeft].answerCount = ansCntLeft;
            g_answerCach[targetLeft].answer = subAnswerLeft;
        }
        if (g_answerCach[targetRight].answer != NULL) // �Ѿ������˴�
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
            //ͬʱ��������
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
