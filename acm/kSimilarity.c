/*
对于某些非负整数 k ，如果交换 s1 中两个字母的位置恰好 k 次，能够使结果字符串等于 s2 ，则认为字符串 s1 和 s2 的 相似度为 k 。
给你两个字母异位词 s1 和 s2 ，返回 s1 和 s2 的相似度 k 的最小值。

例如：
输入：s1 = "abc", s2 = "bca"
输出：2

这个题目我不太会，只会最直接的求K，不能求最小的K

*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

int kSimilar(char * s, const char * want)
{
    if (s == NULL || want == NULL || strlen(s) < 2 || strlen(want) < 2 || strlen(s) != strlen(want))
    {
        return -1;
    }
    int len = strlen(s);
    int i;
    int k = 0;
    for (i = 0; i < len-1; ++i)
    {
        if (s[i] != want[i])
        {
            char * p = strchr(s, want[i]);
            if (p == NULL)
            {
                return -1;
            }
            printf("swap %d and %d\n", i, p-(s+i));
            char tmp = s[i];
            s[i] = *p;
            *p = tmp;
            k++;
            
        }
    }
    if (s[len-1] == want[len-1])
    {
        return k;
    }
    return -1;
}
int shuffle(char *s)
{
    int len = strlen(s);
    srandom(time(NULL));

    int i, j;
    for (i = 0; i < 100; i++)
    {
        j = random() % len;
        char tmp = s[0];
        s[0] = s[j];
        s[j] = tmp;
    }
    return 0;
}
int main()
{
    char s[] = "abcdefghijklmnopqrstuvwxyz";
    char want[27];
    memcpy(want, s, sizeof(want));

    shuffle(want);
    int k = kSimilar(s, want);

    printf("%s\n", s);
    printf("%s\n", want);
    printf("%d\n", k);

    return 0;
}



