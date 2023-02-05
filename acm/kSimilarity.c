/*
����ĳЩ�Ǹ����� k ��������� s1 ��������ĸ��λ��ǡ�� k �Σ��ܹ�ʹ����ַ������� s2 ������Ϊ�ַ��� s1 �� s2 �� ���ƶ�Ϊ k ��
����������ĸ��λ�� s1 �� s2 ������ s1 �� s2 �����ƶ� k ����Сֵ��

���磺
���룺s1 = "abc", s2 = "bca"
�����2

�����Ŀ�Ҳ�̫�ᣬֻ����ֱ�ӵ���K����������С��K

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



