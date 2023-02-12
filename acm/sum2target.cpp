/*
����һ����ѡ�˱�ŵļ���candidates��һ��Ŀ����target���ҳ�candidates�����п���ʹ���ֺ�Ϊtarget����ϡ�
candidates�е�ÿ��������ÿ�������ֻ��ʹ��һ�Ρ�
ע�⣺�⼯���ܰ����ظ�����ϡ�

˼·��

*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <time.h>

using namespace std;

class compareIntVector
{
public:
    bool operator()(const vector<int> &a, const vector<int> &b)
    {
        if (a.size() < b.size())
        {
            return true;
        }
        else if (a.size() > b.size())
        {
            return false;
        }
        int i;
        for (i = 0; i < a.size(); ++i)
        {
            if (a[i] == b[i])
            {
                continue;
            }
            if (a[i] < b[i])
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        return false;
    }
};

int findTarget(const int * candidate, int len, int start, int target, set< vector<int>, compareIntVector> & index)
{
    if (candidate == NULL || start >= len )
    {
        return -1;
    }
    
    if (candidate[start] == target)
    {
        vector<int> result;
        result.push_back(target);
        index.insert(result);
        //���ﲻӦ��ֱ�ӷ��أ�������[7, 1, 6]����7����һ�������㣬 ��1+6Ҳ���㣬�����Ѿ������һ��Ԫ��
    }
    if (start + 1 == len)
    {
        return 0;
    }

    // Ҫôȡ��һ��Ԫ�أ�Ҫô��ȡ

    // ȡ��һ��Ԫ�ص������
    if (candidate[start] < target)
    {
        set< vector<int>, compareIntVector> index2;
        if (findTarget(candidate, len, start+1, target-candidate[start], index2) == 0)
        {
            set< vector<int>, compareIntVector>::iterator it;
            for (it = index2.begin(); it != index2.end(); ++it)
            {
                vector<int> a = *it;
                a.push_back(candidate[start]);
                sort(a.begin(), a.end());
                index.insert(a);
            }
        }
    }
    // ��ȡ��һ��Ԫ�ص������
    set<vector<int>, compareIntVector> index2;
    if (findTarget(candidate, len, start + 1, target , index2) == 0)
    {
        set<vector<int>, compareIntVector>::iterator it;
        for (it = index2.begin(); it != index2.end(); ++it)
        {
            vector<int> a = *it;
            sort(a.begin(), a.end());
            index.insert(a);
        }
    }
    return 0;
}
int genData(int * candidate, int len, int *target)
{
    
    int i;
    for (i = 0; i < len; ++i)
    {
        candidate[i] = random() % 50+1;
    }
    *target = random() % 30+1;
    return 0;
}

int main()
{
    srandom(time(NULL));
    int candidates[100] = {10,1,2,7,6,1,5};
    int target = 8;
    int len = 7;
    int cnt = 0;
    do
    {
        printf("target:%d\n", target);
        printf("candidates:");
        int i;
        for (i = 0; i < len; ++i)
        {
            printf("%d ", candidates[i]);
        }
        printf("\n-----------\n");

        set<vector<int>, compareIntVector> index;
        if (findTarget(candidates, len, 0, target, index) == 0)
        {
            set<vector<int>, compareIntVector>::iterator it;
            for (it = index.begin(); it != index.end(); ++it)
            {
                int i;
                for (i = 0; i < it->size(); i++)
                {
                    printf("%d ", it->at(i));
                }
                printf("\n");
            }
        }
        else
        {
            printf("no match\n");
        }
        printf("\n");
        genData(candidates, sizeof(candidates)/sizeof(int), &target);
        len = sizeof(candidates)/sizeof(int);
        cnt++;
    } while (cnt < 5);
    
    
    return 0;
}
