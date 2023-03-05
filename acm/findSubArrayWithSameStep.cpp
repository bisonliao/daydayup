/*
����һ���������� nums ������ nums ������ �Ȳ�������
��������[1,3,4,5,7,8,12]�����ĵȲ��������У�
1,3,5,7
3,4,5
4,8,12
1,4,7
=====================
˼·����ÿһ���±�i��j��Ψһȷ��һ�����ܵĵȲ�����(���������е�����Ԫ�ض�ȷ����)����array�󽻼����͵õ�array��һ���Ȳ����У�������ȴ���3������һ����
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <stack>
#include <map>
#include <deque>
#include <algorithm>
#include <stdint.h>
#include <set>

using namespace std;

typedef vector<int32_t> subscripts_t;//�±�

map<int32_t, subscripts_t> g_index;//����ֵ�����Բ�ѯ������������ֹ���ֵ���±�
set<int64_t> g_hasFound; //Ϊ�˿���ȥ�أ����Ѿ����ֵĵȲ����е�����Ԫ�ض��Ž���

int32_t prepare(const vector<int32_t> &array) // Ԥ������������
{
    int32_t len = array.size();
    int32_t i;
    for (i = 0; i < len; ++i)
    {
        if (g_index.find(array[i]) == g_index.end())
        {
            subscripts_t subs;
            subs.push_back(i);
            g_index.insert(pair<int32_t, subscripts_t>(array[i], subs));
        }
        else
        {
            g_index.find(array[i])->second.push_back(i);
        }
    }
    return 0;
}
// ��Ԫ��array[i]��array[j]��չ��һ���Ȳ����У����ҳ���array�Ľ������γ�һ��array�������У������subArray��
int32_t checkSubArray(int32_t i, int32_t j, const vector<int32_t> &array, deque<int32_t> &subArray)
{
    const int32_t step = array[j] - array[i];
    

    subArray.push_back(array[i]);
    subArray.push_back(array[j]);


    while (i > 0)
    {
        int32_t value = array[i] - step;
        if (g_index.find(value) != g_index.end())
        {
            subscripts_t & subscripts = g_index.find(value)->second;
            subscripts_t::const_reverse_iterator it;
            for (it = subscripts.rbegin(); it != subscripts.rend(); ++it)
            {
                if (*it < i)
                {
                    i = *it;
                    subArray.push_front(array[i]);
                    break;
                }
            }

        }
        else
        {
            break;   
        }
    }
    while (j < array.size()-1)
    {
        int32_t value = array[j] + step;
        if (g_index.find(value) != g_index.end())
        {
            subscripts_t & subscripts = g_index.find(value)->second;
            subscripts_t::const_iterator it;
            for (it = subscripts.begin(); it != subscripts.end(); ++it)
            {
                if (*it > j)
                {
                    j = *it;
                    subArray.push_back(array[j]);
                    break;
                }
            }

        }
        else
        {
            break;   
        }
    }
   
    return 0;

}
// ����������ҵ��ˣ�д�����ȥ�ص�����
int found(const deque<int32_t>  & subArray)
{
    int k;
    for (k = 0; k < subArray.size()-1; ++k)
    {
        int64_t a = subArray[k];
        int64_t b = subArray[k+1];
        int64_t toSave = (a << 32) | b;
        g_hasFound.insert(toSave);
    }
    return 0;
}
//���Ҹ��������Ƿ��ҵ���
bool hasAlreadyFound(const deque<int32_t>  & subArray)
{
    if (subArray.size() < 2) { return false;}
    int64_t a = subArray[0];
    int64_t b = subArray[1];
    int64_t toSave = (a << 32) | b;
    if (g_hasFound.find(toSave) != g_hasFound.end())
    {
        return true;
    }
    return false;
}

int32_t findSubArrayWithSameStep(const vector<int32_t> array, vector< deque<int32_t>> & result)
{
    int32_t len = array.size();
    int32_t i, j;
    // ��ÿһ��i��j��Ψһȷ��һ�����ܵĵȲ����У���array�󽻼����͵õ�array��һ���Ȳ����У�������ȴ���3������һ����
    for (i = 0; i < len-1; ++i) 
    {
        for (j = i +1; j < len; ++j)
        {
            deque<int32_t> subArray;
            checkSubArray(i, j, array, subArray);
            if (subArray.size() >= 3 && !hasAlreadyFound(subArray))
            {
                result.push_back(subArray);
                found(subArray);
            }
        }
    }
    return 0;
}
int32_t main()
{
    vector<int32_t> array = {1,3,4,5,7,8,12};
    vector< deque<int32_t>> result;
    prepare(array);
    findSubArrayWithSameStep(array, result);
    vector< deque<int32_t>>::const_iterator it;
    for (it = result.begin(); it != result.end(); ++it)
    {
        const deque<int32_t> & subArray = *it;
        int32_t i;
        for (i = 0; i < subArray.size(); ++i)
        {
            printf("%d,", subArray[i]);
        }
        printf("\n");
    }
}
