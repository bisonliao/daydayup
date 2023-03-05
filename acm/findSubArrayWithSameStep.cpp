/*
给你一个整数数组 nums ，返回 nums 中所有 等差子序列
例如输入[1,3,4,5,7,8,12]，它的等差子序列有：
1,3,5,7
3,4,5
4,8,12
1,4,7
=====================
思路：对每一对下标i，j，唯一确定一个可能的等差数列(步长和其中的两个元素都确定了)，与array求交集，就得到array的一个等差数列，如果长度大于3，就是一个解
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

typedef vector<int32_t> subscripts_t;//下标

map<int32_t, subscripts_t> g_index;//根据值，可以查询出在数组里出现过该值的下标
set<int64_t> g_hasFound; //为了快速去重，把已经出现的等差数列的相邻元素都放进来

int32_t prepare(const vector<int32_t> &array) // 预处理，建立索引
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
// 把元素array[i]和array[j]拓展成一个等差数列，并找出与array的交集，形成一个array的子数列，存放在subArray里
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
// 这个子数列找到了，写入快速去重的索引
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
//查找该子数列是否找到过
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
    // 对每一个i，j，唯一确定一个可能的等差数列，与array求交集，就得到array的一个等差数列，如果长度大于3，就是一个解
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
