/*
给你一个整数数组nums和一个整数k，请你返回其中出现频率前k高的元素。你可以按 任意顺序 返回答案。
例如：
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <map>
#include <algorithm>
#include <vector>
#include <time.h>

using namespace std;

class freq_t
{
public:
    int freq;
    int val;

} ;
bool operator<(const freq_t &a, const freq_t&b)
{
    return a.freq > b.freq;
}

int frequencyTopK(const int * nums, int len, int k, vector<int> & result)
{
    if (nums == NULL )
    {
        return -1;
    }
    map<int, int> frequency;
    int i;
    for (i = 0; i < len; ++i)
    {
        map<int, int>::iterator it = frequency.find(nums[i]);
        if (it == frequency.end())
        {
            frequency.insert(pair<int, int>(nums[i], 1));
        }
        else
        {
            it->second++;
        }
    }
    map<int, int>::iterator it;
    vector<freq_t> freqList;
    for (it = frequency.begin(); it != frequency.end(); ++it)
    {
        freq_t f;
        f.freq = it->second;
        f.val = it->first;

        freqList.push_back(f);
    }
    if (k > freqList.size())
    {
        return -1;
    }
    std::sort(freqList.begin(), freqList.end());
    // debug begin
    int total = 0;
    for (i = 0; i < freqList.size(); ++i)
    {
        printf("<%d, %d> ", freqList[i].val, freqList[i].freq);
        total += freqList[i].freq;
    }
    printf("\ntotal=%d\n\n", total);
    // debug end
    result.clear();
    for (i = 0; i < k; ++i)
    {
        result.push_back(freqList[i].val);
    }
    return 0;    
}
int genData(int * nums, int len)
{
    int i;
    for (i = 0; i < len; ++i)
    {
        nums[i] = random()% 1000;
    }
    return 0;
}
int main()
{
    srandom(time(NULL));
    int nums[10000];
    int len = sizeof(nums)/sizeof(int);
    genData(nums, len);
    vector<int> result;
    frequencyTopK(nums, len, 20, result);
    int i;
    for (i = 0; i < 20; ++i)
    {
        printf("%d, ", result[i]);
    }
    printf("\n");
    return 0;

}
