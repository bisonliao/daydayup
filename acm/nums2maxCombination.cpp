/*
输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

例如：
输入: [10,2]
输出: "102"

输入: [3,30,34,5,9]
输出: "3033459"

===================
思路一：
把输入数组进行排序，比对a,b两个元素，看"ab"形成的整数小还是"ba"形成的整数小，哪个放在前面小，就应该把哪个放在前面
然后把数组里的数按字符串连接起来就是结果

思路二：
稍微复杂一点，桶排序。而且要不起字符串的长度，怎么补也是个技术活。
*/


#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <stdint.h>

using namespace std;


bool compar(const uint32_t & a,const uint32_t & b)
{
    char tmp[100];
    uint64_t v1, v2;

    snprintf(tmp, sizeof(tmp), "%u%u", a, b);
    v1 =  atoll(tmp);

    snprintf(tmp, sizeof(tmp), "%u%u", b, a);
    v2 =  atoll(tmp);
    
    if (v1 < v2)
    {
        return true;
    }
    return false;
}


int nums2maxCombination(vector<uint32_t> & nums, string &result)
{
    sort(nums.begin(), nums.end(), compar);
    result = "";
    vector<uint32_t>::const_iterator it ;
    for (it = nums.begin(); it != nums.end(); ++it)
    {
        char tmp[100];
        snprintf(tmp, sizeof(tmp), "%u", *it);
        result.append(string(tmp));
    }
    return 0;
}

int main()
{
    vector<uint32_t> a;
    string result;

    a.clear();
    a.push_back(9);
    a.push_back(3);
    a.push_back(5);
    a.push_back(30);
    a.push_back(34);
    nums2maxCombination(a, result);
    printf("%s\n", result.c_str());

    a.clear();
    a.push_back(2);
    a.push_back(10);
    
    nums2maxCombination(a, result);
    printf("%s\n", result.c_str());


    return 0;
}
