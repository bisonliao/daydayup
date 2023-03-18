/*
����һ���Ǹ��������飬����������������ƴ�������ų�һ��������ӡ��ƴ�ӳ���������������С��һ����

���磺
����: [10,2]
���: "102"

����: [3,30,34,5,9]
���: "3033459"

===================
˼·һ��
����������������򣬱ȶ�a,b����Ԫ�أ���"ab"�γɵ�����С����"ba"�γɵ�����С���ĸ�����ǰ��С����Ӧ�ð��ĸ�����ǰ��
Ȼ���������������ַ��������������ǽ��

Ϊʲô����㷨�ǶԵ��أ��͵�֤������������⣬����

��֪ ab < ba, bc < cb���� abc����С�����ӷ�ʽ��
�⣺
a��b��c����Ԫ�صĸ���������ϣ����ǲ���abc�����������϶�����abc��һ�֣�����֪�ɵã�
acb > a(bc) = abc
bac > (ab)c = abc
bca = b(c)a > a(c)b = acb > abc
cab = c(a)b > b(a)c = bac > abc
cba > c(ab) > abc



˼·����
��΢����һ�㣬Ͱ���򡣶���Ҫ�����ַ����ĳ��ȣ���ô��Ҳ�Ǹ������
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
