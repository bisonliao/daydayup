/*
假设你有两个数组，一个长一个短，短的元素均不相同。找到长数组中包含短数组所有的元素的最短子数组，其出现顺序无关紧要。
返回最短子数组的左端点和右端点，如有多个满足条件的子数组，返回左端点最小的一个。若不存在，返回空数组。

示例 1:
输入:
big = [7,5,9,0,2,1,3,5,7,9,1,1,5,8,8,9,7]
small = [1,5,9]
输出: [7,10]

示例 2:
输入:
big = [1,2,3]
small = [4]
输出: []
=========================
我的思路：
1、对small里的每个元素，找出它在big里的位置，存储为vector。假设small有3个元素，就有3个vector，且每个vector里的下标都是从小到大排序好的。
2、把这三个vector的位置求并集并从小到大排序存储为一个总的vector。
3、对总的vector里每个位置p：在三个分的vector里个找出比p大于或者等于的第一个元素，这些元素放在一起，就是一个最短子数组的可能值
4、对上述可能的最短子数组，标记最短的那个子数组的上下界
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <iterator>
#include <algorithm>


using namespace std;

static void freeVectorInIndex(map<int, vector<int>* > & index)
{
    map<int, vector<int>* >::iterator it;
    for(it = index.begin(); it != index.end(); it++)
    {
        if (it->second != NULL)
        {
            free(it->second);
            it->second = NULL;
        }
    }
}

int findShortestContainer(const vector<int> & big, const vector<int> &small, int & begin, int &end)
{
    vector< vector<int> > positions;
    map<int, vector<int>*> index;
    int i;
    // 对big进行倒排索引，方便后面在big里查找small的各个元素
    for (i = 0; i < big.size(); ++i)
    {
        map<int, vector<int>* >::iterator it = index.find(big.at(i));
        if (it == index.end())
        {
            vector<int>*p = new vector<int>();
            p->push_back(i);
            index.insert(std::pair<int, vector<int>*>(big.at(i), p));
        }
        else
        {
            it->second->push_back(i);
        }
    }
    //对每个small的元素，查找其在big里的位置，存储在vector里。且
    vector<int> posInOneLine;
    for (i = 0; i < small.size(); ++i)
    {
        map<int, vector<int>* >::iterator it = index.find(small.at(i));
        if (it == index.end())// small[i] does NOT exist in big
        {
            freeVectorInIndex(index);
            return -1;
        }
        positions.push_back(*it->second);

        vector<int>::iterator it2;
        for (it2 = it->second->begin(); it2 != it->second->end(); it2++)
        {
            posInOneLine.push_back(*it2);
        }
    }
    std::sort(posInOneLine.begin(), posInOneLine.end());
    //找最小区间
    vector<int>::iterator it3;
    int narrowest = big.size();
    for (it3 = posInOneLine.begin(); it3 != posInOneLine.end(); ++it3)
    {
        int i;
        int left = *it3, right = *it3;
        bool completeChoice = true; // 是否是一个完整的可选的的字串
        for (i = 0; i < small.size(); ++i)
        {
            vector<int>::iterator  it4 = std::lower_bound(positions[i].begin(), positions[i].end(), *it3);
            if (it4 == positions[i].end() )
            {
                completeChoice = false;
                break;
            }
            if (*it4 > right)
            {
                right = *it4;
            }
        }
        if (!completeChoice)
        {
            continue;
        }
        if ( (right-left+1) < narrowest)
        {
            narrowest = right-left+1;
            begin = left;
            end = right;
        }
    }
    freeVectorInIndex(index);

    return 0;

}
int main()
{
    vector<int> big = {7,5,9,0,2,1,3,5,7,9,1,1,5,8,8,9,7};
    vector<int> small = {1,9,5};
    int begin, end;
    if (findShortestContainer(big, small, begin, end) == 0)
    {
        printf("%d,%d\n", begin, end);
    }
    return 0;
}
