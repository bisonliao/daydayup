/*
���������������飬һ����һ���̣��̵�Ԫ�ؾ�����ͬ���ҵ��������а������������е�Ԫ�ص���������飬�����˳���޹ؽ�Ҫ��
����������������˵���Ҷ˵㣬���ж�����������������飬������˵���С��һ�����������ڣ����ؿ����顣

ʾ�� 1:
����:
big = [7,5,9,0,2,1,3,5,7,9,1,1,5,8,8,9,7]
small = [1,5,9]
���: [7,10]

ʾ�� 2:
����:
big = [1,2,3]
small = [4]
���: []
=========================
�ҵ�˼·��
1����small���ÿ��Ԫ�أ��ҳ�����big���λ�ã��洢Ϊvector������small��3��Ԫ�أ�����3��vector����ÿ��vector����±궼�Ǵ�С��������õġ�
2����������vector��λ���󲢼�����С��������洢Ϊһ���ܵ�vector��
3�����ܵ�vector��ÿ��λ��p���������ֵ�vector����ҳ���p���ڻ��ߵ��ڵĵ�һ��Ԫ�أ���ЩԪ�ط���һ�𣬾���һ�����������Ŀ���ֵ
4�����������ܵ���������飬�����̵��Ǹ�����������½�
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
    // ��big���е������������������big�����small�ĸ���Ԫ��
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
    //��ÿ��small��Ԫ�أ���������big���λ�ã��洢��vector���
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
    //����С����
    vector<int>::iterator it3;
    int narrowest = big.size();
    for (it3 = posInOneLine.begin(); it3 != posInOneLine.end(); ++it3)
    {
        int i;
        int left = *it3, right = *it3;
        bool completeChoice = true; // �Ƿ���һ�������Ŀ�ѡ�ĵ��ִ�
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
