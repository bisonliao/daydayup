/*
����һ���� 0 �� 1 ��ɵľ��� mat?�������һ����С��ͬ�ľ�������ÿһ�������� mat �ж�Ӧλ��Ԫ�ص������ 0 �ľ��롣

��������Ԫ�ؼ�ľ���Ϊ 1 ��
���磺
���룺mat = [[0,0,0],[0,1,0],[1,1,1]]
�����[[0,0,0],[0,1,0],[1,2,1]]
*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <algorithm>
#include <stdint.h>
#include <set>

using namespace std;

struct neighbour_t
{
    uint32_t x;
    uint32_t y;
    uint32_t distance;
};

set<uint64_t> g_checked; // �����Ѿ������ܱ��ĸ��ھӵĵ㣬���� x yƴ�ӳ�һ��64bit������

int saveChecked(uint32_t x, uint32_t y)
{
    uint64_t a = x, b = y;
    uint64_t toSave = (a << 32) | b;
    g_checked.insert(toSave);
    return 0;
}
bool hasChecked(uint32_t x, uint32_t y)
{
    uint64_t a = x, b = y;
    uint64_t toSave = (a << 32) | b;
    if (g_checked.find(toSave) != g_checked.end())
    {
        return true;
    }
    return false;
}
// �������i jΪ���ĵ��ĸ��ھӣ��ó�����Ϊi j��Ԫ�صľ��룬����ò������Ͱ��ھ�ѹջ��neighbours��������
int check4neighbours(uint32_t distance, const uint32_t *m,  uint32_t i, uint32_t j, uint32_t width, deque<neighbour_t> & neighbours)
{
    if (i > 0) 
    { 
        uint32_t x = i -1;
        uint32_t y = j;
        if (m[x*width + y] == 0) return distance;
        else 
        {
            if (!hasChecked(x, y))
            {
                neighbour_t n;
                n.x = x;
                n.y = y;
                n.distance = distance;
                neighbours.push_back(n);
            }
        }
    }
    if (j > 0) 
    { 
        uint32_t x = i;
        uint32_t y = j - 1;
        if (m[x*width + y] == 0) return distance;
        else 
        {
            if (!hasChecked(x, y))
            {
                neighbour_t n;
                n.x = x;
                n.y = y;
                n.distance = distance;
                neighbours.push_back(n);
            }
        }
    }
    if (i < width-1) 
    { 
        uint32_t x = i +1 ;
        uint32_t y = j;
        if (m[x*width + y] == 0) return distance;
        else 
        {
            if (!hasChecked(x, y))
            {
                neighbour_t n;
                n.x = x;
                n.y = y;
                n.distance = distance;
                neighbours.push_back(n);
            }
        }
    }
    if (j < width-1) 
    { 
        uint32_t x = i;
        uint32_t y = j + 1;
        if (m[x*width + y] == 0) return distance;
        else 
        {
            if (!hasChecked(x, y))
            {
                neighbour_t n;
                n.x = x;
                n.y = y;
                n.distance = distance;
                neighbours.push_back(n);
            }
        }
    }
    return -1;
}

int updateMatrix(const uint32_t  *m, uint32_t * u,  uint32_t width)
{
    uint32_t i, j;
    for (i = 0; i < width; ++i)
    {
        for (j = 0; j < width; ++j)
        {
            if (m[i*width + j] == 0) // ����Լ��������0�� �ǵ�0�ľ�����0
            {
                u[i*width + j] = 0;
            }
            else 
            // �����Ҫ�����Χ��4���ھӣ�����ھ�����һ��Ϊ0�����ɵó�Ԫ�أ�i��j����0 �ľ��룬����ͼ�������ھӵ��ھӣ�
            // �ھӼ���˳�����Ƚ��ȳ���һȦһȦ���ң�ȷ���ó�Ԫ�أ�i�� j����0����С���롣
            {
                deque<neighbour_t> neighbours;
                uint32_t distance;
                uint32_t x,y;
                x = i;
                y = j;
                for (distance = 1; distance < 10; ++distance)
                {
                    saveChecked(x, y);
                    int d = check4neighbours(distance, m, x, y, width, neighbours);
                    if (d >= 0)
                    {
                        u[i*width + j] = d;
                        break;
                    }
                    else
                    {
                        if (neighbours.size() < 1) { fprintf(stderr, "invalid!\n"); return -1;}
                        neighbour_t n = neighbours.front();
                        neighbours.pop_front();
                        distance = n.distance;
                        x = n.x;
                        y = n.y;
                    }
                }
            }

        }
    }
    return 0;
}

int main()
{
    {
        uint32_t m[] = {0, 0, 0, 0, 1, 0, 1, 1, 1};
        uint32_t width = 3;
        uint32_t u[9];
        updateMatrix(m, u, width);
        for (int i = 0; i < sizeof(u) / sizeof(uint32_t); ++i)
        {

            if ((i % width) == 0)
            {
                printf("\n");
            }
            printf("%u ", u[i]);
        }
        printf("\n");
    }
    {
        uint32_t m[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        uint32_t width = 3;
        uint32_t u[9];
        updateMatrix(m, u, width);
        for (int i = 0; i < sizeof(u) / sizeof(uint32_t); ++i)
        {

            if ((i % width) == 0)
            {
                printf("\n");
            }
            printf("%u ", u[i]);
        }
        printf("\n");
    }
    
    return 0;
}
