/*
给定一个由 0 和 1 组成的矩阵 mat?，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。
例如：
输入：mat = [[0,0,0],[0,1,0],[1,1,1]]
输出：[[0,0,0],[0,1,0],[1,2,1]]
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

set<uint64_t> g_checked; // 保存已经检查过周边四个邻居的点，坐标 x y拼接成一个64bit的整数

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
// 检查坐标i j为中心的四个邻居，得出坐标为i j的元素的距离，如果得不出，就把邻居压栈到neighbours里，继续检查
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
            if (m[i*width + j] == 0) // 如果自己本身就是0， 那到0的距离是0
            {
                u[i*width + j] = 0;
            }
            else 
            // 否则就要检查周围的4个邻居，如果邻居中有一个为0，即可得出元素（i，j）到0 的距离，否则就继续检查邻居的邻居，
            // 邻居检查的顺序是先进先出，一圈一圈的找，确保得出元素（i， j）到0的最小距离。
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
